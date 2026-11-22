"""Behaviour tests over isolated persistent state; no LLM mocks presented as live evidence."""
import json
import os
import tempfile

os.environ["NEXTDEAL_RUNTIME"] = tempfile.mkdtemp(prefix="nextdeal-tests-")
os.environ['NEXTDEAL_MAINTENANCE'] = '0'

import pytest
from fastapi.testclient import TestClient
from backend.db import connect, init, now
from backend.governance import match_pair, pipeline, quality_check
from backend.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def draft_case(client):
    ref = client.get('/api/clients?kind=corporate&limit=1').json()[0]['id']
    cid = client.post('/api/cases', json={"client_id": ref, "subject": "test", "message": "Please advise"}).json()['id']
    with connect() as db:
        d = db.execute("INSERT INTO drafts(case_id,version,body,evidence,quality,created_at) VALUES(?,1,?,'{}','{}',?)", (cid, "A manual test draft", now()))
        did = d.lastrowid
        db.execute("UPDATE cases SET status='awaiting_review' WHERE id=?", (cid,))
    return cid, did


def test_dispatch_requires_approval_and_is_idempotent(client):
    cid, did = draft_case(client)
    assert client.post(f'/api/drafts/{did}/dispatch').status_code == 409
    assert client.post(f'/api/drafts/{did}/decision', json={"action": "approve"}).status_code == 200
    a = client.post(f'/api/drafts/{did}/dispatch').json()
    b = client.post(f'/api/drafts/{did}/dispatch').json()
    assert a['id'] == b['id'] and b['reused']
    assert client.get(f'/api/cases/{cid}').json()['status'] == 'dispatched'


def test_edit_invalidates_approval_and_stale_draft(client):
    cid, did = draft_case(client)
    client.post(f'/api/drafts/{did}/decision', json={"action": "approve"})
    edit = client.post(f'/api/drafts/{did}/edit', json={"body": "Updated by advisor"}).json()
    assert edit['status'] == 'pending'
    assert client.post(f'/api/drafts/{did}/dispatch').status_code == 409
    assert client.post(f"/api/drafts/{edit['id']}/dispatch").status_code == 409
    assert client.post(f"/api/drafts/{edit['id']}/decision", json={"action": "approve"}).status_code == 200
    assert client.post(f"/api/drafts/{edit['id']}/dispatch").status_code == 200


def test_reply_supersedes_previous_approval(client):
    cid, did = draft_case(client)
    client.post(f'/api/drafts/{did}/decision', json={"action": "approve"})
    assert client.post(f'/api/cases/{cid}/messages', json={"body": "Actually I decline"}).status_code == 200
    assert client.post(f'/api/drafts/{did}/dispatch').status_code == 409


def test_reject_requires_reason(client):
    _, did = draft_case(client)
    assert client.post(f'/api/drafts/{did}/decision', json={"action": "reject"}).status_code == 400
    assert client.post(f'/api/drafts/{did}/decision', json={"action": "reject", "reason": "Unsupported coverage claim"}).status_code == 200


def test_pipeline_quarantine_retry_and_idempotence(client):
    bad = [{"id": "broken", "kind": "corporate", "label": "Broken", "age": -5}]
    result = pipeline(bad)
    assert result['status'] == 'quarantined'
    good = [{"id": "good", "kind": "corporate", "label": "Good"}]
    result = pipeline(good, True)
    assert result['status'] == 'accepted' and any(t['action'] == 'retry' for t in result['trace'])
    assert pipeline(good)['reused']
    with connect() as db:
        assert db.execute("SELECT COUNT(*) FROM imported_records WHERE id='corporate:good'").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM imported_records WHERE id='corporate:broken'").fetchone()[0] == 0


def test_no_cross_type_identity_merge_and_review_persistence(client):
    a = {"id": "one", "kind": "corporate", "label": "Same Name", "email": "same@example.test"}
    b = {**a, "id": "two", "kind": "individual"}
    assert match_pair(a, b)['decision'] == 'different_type'
    b['kind'] = 'corporate'
    assert client.post('/api/duplicates/check', json={"left": a, "right": b}).json()['decision'] == 'duplicate'
    review = client.get('/api/duplicates').json()[0]
    client.post(f"/api/duplicates/{review['id']}", json={"confirmed": True})
    with connect() as db:
        assert db.execute("SELECT COUNT(*) FROM identity_links").fetchone()[0] == 1


def test_recommendations_exclude_held_and_do_not_apply_to_individuals(client):
    ref = client.get('/api/clients?kind=corporate&limit=1').json()[0]['id']
    r = client.get(f'/api/clients/{ref}/recommendations').json()
    assert not set(r['held']) & {p['product'] for p in r['recommendations']}
    ref = client.get('/api/clients?kind=individual&limit=1').json()[0]['id']
    assert client.get(f'/api/clients/{ref}/recommendations').json()['recommendations'] == []


def test_historical_clock_and_no_unpaid_predictor_leak(client):
    assert client.post('/api/clock', json={"date": "2025-08-01"}).status_code == 200
    c = client.get('/api/renewals').json()[0]
    assert c['days_to_expiry'] == 31
    assert client.post('/api/clock', json={"date": "not-a-date"}).status_code == 422


def test_grounding_sources_have_provenance_and_unknown_question_abstains(client):
    r = client.get('/api/search', params={"q": "motor car insurance"}).json()
    assert r['passages'] and all(not p['contractual_conditions'] for p in r['passages'])
    assert client.get('/api/search', params={"q": "xyzzqvnonword"}).json()['passages'] == []


def test_quality_gate_rejects_type_and_date_failures():
    f = quality_check([{"id": "a", "kind": "unknown", "label": "A", "effective_date": "2025-03-01", "expiry_date": "2024-01-01"}])
    assert {x['rule'] for x in f} >= {'unknown_client_type', 'before_effective_date'}


def test_database_state_survives_reopen(client):
    cid, _ = draft_case(client)
    init()
    with connect() as db:
        assert db.execute("SELECT subject FROM cases WHERE id=?", (cid,)).fetchone()[0] == 'test'


def test_actions_blocked_during_replacement_generation(client):
    cid, did = draft_case(client)
    with connect() as db:
        db.execute("UPDATE cases SET status='processing' WHERE id=?", (cid,))
    assert client.post(f'/api/drafts/{did}/decision', json={"action": "approve"}).status_code == 409
    assert client.post(f'/api/cases/{cid}/messages', json={"body": "new reply"}).status_code == 409


def test_schedule_preparation_is_idempotent_and_does_not_send(client):
    client.post('/api/clock', json={'date': '2025-08-02'})
    first = client.post('/api/renewals/prepare-due').json()
    second = client.post('/api/renewals/prepare-due').json()
    assert first['created'] > 0 and second['created'] == 0
    for cid in first['case_ids']:
        assert client.get(f'/api/cases/{cid}').json()['status'] == 'received'


def test_failed_inference_is_recorded_without_fake_draft(client, monkeypatch):
    from backend import agent
    cid, _ = draft_case(client)
    def fail(*args, **kwargs):
        raise agent.AgentError('Injected unavailable Ollama')
    monkeypatch.setattr(agent, 'chat', fail)
    agent.run_case(cid)
    c = client.get(f'/api/cases/{cid}').json()
    assert c['status'] == 'failed' and 'Injected unavailable' in c['error']
    assert len(c['drafts']) == 1


def test_artifact_promotion_gate_preserves_minimum_quality():
    from backend.artifacts import promotion_gate
    assert not promotion_gate(.1, .6, 139)
    assert not promotion_gate(.8, .6, 0)
    assert promotion_gate(.65, .6, 139)


def test_manual_draft_is_explicit_and_needs_approval(client):
    cid, _ = draft_case(client)
    d = client.post(f'/api/cases/{cid}/manual-draft', json={'body': 'Advisor manual response'}).json()
    c = client.get(f'/api/cases/{cid}').json()
    assert c['drafts'][0]['evidence']['live'] is False
    assert client.post(f"/api/drafts/{d['id']}/dispatch").status_code == 409


def test_catalogue_search_cannot_borrow_another_product_family(client):
    result = client.get('/api/search', params={'q': 'MULTIRISQUES PROFESSIONNELLES CENTRALISE coverage details'}).json()
    assert result['passages']
    assert all('multirisque-professionnelle#' in p['chunk_id'] for p in result['passages'])
