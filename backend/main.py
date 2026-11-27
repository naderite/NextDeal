import json
import asyncio
import os
from contextlib import asynccontextmanager
from datetime import date
from typing import Literal

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .agent import chat, run_case
from .config import ARTIFACT_VERSION, CUSTOM_DATASET, OLLAMA_MODEL, OLLAMA_URL, ROOT
from .db import connect, decode, demo_date, event, init, now
from .governance import match_pair, pipeline
from .services import engine
from .review_metrics import draft_acceptance_metrics, human_review_metrics
from .artifacts import MANIFEST, refresh_artifacts


@asynccontextmanager
async def lifespan(app):
    init()
    engine()
    # A terminated worker must not leave cases permanently marked as running.
    with connect() as db:
        db.execute("UPDATE cases SET status='failed',error='Application restarted during processing; retry the case.' WHERE status='processing'")
    maintenance = asyncio.create_task(maintenance_loop()) if os.environ.get('NEXTDEAL_MAINTENANCE', '1') != '0' else None
    try:
        yield
    finally:
        if maintenance:
            maintenance.cancel()
            try:
                await maintenance
            except asyncio.CancelledError:
                pass


app = FastAPI(title="NextDeal advisor API", version="2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"], allow_methods=["*"], allow_headers=["*"])


async def maintenance_loop():
    """Local daily refresh while the app runs; durable date guard, no external cron."""
    await asyncio.sleep(5)
    while True:
        today = now()[:10]
        with connect() as db:
            row = db.execute("SELECT value FROM settings WHERE key='last_daily_check'").fetchone()
        if row is None or row[0] != today:
            try:
                with connect() as db:
                    records = [{"id": r['id'], "kind": r['kind'], "label": r['label']} for r in db.execute('SELECT id,kind,label FROM clients')]
                checked = await asyncio.to_thread(pipeline, records)
                if checked['status'] == 'accepted':
                    await asyncio.to_thread(refresh_artifacts)
                with connect() as db:
                    db.execute("INSERT INTO settings(key,value) VALUES('last_daily_check',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (today,))
                event('daily_quality_check', {"date": today, "status": checked['status'], "scope": "canonical client metadata; model refresh has separate feature/promotion gate"})
            except Exception as exc:
                event('daily_maintenance_failed', {'error': str(exc)[:500]})
        try:
            await asyncio.to_thread(prepare_due)
        except Exception as exc:
            event('renewal_scheduler_failed', {'error': str(exc)[:500]})
        await asyncio.sleep(60)


@app.exception_handler(ValueError)
async def value_error(request, exc):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.get("/api/health")
def health():
    try:
        response = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        response.raise_for_status()
        models = [m["name"] for m in response.json().get("models", [])]
        ollama = {"available": OLLAMA_MODEL in models, "models": models, "selected": OLLAMA_MODEL}
    except Exception as exc:
        ollama = {"available": False, "error": str(exc), "selected": OLLAMA_MODEL}
    return {"status": "ok", "version": ARTIFACT_VERSION, "ollama": ollama, "demo_date": demo_date(),
            "retrieval": "hybrid" if engine().dense else "bm25 bilingual", "source_type": "public product pages"}


@app.get("/api/clients")
def clients(q: str = "", kind: str = "", limit: int = 100):
    with connect() as db:
        rows = db.execute("SELECT * FROM clients WHERE (label LIKE ? OR id LIKE ?) AND (?='' OR kind=?) ORDER BY kind,id LIMIT ?",
                          (f"%{q}%", f"%{q}%", kind, kind, min(max(limit, 1), 1000))).fetchall()
    return [decode(r) for r in rows]


@app.get("/api/clients/{client_id}")
def client(client_id: str):
    try:
        return engine().client(client_id)
    except ValueError:
        raise HTTPException(404, "Client not found")


@app.get("/api/clients/{client_id}/recommendations")
def recommendations(client_id: str):
    return engine().recommendations(client_id)


@app.get("/api/clients/{client_id}/renewals")
def client_renewals(client_id: str):
    return engine().renewal(client_id)


@app.get("/api/renewals")
def renewals():
    return engine().contracts()


@app.post("/api/renewals/prepare-due")
def prepare_due():
    """Materialise due scheduled actions as reviewable cases; never approve or send automatically."""
    created = []
    day = demo_date()
    with connect() as db:
        db.execute('BEGIN IMMEDIATE')
        due = db.execute("SELECT renewal_schedule.*,contracts.client_id FROM renewal_schedule JOIN contracts ON contracts.id=renewal_schedule.contract_id WHERE renewal_schedule.status='scheduled' AND contact_date<=?", (day,)).fetchall()
        for r in due:
            c = db.execute("INSERT INTO cases(client_id,subject,created_at) VALUES(?,?,?)", (r['client_id'], 'Scheduled renewal · ' + r['contract_id'], now()))
            cid = c.lastrowid
            db.execute("INSERT INTO messages(case_id,role,body,created_at) VALUES(?,'system',?,?)", (cid, 'Historical renewal trigger: review this client renewal and prepare a reminder, offer or escalation for advisor approval.', now()))
            db.execute("UPDATE renewal_schedule SET status='case_prepared',case_id=? WHERE contract_id=?", (cid, r['contract_id']))
            created.append(cid)
    if created:
        event('renewals_prepared', {'demo_date': day, 'case_ids': created})
    return {'created': len(created), 'case_ids': created, 'note': 'Draft generation and advisor approval are still required'}


@app.get("/api/search")
def search(q: str):
    return engine().search(q)


@app.get("/api/segments")
def segments():
    return engine().personas.reset_index().to_dict("records")


@app.get("/api/cases")
def cases():
    with connect() as db:
        return [dict(r) for r in db.execute("SELECT cases.*,clients.label FROM cases JOIN clients ON clients.id=cases.client_id ORDER BY cases.id DESC")]


class NewCase(BaseModel):
    client_id: str
    subject: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1, max_length=4000)


@app.post("/api/cases", status_code=201)
def create_case(body: NewCase):
    engine().client(body.client_id)
    with connect() as db:
        c = db.execute("INSERT INTO cases(client_id,subject,created_at) VALUES(?,?,?)", (body.client_id, body.subject, now()))
        cid = c.lastrowid
        db.execute("INSERT INTO messages(case_id,role,body,created_at) VALUES(?,'client',?,?)", (cid, body.message, now()))
    event("case_created", {}, cid)
    return {"id": cid}


@app.get("/api/cases/{case_id}")
def get_case(case_id: int):
    with connect() as db:
        case = db.execute("SELECT * FROM cases WHERE id=?", (case_id,)).fetchone()
        if case is None:
            raise HTTPException(404, "Case not found")
        return {**dict(case), "client": engine().client(case["client_id"]),
                "messages": [dict(r) for r in db.execute("SELECT * FROM messages WHERE case_id=? ORDER BY id", (case_id,))],
                "drafts": [decode(r, ("evidence", "quality")) for r in db.execute("SELECT * FROM drafts WHERE case_id=? ORDER BY version DESC", (case_id,))]}


class Message(BaseModel):
    body: str = Field(min_length=1, max_length=4000)


@app.post("/api/cases/{case_id}/messages")
def add_message(case_id: int, body: Message):
    get_case(case_id)
    with connect() as db:
        if db.execute("SELECT status FROM cases WHERE id=?", (case_id,)).fetchone()[0] == "processing":
            raise HTTPException(409, "Wait for the current agent run before adding a reply")
        db.execute("INSERT INTO messages(case_id,role,body,created_at) VALUES(?,'client',?,?)", (case_id, body.body, now()))
        db.execute("UPDATE drafts SET status='superseded' WHERE case_id=? AND status IN ('pending','approved')", (case_id,))
        db.execute("UPDATE cases SET status='received',error=NULL WHERE id=?", (case_id,))
    event("client_reply", {}, case_id)
    return {"ok": True}


@app.post("/api/cases/{case_id}/run", status_code=202)
def run(case_id: int, tasks: BackgroundTasks):
    with connect() as db:
        c = db.execute("UPDATE cases SET status='processing',error=NULL WHERE id=? AND status!='processing'", (case_id,))
        if not c.rowcount:
            raise HTTPException(409, "Case missing or already processing")
    event("agent_started", {}, case_id)
    tasks.add_task(run_case, case_id)
    return {"status": "processing"}


def active_draft(db, draft_id):
    r = db.execute("SELECT * FROM drafts WHERE id=?", (draft_id,)).fetchone()
    if r is None:
        raise HTTPException(404, "Draft not found")
    if r["status"] not in ("pending", "approved"):
        raise HTTPException(409, "Draft has already been dispatched, rejected or superseded")
    if db.execute("SELECT status FROM cases WHERE id=?", (r["case_id"],)).fetchone()[0] == "processing":
        raise HTTPException(409, "The agent is generating a replacement; wait before acting on the draft")
    return r


@app.post("/api/drafts/{draft_id}/edit")
def edit(draft_id: int, body: Message):
    with connect() as db:
        db.execute("BEGIN IMMEDIATE")
        d = active_draft(db, draft_id)
        version = db.execute("SELECT MAX(version)+1 FROM drafts WHERE case_id=?", (d["case_id"],)).fetchone()[0]
        quality = {"human_review_required": True, "status": "edited; previous automated assessment invalidated"}
        cursor = db.execute("INSERT INTO drafts(case_id,version,body,evidence,quality,created_at) VALUES(?,?,?,?,?,?)",
                            (d["case_id"], version, body.body, d["evidence"], json.dumps(quality), now()))
        new_id = cursor.lastrowid
        db.execute("UPDATE drafts SET status='superseded' WHERE id=?", (draft_id,))
        db.execute("INSERT INTO decisions(draft_id,action,reason,created_at) VALUES(?,'edit','Advisor edited draft',?)", (new_id, now()))
        db.execute("UPDATE cases SET status='awaiting_review' WHERE id=?", (d["case_id"],))
    event("draft_edited", {"draft_id": new_id}, d["case_id"])
    return {"id": new_id, "version": version, "status": "pending"}


class Decision(BaseModel):
    action: Literal["approve", "reject", "escalate"]
    reason: str = Field(default="", max_length=1000)


@app.post("/api/drafts/{draft_id}/decision")
def decision(draft_id: int, body: Decision):
    if body.action in ("reject", "escalate") and not body.reason.strip():
        raise HTTPException(400, "A reason is required")
    with connect() as db:
        db.execute("BEGIN IMMEDIATE")
        d = active_draft(db, draft_id)
        if d["status"] != "pending":
            raise HTTPException(409, "Draft is already approved")
        edited = db.execute("SELECT 1 FROM decisions WHERE draft_id=? AND action='edit'", (draft_id,)).fetchone()
        action = ("approve_edited" if edited else "approve_without_edit") if body.action == "approve" else body.action
        status = {"approve": "approved", "reject": "rejected", "escalate": "escalated"}[body.action]
        db.execute("UPDATE drafts SET status=? WHERE id=?", (status, draft_id))
        db.execute("UPDATE cases SET status=? WHERE id=?", (status, d["case_id"]))
        db.execute("INSERT INTO decisions(draft_id,action,reason,created_at) VALUES(?,?,?,?)", (draft_id, action, body.reason, now()))
    event("advisor_decision", {"action": action, "draft_id": draft_id}, d["case_id"])
    return {"status": status}


@app.post("/api/drafts/{draft_id}/dispatch")
def dispatch(draft_id: int):
    with connect() as db:
        db.execute("BEGIN IMMEDIATE")
        existing = db.execute("SELECT id FROM outbox WHERE draft_id=?", (draft_id,)).fetchone()
        if existing:
            return {"id": existing[0], "reused": True}
        d = active_draft(db, draft_id)
        if d["status"] != "approved":
            raise HTTPException(409, "Advisor approval is required before simulated dispatch")
        day = db.execute("SELECT value FROM settings WHERE key='demo_date'").fetchone()[0]
        c = db.execute("INSERT INTO outbox(draft_id,case_id,body,dispatched_at,demo_date) VALUES(?,?,?,?,?)", (draft_id, d["case_id"], d["body"], now(), day))
        db.execute("UPDATE drafts SET status='dispatched' WHERE id=?", (draft_id,))
        db.execute("UPDATE cases SET status='dispatched' WHERE id=?", (d["case_id"],))
        db.execute("INSERT INTO messages(case_id,role,body,created_at) VALUES(?,'advisor',?,?)", (d["case_id"], d["body"], now()))
        oid = c.lastrowid
    event("simulated_dispatch", {"draft_id": draft_id, "demo_date": day}, d["case_id"])
    return {"id": oid, "reused": False}


@app.get("/api/outbox")
def outbox():
    with connect() as db:
        return [dict(r) for r in db.execute("SELECT outbox.*,cases.subject,cases.client_id FROM outbox JOIN cases ON cases.id=outbox.case_id ORDER BY outbox.id DESC")]


class Clock(BaseModel):
    date: date


@app.post("/api/clock")
def clock(body: Clock):
    with connect() as db:
        db.execute("UPDATE settings SET value=? WHERE key='demo_date'", (str(body.date),))
    return {"demo_date": str(body.date)}


@app.get("/api/metrics")
def metrics():
    return engine().metrics()


@app.post("/api/monitoring/report")
def monitoring_report():
    m = engine().weekly_activity()
    result = chat([{"role": "system", "content": "Write a short English weekly advisor activity report using ONLY these observed demo counts. Do not invent conversion, revenue or real business improvements. State that this is simulated communication. Include two useful follow-up actions."}, {"role": "user", "content": json.dumps(m)}])
    event("monitoring_report", {"metrics": m})
    return {"report": result["content"], "metrics": m, "model": OLLAMA_MODEL, "live": True}


@app.get("/api/evaluations")
def evaluations():
    path = ROOT / "reports/validation2_evaluation.json"
    result = json.loads(path.read_text()) if not CUSTOM_DATASET and path.exists() else {"status": "not_run", "objectives": []}
    result["human_reviews"] = human_review_metrics()
    result["draft_acceptance"] = draft_acceptance_metrics()
    return result


class HumanReview(BaseModel):
    reviewer: str = Field(min_length=1, max_length=100)
    grounded: bool
    relevant: bool
    professional: bool
    notes: str = Field(default="", max_length=2000)


@app.post("/api/drafts/{draft_id}/human-review")
def human_review(draft_id: int, body: HumanReview):
    with connect() as db:
        if not db.execute("SELECT 1 FROM drafts WHERE id=?", (draft_id,)).fetchone():
            raise HTTPException(404, "Draft not found")
        db.execute("INSERT INTO human_reviews(draft_id,reviewer,data,created_at) VALUES(?,?,?,?) ON CONFLICT(draft_id,reviewer) DO UPDATE SET data=excluded.data,created_at=excluded.created_at",
                   (draft_id, body.reviewer, body.model_dump_json(), now()))
    return {"recorded": True}


@app.get("/api/artifacts")
def artifacts():
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"status": "initial baseline", "version": ARTIFACT_VERSION}


@app.post("/api/artifacts/refresh", status_code=202)
def refresh(tasks: BackgroundTasks):
    tasks.add_task(refresh_artifacts)
    return {"status": "scheduled", "note": "Candidate training is gated. The last valid artifact remains active if checks fail."}


@app.post("/api/cases/{case_id}/manual-draft", status_code=201)
def manual_draft(case_id: int, body: Message):
    get_case(case_id)
    with connect() as db:
        db.execute("BEGIN IMMEDIATE")
        if db.execute("SELECT status FROM cases WHERE id=?", (case_id,)).fetchone()[0] == 'processing':
            raise HTTPException(409, "Wait for agent processing to finish")
        version = db.execute("SELECT COALESCE(MAX(version),0)+1 FROM drafts WHERE case_id=?", (case_id,)).fetchone()[0]
        db.execute("UPDATE drafts SET status='superseded' WHERE case_id=? AND status IN ('pending','approved')", (case_id,))
        cursor = db.execute("INSERT INTO drafts(case_id,version,body,evidence,quality,created_at) VALUES(?,?,?,?,?,?)", (case_id, version, body.body,
                            json.dumps({"live": False, "model": "advisor manual entry", "trace": [], "passages": []}),
                            json.dumps({"human_review_required": True, "status": "manual draft; not model evaluated"}), now()))
        db.execute("UPDATE cases SET status='awaiting_review',error=NULL WHERE id=?", (case_id,))
    event('manual_draft', {"version": version}, case_id)
    return {"id": cursor.lastrowid, "version": version}


class Batch(BaseModel):
    records: list = Field(max_length=2000)
    simulate_transient_failure: bool = False


@app.post("/api/pipeline")
def import_batch(body: Batch):
    return pipeline(body.records, body.simulate_transient_failure)


@app.get("/api/pipeline")
def batches():
    with connect() as db:
        return [decode(r, ("data", "findings", "trace")) for r in db.execute("SELECT * FROM batches ORDER BY id DESC LIMIT 50")]


class Pair(BaseModel):
    left: dict
    right: dict


@app.post("/api/duplicates/check")
def duplicates(body: Pair):
    if not body.left.get("id") or not body.right.get("id"):
        raise HTTPException(400, "Both records require an id")
    result = match_pair(body.left, body.right)
    if result["decision"] in ("duplicate", "review"):
        left = f"{body.left.get('kind')}:{body.left['id']}"
        right = f"{body.right.get('kind')}:{body.right['id']}"
        if left == right:
            raise HTTPException(400, "Compare two different source records")
        left, right = sorted((left, right))
        with connect() as db:
            db.execute("INSERT OR IGNORE INTO duplicate_reviews(left_id,right_id,score,evidence) VALUES(?,?,?,?)", (left, right, result["score"], json.dumps(result["evidence"])))
    return result


@app.get("/api/duplicates")
def duplicate_queue():
    with connect() as db:
        return [decode(r, ("evidence",)) for r in db.execute("SELECT * FROM duplicate_reviews ORDER BY id DESC")]


class MatchDecision(BaseModel):
    confirmed: bool


@app.post("/api/duplicates/{review_id}")
def match_decision(review_id: int, body: MatchDecision):
    with connect() as db:
        r = db.execute("SELECT * FROM duplicate_reviews WHERE id=?", (review_id,)).fetchone()
        if r is None:
            raise HTTPException(404, "Review not found")
        db.execute("UPDATE duplicate_reviews SET status=? WHERE id=?", ("confirmed" if body.confirmed else "distinct", review_id))
        if body.confirmed:
            db.execute("INSERT OR REPLACE INTO identity_links VALUES(?,?,?)", (r["right_id"], r["left_id"], review_id))
        else:
            db.execute("DELETE FROM identity_links WHERE review_id=?", (review_id,))
    event("identity_review", {"review_id": review_id, "confirmed": body.confirmed})
    return {"ok": True}
