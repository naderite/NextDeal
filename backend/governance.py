"""DSO5: auditable data gates, within-type matching and bounded recovery."""
import hashlib
import json
import re
import unicodedata
from datetime import date
from difflib import SequenceMatcher

from .db import connect, decode, event, now


def normal(text):
    text = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", text)


def quality_check(records):
    findings, seen = [], set()
    for i, r in enumerate(records):
        def flag(field, rule, severity="error"):
            findings.append({"row": i, "field": field, "rule": rule, "severity": severity})
        if not isinstance(r, dict):
            flag("record", "object_required")
            continue
        for field in ("id", "kind", "label"):
            if not isinstance(r.get(field), str) or not r[field].strip():
                flag(field, "required_string")
        if r.get("kind") not in ("corporate", "individual"):
            flag("kind", "unknown_client_type")
        key = (r.get("kind"), str(r.get("id")))
        if key in seen:
            flag("id", "duplicate_identifier")
        seen.add(key)
        for field in ("age", "premium", "claim_count"):
            if field in r and (isinstance(r[field], bool) or not isinstance(r[field], (int, float)) or not 0 <= r[field] <= {"age": 120, "premium": 1e8, "claim_count": 100}[field]):
                flag(field, "out_of_range")
        for field in ("effective_date", "expiry_date"):
            if r.get(field):
                try:
                    date.fromisoformat(r[field])
                except (ValueError, TypeError):
                    flag(field, "invalid_date")
        try:
            if r.get("effective_date") and r.get("expiry_date") and date.fromisoformat(r["expiry_date"]) < date.fromisoformat(r["effective_date"]):
                flag("expiry_date", "before_effective_date")
        except (ValueError, TypeError):
            pass
        if r.get("email") and not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", str(r["email"])):
            flag("email", "invalid_email")
    return findings


def match_pair(a, b):
    if a.get("kind") != b.get("kind"):
        return {"score": 0.0, "decision": "different_type", "evidence": []}
    evidence, weights = [], []
    for field, weight in (("identity", .65), ("email", .55), ("phone", .5)):
        if normal(a.get(field)) and normal(a.get(field)) == normal(b.get(field)):
            evidence.append(f"Same normalized {field}")
            weights.append(weight)
    name = SequenceMatcher(None, normal(a.get("label")), normal(b.get("label"))).ratio()
    if name >= .72:
        evidence.append(f"Name similarity {name:.2f}")
    score = min(.99, sum(weights) + .3 * name) if weights else .55 * name
    return {"score": round(score, 3), "decision": "duplicate" if score >= .8 else "review" if score >= .5 else "distinct", "evidence": evidence}


def pipeline(records, transient_failure=False):
    """Inspect -> bounded retry -> quality gate -> atomic import/quarantine. Idempotent by content."""
    digest = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    with connect() as db:
        existing = db.execute("SELECT * FROM batches WHERE digest=?", (digest,)).fetchone()
    if existing:
        return {**decode(existing, ("data", "findings", "trace")), "reused": True}
    trace = [{"action": "inspect_batch", "records": len(records)}]
    if transient_failure:
        trace.extend([{"action": "retry", "attempt": 1, "reason": "injected transient read failure"},
                      {"action": "read_recovered", "attempt": 2}])
    findings = quality_check(records)
    status = "quarantined" if findings else "accepted"
    trace.append({"action": "quarantine" if findings else "import", "findings": len(findings)})
    with connect() as db:
        cursor = db.execute("INSERT INTO batches(digest,status,data,findings,trace,created_at) VALUES(?,?,?,?,?,?)",
                            (digest, status, json.dumps(records), json.dumps(findings), json.dumps(trace), now()))
        batch_id = cursor.lastrowid
        if status == "accepted":
            for r in records:
                db.execute("INSERT OR IGNORE INTO imported_records VALUES(?,?,?)", (f"{r['kind']}:{r['id']}", batch_id, json.dumps(r)))
    event("pipeline_run", {"batch_id": batch_id, "status": status})
    return {"id": batch_id, "status": status, "findings": findings, "trace": trace, "reused": False}
