"""Groundedness and judge agreement from recorded human reviews."""
import json
from collections import defaultdict

from sklearn.metrics import cohen_kappa_score

from .db import connect


def human_review_metrics():
    with connect() as db:
        rows = db.execute("""
            SELECT h.draft_id, h.data, d.quality FROM human_reviews h
            JOIN drafts d ON d.id = h.draft_id
        """).fetchall()
    labels = defaultdict(set)
    assessed = {}
    for row in rows:
        quality = json.loads(row["quality"])
        if not quality.get("model") or type(quality.get("grounded")) is not bool:
            continue
        draft_id = row["draft_id"]
        labels[draft_id].add(json.loads(row["data"])["grounded"])
        assessed[draft_id] = quality["grounded"]

    agreed = [(draft_id, next(iter(values))) for draft_id, values in labels.items() if len(values) == 1]
    truth = [value for _, value in agreed]
    prediction = [assessed[draft_id] for draft_id, _ in agreed]
    return {
        "count": len(rows),
        "reviewed_judged_drafts": len(labels),
        "consensus_drafts": len(agreed),
        "disputed_drafts": len(labels) - len(agreed),
        "faithfulness": sum(truth) / len(truth) if truth else None,
        "quality_kappa": float(cohen_kappa_score(truth, prediction)) if len(set(truth)) == 2 else None,
    }


def draft_acceptance_metrics():
    """Count first advisor intervention once per model-generated draft."""
    with connect() as db:
        rows = db.execute("""
            SELECT d.quality,
                   EXISTS (
                       SELECT 1 FROM drafts revision
                       JOIN decisions decision ON decision.draft_id = revision.id
                       WHERE revision.case_id = d.case_id AND revision.version = d.version + 1
                         AND decision.action = 'edit'
                   ) AS edited,
                   (SELECT action FROM decisions
                    WHERE draft_id = d.id AND action IN ('approve_without_edit', 'reject', 'escalate')
                    ORDER BY id LIMIT 1) AS decision
            FROM drafts d
        """).fetchall()
    counts = dict.fromkeys(("approved_without_edit", "edited", "rejected", "escalated", "unreviewed"), 0)
    for row in rows:
        if not json.loads(row["quality"]).get("model"):
            continue
        outcome = {
            "approve_without_edit": "approved_without_edit",
            "reject": "rejected",
            "escalate": "escalated",
        }.get(row["decision"], "unreviewed")
        counts["edited" if row["edited"] else outcome] += 1
    reviewed = sum(counts.values()) - counts["unreviewed"]
    return {**counts, "reviewed": reviewed,
            "approval_without_edit_rate": counts["approved_without_edit"] / reviewed if reviewed else None}
