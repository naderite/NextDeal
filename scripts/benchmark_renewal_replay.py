"""Replay historical renewals with scripted decisions in a disposable advisor database."""
import argparse
from datetime import date
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def replay(as_of):
    from fastapi.testclient import TestClient
    from backend.main import app
    from backend.db import connect

    with TestClient(app) as api:
        with connect() as db:
            days = [date.fromisoformat(row[0]) for row in db.execute(
                "SELECT DISTINCT contact_date FROM renewal_schedule ORDER BY contact_date"
            )]
        if not days:
            raise ValueError("The dataset contains no scheduled renewal cases")
        approved = 0
        approval_gate_checked = False
        for day in sorted({max(day, as_of) for day in days}):
            api.post("/api/clock", json={"date": day.isoformat()}).raise_for_status()
            prepared = api.post("/api/renewals/prepare-due").raise_for_status().json()
            for case_id in prepared["case_ids"]:
                draft = api.post(f"/api/cases/{case_id}/manual-draft", json={
                    "body": "Benchmark fixture for reviewing an upcoming policy renewal."
                }).raise_for_status().json()["id"]
                if not approval_gate_checked:
                    denied = api.post(f"/api/drafts/{draft}/dispatch")
                    if denied.status_code != 409:
                        raise RuntimeError("Expected dispatch to reject an unapproved draft")
                    approval_gate_checked = True
                api.post(f"/api/drafts/{draft}/decision", json={"action": "approve"}).raise_for_status()
                api.post(f"/api/drafts/{draft}/dispatch").raise_for_status()
                approved += 1
        metrics = api.get("/api/metrics").raise_for_status().json()["renewal_dispatch"]
    return {
        "as_of": as_of.isoformat(), "scripted_approvals": approved,
        "unapproved_dispatch_rejected": approval_gate_checked,
        "target_timely_coverage": .95, **metrics,
        "scope": "Historical date replay with scripted approvals and simulated dispatch. "
                 "Measures workflow execution; does not measure reviewer acceptance or actual delivery.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Prepared renewal dataset directory")
    parser.add_argument("--as-of", type=date.fromisoformat, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    bundled = Path(__file__).resolve().parents[1] / "data"
    if dataset == bundled.resolve() or not dataset.is_dir():
        parser.error("Provide a separate prepared external dataset directory")
    with TemporaryDirectory(prefix="nextdeal-renewal-replay-") as runtime:
        os.environ["NEXTDEAL_DATA_DIR"] = str(dataset)
        os.environ["NEXTDEAL_RUNTIME"] = runtime
        os.environ["NEXTDEAL_MAINTENANCE"] = "0"
        result = replay(args.as_of)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
