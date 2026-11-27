"""Measure the NextDeal business matcher against published restaurant gold pairs."""

import argparse
import csv
from itertools import combinations
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.governance import match_pair


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def evaluate(data):
    records = [
        {"id": row["id"], "kind": "corporate", "label": row["name"],
         "phone": row["phone"], "address": row["addr"], "city": row["city"]}
        for row in read_csv(data / "restaurant.csv")
    ]
    truth = {
        frozenset((row["id_1"], row["id_2"]))
        for row in read_csv(data / "restaurant_gold.csv")
    }
    ids = {row["id"] for row in records}
    if len(ids) != len(records) or not truth or any(len(pair) != 2 or not pair <= ids for pair in truth):
        raise ValueError("Expected unique record IDs and valid gold pairs")
    predicted = {
        frozenset((left["id"], right["id"]))
        for left, right in combinations(records, 2)
        if match_pair(left, right)["decision"] == "duplicate"
    }
    true_positives = len(predicted & truth)
    precision = true_positives / len(predicted) if predicted else 0.0
    recall = true_positives / len(truth)
    return {
        "source": "https://cran.r-project.org/package=restaurant",
        "scope": "Development benchmark across all unordered record pairs. The corporate matching rule was developed using this dataset.",
        "records": len(records),
        "pairs": len(records) * (len(records) - 1) // 2,
        "true_positives": true_positives,
        "false_positives": len(predicted - truth),
        "false_negatives": len(truth - predicted),
        "precision": precision,
        "recall": recall,
        "target_precision": 0.95,
        "target_recall": 0.85,
        "meets_development_targets": precision >= 0.95 and recall >= 0.85,
    }


def evaluate_pairs(path):
    rows = read_csv(path)
    if not rows or any(row.get("label") not in {"0", "1"} for row in rows):
        raise ValueError("Expected a nonempty CSV with binary match labels")

    true_positives = false_positives = false_negatives = 0
    for row in rows:
        fields = {key.lower(): value for key, value in row.items()}
        left, right = (
            {"kind": "corporate", "label": fields[f"{side}_name"],
             "city": fields.get(f"{side}_city"),
             "address": fields.get(f"{side}_address") or fields.get(f"{side}_streetaddress"),
             "phone": fields.get(f"{side}_phone") or fields.get(f"{side}_phonenumber")}
            for side in ("a", "b")
        )
        predicted = match_pair(left, right)["decision"] == "duplicate"
        expected = row["label"] == "1"
        true_positives += predicted and expected
        false_positives += predicted and not expected
        false_negatives += not predicted and expected

    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, true_positives + false_negatives)
    return {
        "dataset": path.name,
        "pairs": len(rows),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "target_precision": 0.95,
        "target_recall": 0.85,
        "meets_targets": precision >= 0.95 and recall >= 0.85,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labeled-pairs", action="store_true")
    args = parser.parse_args()
    result = evaluate_pairs(args.data) if args.labeled_pairs else evaluate(args.data)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
