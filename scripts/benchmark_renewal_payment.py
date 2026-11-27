"""Evaluate a renewal-premium non-renewal proxy on the public DataHack table."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

SOURCE = "https://datahack-prod.s3.amazonaws.com/train_file/train_ZoGVYWq.csv"
CATEGORICAL = ["sourcing_channel", "residence_area_type"]


def benchmark(path):
    frame = pd.read_csv(path)
    if frame.id.isna().any() or frame.id.duplicated().any() or set(frame.renewal) != {0, 1}:
        raise ValueError("Expected one public challenge row per policy and binary renewal labels")
    labels = frame.renewal.eq(0).to_numpy()
    inputs = frame.drop(columns=["id", "renewal"])
    train, heldout = train_test_split(np.arange(len(frame)), test_size=.4, stratify=labels, random_state=42)
    validation, test = train_test_split(heldout, test_size=.5, stratify=labels[heldout], random_state=42)
    x_train = pd.get_dummies(inputs.iloc[train], columns=CATEGORICAL)

    def features(indices):
        return pd.get_dummies(inputs.iloc[indices], columns=CATEGORICAL).reindex(
            columns=x_train.columns, fill_value=False,
        )

    x_val, x_test = features(validation), features(test)
    candidates = []
    for depth in (3, 5, 7):
        model = HistGradientBoostingClassifier(max_depth=depth, learning_rate=.06, max_iter=220,
                                               min_samples_leaf=50, random_state=42)
        model.fit(x_train, labels[train])
        val_score = model.predict_proba(x_val)[:, 1]
        candidates.append((average_precision_score(labels[validation], val_score), depth, model, val_score))
    val_ap, depth, model, probabilities = max(candidates, key=lambda result: result[0])
    positives = np.sort(probabilities[labels[validation]])[::-1]
    # Set a small buffer on validation recall; the test partition is scored once.
    threshold = positives[int(np.ceil(.73 * len(positives))) - 1]
    scores = model.predict_proba(x_test)[:, 1]
    alerts = scores >= threshold
    ap = float(average_precision_score(labels[test], scores))
    recall = float(recall_score(labels[test], alerts))
    return {"source": SOURCE, "outcome": "renewal == 0",
            "scope": "Public renewal-premium challenge proxy. Non-renewal does not distinguish missed invoices "
                     "from other reasons and the published table has no dates for temporal evaluation.",
            "training_policies": len(train), "validation_policies": len(validation), "test_policies": len(test),
            "test_positive_policies": int(labels[test].sum()), "prevalence": float(labels[test].mean()),
            "model": f"histogram gradient boosting, max depth {depth}", "validation_average_precision": float(val_ap),
            "threshold": float(threshold), "alerts": int(alerts.sum()), "recall": recall,
            "precision": float(precision_score(labels[test], alerts, zero_division=0)),
            "pr_auc_average_precision": ap,
            "reference_targets": {"recall": .70, "pr_auc": .40},
            "reference_targets_met_on_proxy": recall >= .7 and ap >= .4}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(benchmark(args.data), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
