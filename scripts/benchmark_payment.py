"""Benchmark whether a premium invoice remains unpaid at a fixed horizon."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def evaluate(path, *, observed_through, validation_start, test_start, horizon_days):
    invoices = pd.read_csv(path, dtype={"invoice_id": str, "policy_id": str})
    required = {"invoice_id", "policy_id", "issued_on", "due_on", "paid_on", "premium"}
    if not required <= set(invoices):
        raise ValueError(f"Missing columns: {', '.join(sorted(required - set(invoices)))}")
    if invoices.invoice_id.isna().any() or invoices.invoice_id.duplicated().any() or invoices.policy_id.isna().any():
        raise ValueError("Invoice IDs must be unique and policy IDs must be present")
    for field in ("issued_on", "due_on", "paid_on"):
        invoices[field] = pd.to_datetime(invoices[field], format="%Y-%m-%d", errors="raise")
    if invoices[["issued_on", "due_on"]].isna().any().any() or (invoices.due_on < invoices.issued_on).any():
        raise ValueError("Every invoice needs a due date on or after its issue date")
    if (invoices.paid_on < invoices.issued_on).any():
        raise ValueError("Payment date precedes invoice issue date")
    if (invoices.paid_on > observed_through).any():
        raise ValueError("Payment date exceeds the observation cutoff")
    invoices["premium"] = pd.to_numeric(invoices.premium, errors="raise")
    if (~np.isfinite(invoices.premium) | (invoices.premium < 0)).any():
        raise ValueError("Premium must be finite and nonnegative")
    invoices["term_days"] = (invoices.due_on - invoices.issued_on).dt.days
    invoices["label_date"] = invoices.due_on + pd.Timedelta(days=horizon_days)
    invoices["unpaid"] = invoices.paid_on.isna() | (invoices.paid_on > invoices.label_date)

    # Only use historical invoice outcomes that had matured before the new invoice was issued.
    history = invoices[["policy_id", "label_date", "unpaid"]].sort_values("label_date")
    history["past_invoices"] = history.groupby("policy_id").cumcount() + 1
    history["past_unpaid"] = history.groupby("policy_id").unpaid.cumsum()
    invoices = pd.merge_asof(
        invoices.sort_values("issued_on"),
        history.drop(columns="unpaid").rename(columns={"label_date": "past_label_date"}),
        by="policy_id", left_on="issued_on", right_on="past_label_date", allow_exact_matches=False,
    )
    invoices[["past_invoices", "past_unpaid"]] = invoices[["past_invoices", "past_unpaid"]].fillna(0)
    features = ["premium", "term_days", "past_invoices", "past_unpaid"]
    train = invoices.loc[invoices.label_date < validation_start]
    validation = invoices.loc[invoices.issued_on.ge(validation_start) & invoices.label_date.lt(test_start)]
    test = invoices.loc[invoices.issued_on.ge(test_start) & invoices.label_date.le(observed_through)]
    for name, split in (("training", train), ("validation", validation), ("test", test)):
        if split.unpaid.nunique() != 2:
            raise ValueError(f"{name.title()} split must contain both paid and unpaid outcomes")
    model = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(train[features], train.unpaid)
    scores = model.predict_proba(validation[features])[:, 1]
    cutoff = np.sort(scores[validation.unpaid])[::-1][int(np.ceil(.70 * validation.unpaid.sum())) - 1]
    scores = model.predict_proba(test[features])[:, 1]
    alerts = scores >= cutoff
    ap = float(average_precision_score(test.unpaid, scores))
    recall = float(recall_score(test.unpaid, alerts))
    return {
        "horizon_days_after_due": horizon_days,
        "validation_start": validation_start.date().isoformat(),
        "test_start": test_start.date().isoformat(),
        "observed_through": observed_through.date().isoformat(),
        "training_invoices": len(train), "validation_invoices": len(validation), "test_invoices": len(test),
        "test_unpaid": int(test.unpaid.sum()), "test_prevalence": float(test.unpaid.mean()),
        "alerts": int(alerts.sum()), "recall": recall, "precision": float(precision_score(test.unpaid, alerts, zero_division=0)),
        "pr_auc_average_precision": ap, "validation_threshold": float(cutoff),
        "targets_met": recall >= .70 and ap >= .40,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("invoices", type=Path)
    parser.add_argument("--observed-through", type=pd.Timestamp, required=True)
    parser.add_argument("--validation-start", type=pd.Timestamp, required=True)
    parser.add_argument("--test-start", type=pd.Timestamp, required=True)
    parser.add_argument("--horizon-days", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.horizon_days < 0 or not args.validation_start < args.test_start <= args.observed_through:
        parser.error("Use ordered cutoffs and a nonnegative horizon")
    result = evaluate(args.invoices, observed_through=args.observed_through,
                      validation_start=args.validation_start, test_start=args.test_start,
                      horizon_days=args.horizon_days)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
