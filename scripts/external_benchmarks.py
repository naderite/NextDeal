"""Run external insurance benchmarks without loading NextDeal's runtime or source extract."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def ranking_episodes(profiles, holdings):
    row, target = np.nonzero(holdings & (holdings.sum(axis=1) >= 2)[:, None])
    visible = holdings[row].copy()
    visible[np.arange(len(target)), target] = False
    return np.column_stack((profiles[row], visible)), visible, target


def reconstruction_benchmark(train_profiles, train_holdings, test_profiles, test_holdings):
    train, _, target = ranking_episodes(train_profiles, train_holdings)
    test, visible, truth = ranking_episodes(test_profiles, test_holdings)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=400)).fit(train, target)
    counts = np.bincount(target, minlength=train_holdings.shape[1])
    popular = np.argsort(-counts, kind="stable")[:3]
    tail = ~np.isin(truth, popular)
    baseline = np.broadcast_to(counts, visible.shape).astype(float).copy()
    scores = np.full(visible.shape, -np.inf)
    scores[:, model.classes_] = model.predict_proba(test)

    def metrics(scores):
        scores[visible] = -np.inf
        top = np.argsort(-scores, axis=1, kind="stable")[:, :3]
        valid = np.isfinite(np.take_along_axis(scores, top, axis=1))
        hits = ((top == truth[:, None]) & valid).any(axis=1)
        return {"hit_rate_at_3": float(hits.mean()),
                "long_tail_hit_rate_at_3": float(hits[tail].mean()) if tail.any() else None}

    return {"training_customers": len(train_profiles), "test_customers": len(test_profiles),
            "test_episodes": len(truth), "long_tail_episodes": int(tail.sum()),
            "logistic": metrics(scores), "popularity": metrics(baseline)}


def coil_benchmark(training, test, labels):
    train = np.loadtxt(training, dtype=int, ndmin=2)
    holdout = np.loadtxt(test, dtype=int, ndmin=2)
    labels = np.loadtxt(labels, dtype=int, ndmin=1)
    if train.shape[1] != 86 or holdout.shape[1] != 85 or labels.shape != (len(holdout),):
        raise ValueError("Expected UCI CoIL files with 86 training and 85 test columns")
    holdout = np.column_stack((holdout, labels))
    return {
        **reconstruction_benchmark(train[:, :43], train[:, 64:] > 0, holdout[:, :43], holdout[:, 64:] > 0),
        "source": "https://kdd.ics.uci.edu/databases/tic/tic.html",
        "scope": "Consumer portfolio reconstruction on the publisher split. Premium contribution columns excluded.",
        "targets": {"hit_rate_at_3": .70, "long_tail_hit_rate_at_3": .40},
    }


def annual_holdings_benchmark(snapshots):
    if snapshots.duplicated(["entity", "date"]).any():
        raise ValueError("Expected unique entity and date pairs")
    rows = snapshots.assign(outcome_date=snapshots.date + pd.DateOffset(years=1))
    outcomes = snapshots.rename(columns={"date": "outcome_date", "holdings": "future_holdings"})
    paired = rows.merge(outcomes, on=["entity", "outcome_date"], validate="many_to_one")
    cohorts = []
    last_date = snapshots.date.max()
    for year, year_rows in rows.groupby(rows.date.dt.year):
        if last_date.year <= year:
            continue
        matched = paired[paired.date.dt.year == year]
        rho = None
        if matched.holdings.nunique() > 1 and matched.future_holdings.nunique() > 1:
            rho = float(spearmanr(matched.holdings, matched.future_holdings).statistic)
        cohorts.append({"start_year": int(year), "starting_records": len(year_rows),
                        "anniversary_in_observation_window": int((year_rows.outcome_date <= last_date).sum()),
                        "matched_pairs": len(matched), "persistence_spearman": rho})
    return cohorts


def motor_benchmark(path):
    fields = ["ID", "Date_last_renewal", "Policies_in_force"]
    rows = pd.read_csv(path, sep=None, engine="python", usecols=fields, dtype={"ID": str})
    rows = rows.rename(columns=dict(zip(fields, ("entity", "date", "holdings"))))
    rows["date"] = pd.to_datetime(rows.date, format="%d/%m/%Y")
    return {"source": "https://doi.org/10.17632/5cxyb5fp4f.2", "license": "CC BY 4.0",
            "authors": ["Josep Lledó", "Jose M. Pavía"], "rows": len(rows),
            "scope": "Spanish motor portfolio. Same policyholder-product observed one calendar year later. "
                     "Unmatched records are excluded from correlation. Lapse combines different causes.",
            "target_spearman": .50, "cohorts": annual_holdings_benchmark(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory containing the four published data files")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [args.directory / name for name in (
        "coil-training.tsv", "coil-test.tsv", "coil-test-target.tsv", "motor-portfolio.csv"
    )]
    report = {"ranking": coil_benchmark(*paths[:3]), "holdings": motor_benchmark(paths[3]),
              "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
