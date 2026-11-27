"""Measure coverage-addition suggestions on disjoint insureds and later years."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

COVERAGES = ("liability", "property_damage", "theft", "fire", "glass", "legal_protection", "occupants")
SOURCE = "https://doi.org/10.17632/sw4jmdb2sm.1"
NUMERIC = ["bonus_score", "vehicle_age", "vehicle_value", "seats", "power_to_weight_ratio"]
FIELDS = ["insured_id", "year", "policy_type", *(f"{coverage}_premium" for coverage in COVERAGES), *NUMERIC]


def load(path):
    frame = pd.read_csv(path, sep=";", usecols=FIELDS, dtype={"insured_id": str, "policy_type": str})
    if len(frame) != 354140:
        raise ValueError("Expected all 354,140 source rows")
    premiums = frame[[f"{name}_premium" for name in COVERAGES]].apply(
        lambda s: pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="raise")
    )
    for name in NUMERIC:
        frame[name] = pd.to_numeric(frame[name].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    for index, name in enumerate(COVERAGES):
        frame[name] = premiums.iloc[:, index].gt(0) & premiums.iloc[:, index].notna()
    # Ambiguous insured-year keys cannot reliably identify a coverage change.
    valid = frame.dropna(subset=["insured_id", "year", "policy_type"])
    valid = valid.loc[premiums.loc[valid.index].notna().all(axis=1)]
    return valid.loc[~valid.duplicated(["insured_id", "year"], keep=False)]


def transitions(frame, year):
    current = frame.loc[frame.year.eq(year), ["insured_id", "policy_type", *COVERAGES, *NUMERIC]].copy()
    following = frame.loc[frame.year.eq(year + 1), ["insured_id", *COVERAGES]]
    paired = current.merge(following, on="insured_id", suffixes=("", "_next"), validate="one_to_one")
    paired["basket"] = sum((paired[name].astype(int) * (2 ** i))
                           for i, name in enumerate(COVERAGES))
    return paired


def estimates(rows):
    baseline, rules = {}, {}
    for name in COVERAGES:
        eligible = rows.loc[~rows[name]]
        baseline[name] = float(eligible[name + "_next"].mean()) if len(eligible) else 0.0
        if baseline[name] <= 0:
            continue
        groups = eligible.groupby(["policy_type", "basket"])[name + "_next"].agg(["size", "sum", "mean"])
        for key, group in groups.iterrows():
            lift = group["mean"] / baseline[name]
            if group["size"] >= 30 and group["sum"] >= 5 and lift >= 1.5:
                rules.setdefault(key, []).append((name, float(group["mean"]), float(lift)))
    return {key: max(candidates, key=lambda r: (r[1], r[2], r[0]))[0]
            for key, candidates in rules.items()}


def model_suggestions(train, test):
    """Select the highest above-base-rate forecast among supported unheld coverages."""
    train_features = pd.get_dummies(train[["policy_type", *COVERAGES, *NUMERIC]], columns=["policy_type"])
    test_features = pd.get_dummies(test[["policy_type", *COVERAGES, *NUMERIC]], columns=["policy_type"])
    test_features = test_features.reindex(columns=train_features.columns, fill_value=False)
    scores = np.full((len(test), len(COVERAGES)), -np.inf)
    supported = []
    for i, name in enumerate(COVERAGES):
        available = ~train[name]
        y = train.loc[available, name + "_next"]
        if y.sum() < 20 or (~y).sum() < 20:
            continue
        model = HistGradientBoostingClassifier(max_depth=3, max_iter=75, min_samples_leaf=40, random_state=0)
        features = train_features.columns[train_features.loc[available].nunique() > 1]
        model.fit(train_features.loc[available, features], y)
        choices = ~test[name]
        if not choices.any():
            continue
        supported.append(name)
        scores[choices.to_numpy(), i] = model.predict_proba(test_features.loc[choices, features])[:, 1] / y.mean()
    confidence = scores.max(axis=1)
    eligible = np.flatnonzero(np.isfinite(confidence))
    order = eligible[np.argsort(-confidence[eligible], kind="stable")]
    # Fixed quota uses scores only. No holdout outcomes are used for selection.
    chosen = order[:int(np.ceil(.60 * len(test)))]
    proposals = pd.Series(None, index=test.index, dtype=object)
    proposals.iloc[chosen] = [COVERAGES[i] for i in scores[chosen].argmax(axis=1)]
    return proposals, supported


def benchmark(path, method="boosted"):
    rows = load(path)
    fit = transitions(rows, 2022)
    future = transitions(rows, 2023)
    train_half = pd.util.hash_pandas_object(fit.insured_id, index=False).mod(2).eq(0)
    test_half = pd.util.hash_pandas_object(future.insured_id, index=False).mod(2).eq(1)
    train, test = fit.loc[train_half], future.loc[test_half]
    if train.empty or test.empty:
        raise ValueError("No complete transitions for the benchmark")
    rules = estimates(train)
    unheld = (~test[list(COVERAGES)]).any(axis=1)
    additions = {name: int((~test[name] & test[name + "_next"]).sum()) for name in COVERAGES}
    if method == "boosted":
        proposed, supported = model_suggestions(train, test)
    else:
        proposed = pd.Series([rules.get(k) for k in zip(test.policy_type, test.basket)], index=test.index)
        supported = sorted(set(rules.values()))
    possible_hit = np.zeros(len(test), dtype=bool)
    rates = []
    for name in supported:
        available = ~test[name]
        if available.any():
            possible_hit |= (available & test[name + "_next"]).to_numpy()
            rates.append(float(test.loc[available, name + "_next"].mean()))
    min_expected = np.ceil(.60 * len(test)) * min(rates, default=0)
    oracle_upper_bound = float(possible_hit.sum() / min_expected) if min_expected else None
    suggested = int(proposed.notna().sum())
    hits, expected_hits = 0, 0.0
    results = {}
    for name in COVERAGES:
        picks = proposed.eq(name)
        if not picks.any():
            continue
        at_risk = test.loc[~test[name]]
        assert not test.loc[picks, name].any()
        base_rate = float(at_risk[name + "_next"].mean())
        count = int(picks.sum())
        picked_hits = int(test.loc[picks, name + "_next"].sum())
        hits += picked_hits
        expected_hits += count * base_rate
        results[name] = {"suggestions": count, "subsequent_additions": picked_hits,
                         "holdout_addition_rate_among_unheld": base_rate}
    return {"source": SOURCE, "authors": ["Priscila Espinosa", "Josep Lledó", "David Atance"], "license": "CC BY 4.0",
            "training_years": [2022, 2023], "test_years": [2023, 2024],
            "training_transitions": len(train), "test_transitions": len(test),
            "method": method, "supported_coverages": supported,
            "oracle_lift_upper_bound_at_target_coverage": oracle_upper_bound,
            "suggestions": suggested, "coverage": suggested / len(test), "subsequent_additions": hits,
            "learned_rule_groups": len(rules),
            "holdout_with_any_unheld_coverage": int(unheld.sum()),
            "holdout_observed_additions": additions,
            "lift": hits / expected_hits if expected_hits else None, "by_coverage": results,
            "targets": {"coverage": .60, "lift": 1.5},
            "scope": "Continued or re-observed insureds with one usable record per year. "
                     "Positive premium marks observed paid-for coverage. Lift compares observed additions "
                     "to the held-out unheld-coverage base rates for the suggested mix. "
                     "This benchmark measures association with additions, not contractual eligibility or causal impact."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("portfolio", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("boosted", "rules"), default="boosted")
    args = parser.parse_args()
    args.output.write_text(json.dumps(benchmark(args.portfolio, args.method), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
