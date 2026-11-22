"""DSO2.2 association rules on guarantee baskets (Apriori, antecedents of size 1-2).

A basket = the guarantees of one contract + context items (`sub::<sub-branch>`, `prod::<product>`)
so a rule can say "motor contracts with RC-RTI and INCENDIE also take VOL". Only guarantees are
allowed as consequents. Small, dependency-free implementation (mlxtend is not required).
"""
from __future__ import annotations

from itertools import combinations

import pandas as pd


def baskets_from_contracts(pp: pd.DataFrame) -> list[frozenset]:
    out = []
    for gl, sub, prod in zip(pp["guarantee_list"], pp["sub_branch"], pp["product"]):
        out.append(frozenset([f"gar::{g}" for g in gl] + [f"sub::{sub}", f"prod::{prod}"]))
    return out


def mine_rules(baskets, min_support: int = 3, min_confidence: float = 0.3, min_lift: float = 1.5,
               max_antecedent: int = 2) -> pd.DataFrame:
    """All rules A -> g (g a guarantee) with support count >= min_support, confidence and lift thresholds."""
    n = len(baskets)
    count = {}
    for b in baskets:
        for size in range(1, max_antecedent + 2):
            for items in combinations(sorted(b), size):
                count[frozenset(items)] = count.get(frozenset(items), 0) + 1
    rows = []
    for itemset, c in count.items():
        if c < min_support or len(itemset) < 2:
            continue
        for cons in itemset:
            if not cons.startswith("gar::"):
                continue
            ante = itemset - {cons}
            if len(ante) > max_antecedent:
                continue
            conf = c / count[ante]
            lift = conf / (count[frozenset([cons])] / n)
            if conf >= min_confidence and lift >= min_lift:
                rows.append({"antecedent": tuple(sorted(ante)), "consequent": cons[5:], "support": c / n,
                             "support_count": c, "confidence": conf, "lift": lift})
    cols = ["antecedent", "consequent", "support", "support_count", "confidence", "lift"]
    return pd.DataFrame(rows, columns=cols).sort_values(["confidence", "lift"], ascending=False).reset_index(drop=True)


def recommend(basket: frozenset, rules: pd.DataFrame, k: int = 3) -> list[dict]:
    """Guarantees not held yet, from the most confident firing rule per consequent."""
    held = {i[5:] for i in basket if i.startswith("gar::")}
    best = {}
    for r in rules.itertuples():
        if r.consequent in held or not set(r.antecedent) <= basket:
            continue
        if r.consequent not in best or r.confidence > best[r.consequent]["confidence"]:
            best[r.consequent] = {"guarantee": r.consequent, "confidence": round(r.confidence, 3),
                                  "lift": round(r.lift, 2), "because": [a.split("::", 1)[1] for a in r.antecedent]}
    return sorted(best.values(), key=lambda d: -d["confidence"])[:k]
