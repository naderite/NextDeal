"""DSO1.2 client segmentation (sales personas) for corporate clients.

Companies are described by mixed binary features - business category (one-hot), insurance
lines held (multi-hot), multi-product flag and contract history - and clustered with K-means
(on binary data K-means behaves like a K-modes with squared distances; the features are
deliberately kept binary so no scaling choice can drive the result).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

LINE_LABELS = {"AUTOMOBILE": "motor", "INCENDIE": "property / multi-risk", "GROUPE-MALADIE": "group health",
               "TRANSPORT": "transport", "RISQUES DIVERS": "liability & misc.", "ENGINEERING": "engineering",
               "VIE": "life"}


def company_features(companies: pd.DataFrame, categories=None, lines=None) -> pd.DataFrame:
    """Binary feature matrix (one row per company). Pass `categories` / `lines` to reuse a fitted layout."""
    cats = categories or sorted(companies["category"].unique())
    lns = lines or sorted({l for v in companies["branches"].fillna("") for l in filter(None, v.split(";"))})
    X = pd.DataFrame(0.0, index=companies["REF_pm"].to_numpy(), columns=[f"cat::{c}" for c in cats] +
                     [f"line::{l}" for l in lns] + ["multi_product", "has_history"])
    for r, (c, b, n, h) in enumerate(zip(companies["category"], companies["branches"].fillna(""),
                                         companies["n_products"], companies["has_history"])):
        if f"cat::{c}" in X.columns:
            X.iat[r, X.columns.get_loc(f"cat::{c}")] = 1.0
        for l in filter(None, b.split(";")):
            if f"line::{l}" in X.columns:
                X.iat[r, X.columns.get_loc(f"line::{l}")] = 1.0
        X.iat[r, X.columns.get_loc("multi_product")] = float(n > 1)
        X.iat[r, X.columns.get_loc("has_history")] = float(h)
    return X


def fit_segments(X: pd.DataFrame, k: int, seed: int = 0) -> KMeans:
    return KMeans(n_clusters=k, n_init=20, random_state=seed).fit(X.to_numpy())


def describe_segments(X: pd.DataFrame, labels: np.ndarray, companies: pd.DataFrame) -> pd.DataFrame:
    """One row per segment: size, dominant category, lines held, depth -> an automatic persona name."""
    rows = []
    for s in sorted(set(labels)):
        m = labels == s
        sub = X[m]
        cat_cols = [c for c in X.columns if c.startswith("cat::")]
        line_cols = [c for c in X.columns if c.startswith("line::")]
        top_cat = sub[cat_cols].mean().idxmax()
        line_share = sub[line_cols].mean().sort_values(ascending=False)
        main_lines = [LINE_LABELS.get(c[6:], c[6:].lower()) for c, v in line_share.items() if v >= 0.4][:2]
        depth = companies.loc[m, "n_products"].mean()
        name = f"{top_cat[5:].title()} · {' + '.join(main_lines) or 'mixed lines'}" + (" · multi-product" if depth >= 1.8 else "")
        rows.append({"segment": int(s), "persona": name, "companies": int(m.sum()),
                     "share_top_category": round(float(sub[top_cat].mean()), 2),
                     "avg_products": round(float(depth), 2),
                     "mono_product_%": round(float((companies.loc[m, "n_products"] == 1).mean() * 100), 0),
                     "lines_held_%": ", ".join(f"{LINE_LABELS.get(c[6:], c[6:])} {v:.0%}" for c, v in line_share.head(3).items() if v > 0)})
    return pd.DataFrame(rows).set_index("segment")
