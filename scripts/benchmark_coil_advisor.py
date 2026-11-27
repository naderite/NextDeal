"""Evaluate the advisor ranker on the official UCI CoIL holdout."""
import argparse
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from prepare_coil import FAMILIES, PRODUCTS, SOURCE, prepare
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def benchmark(holdout_path, labels_path):
    from backend.db import init
    from backend.services import engine
    from src.utils import normalize_label
    from src.segments import company_features

    holdout = np.loadtxt(holdout_path, dtype=int, ndmin=2)
    labels = np.loadtxt(labels_path, dtype=int, ndmin=1)
    if holdout.shape != (4000, 85) or labels.shape != (4000,):
        raise ValueError("Expected the publisher's 4,000-row CoIL test data and labels")
    init()
    e = engine()
    held = np.column_stack((holdout[:, 64:], labels)) > 0
    rows, targets = np.nonzero(held & (held.sum(axis=1) >= 2)[:, None])
    visible = held[rows].copy()
    visible[np.arange(len(targets)), targets] = False
    products = np.array(PRODUCTS)
    portfolios = [products[mask].tolist() for mask in visible]
    categories = [normalize_label(f"Subtype {value}") for value in holdout[rows, 0]]
    frame = pd.DataFrame({"category": categories, "activity": categories,
                          "sector": [normalize_label(f"Group {value}") for value in holdout[rows, 4]],
                          "visible_portfolio": [";".join(held) for held in portfolios],
                          "n_visible": visible.sum(axis=1), "has_history": 0,
                          "max_contract_age": 0, "sector_imputed": 0})
    scores = e.recommender.predict_proba(frame)
    hits = [products[target] in [p["product"] for p in e.rank_products(portfolio, score)[:3]]
            for target, portfolio, score in zip(targets, portfolios, scores, strict=True)]
    counts = e.episodes.hidden_product.value_counts().reindex(PRODUCTS, fill_value=0).to_numpy()
    tail = ~np.isin(targets, np.argsort(-counts, kind="stable")[:3])
    # Predict memberships directly with the app's training-only centroids.
    families = np.array(FAMILIES)
    test_profiles = pd.DataFrame({
        "REF_pm": np.arange(len(holdout)),
        "category": [f"Subtype {value}" for value in holdout[:, 0]],
        "branches": [";".join(sorted(set(families[mask]))) for mask in held],
        "n_products": held.sum(axis=1), "has_history": 0,
    })
    features = company_features(test_profiles).reindex(columns=e.segment_features.columns, fill_value=0)
    categories = [name for name in features if name.startswith("cat::")]
    features[categories] = features[categories].astype(float) * e.segment_category_weight
    memberships = e.segmenter.predict(features)
    alternate = KMeans(n_clusters=e.segmenter.n_clusters, n_init=20, random_state=42).fit(e.segment_features)
    counts_by_cluster = np.bincount(memberships, minlength=e.segmenter.n_clusters)
    segments = {
        "category_weight": e.segment_category_weight,
        "test_silhouette": float(silhouette_score(features, memberships)),
        "test_segment_sizes": counts_by_cluster.tolist(),
        "stability_ari_across_initializations": float(adjusted_rand_score(memberships, alternate.predict(features))),
        "target_silhouette": .25,
    }
    return {"source": SOURCE, "training_portfolios": len(e.companies), "test_portfolios": len(holdout),
            "episodes": len(rows), "long_tail_episodes": int(tail.sum()), "hit_rate_at_3": float(np.mean(hits)),
            "long_tail_hit_rate_at_3": float(np.mean(np.array(hits)[tail])),
            "segmentation": segments,
            "scope": "Independent publisher split; reconstructs existing consumer portfolio holdings."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory with the three CoIL files")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with TemporaryDirectory(prefix="nextdeal-coil-benchmark-") as work:
        base = Path(work)
        training = args.directory / "coil-training.tsv"
        prepare(training, base / "data")
        os.environ["NEXTDEAL_DATA_DIR"] = str(base / "data")
        os.environ["NEXTDEAL_RUNTIME"] = str(base / "runtime")
        os.environ["NEXTDEAL_MAINTENANCE"] = "0"
        result = benchmark(args.directory / "coil-test.tsv", args.directory / "coil-test-target.tsv")
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
