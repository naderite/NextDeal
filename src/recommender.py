"""DSO1.1 (v2) next-product recommender for corporate clients.

Two complementary scorers, blended:

* **Multinomial logistic regression** on the v2 features (`features.build_preprocessor_v2`):
  one weight vector per product, good at the frequent products.
* **Pairwise candidate ranker**: one row per (episode, candidate product) described by
  *candidate-level* features (popularity, sector / category / activity-word propensity,
  co-occurrence with the visible portfolio, line already held...). Because the parameters are
  shared across products, rare products borrow strength from frequent ones.

An optional **long-tail slot** reserves the 3rd proposal for a product outside the training
best-sellers when the first two are both best-sellers - a business rule ("two safe pitches,
one discovery") whose cost on overall HitRate@3 is to be measured in a future evaluation notebook.

Every statistic used as a feature is computed on training rows only; for the training rows of the
pairwise ranker themselves, the tables are computed *out-of-fold* (inner GroupKFold by company) so
a row never sees its own target.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

from .features import WORD_PATTERN, add_profile_columns, build_preprocessor_v2

_WORD_RE = re.compile(WORD_PATTERN)


def visible_matrix(df: pd.DataFrame, index: dict) -> np.ndarray:
    """Boolean (rows x products) matrix of the products visible in `visible_portfolio`."""
    out = np.zeros((len(df), len(index)), dtype=bool)
    for r, v in enumerate(df["visible_portfolio"].fillna("")):
        for p in filter(None, v.split(";")):
            if p in index:
                out[r, index[p]] = True
    return out


class PairwiseRanker:
    """Gradient-boosted scorer of (episode, candidate) pairs."""

    FEATURES = ["log_popularity", "log_rate_sector", "log_rate_category", "log_rate_words_mean",
                "log_rate_words_max", "log_cooccurrence", "candidate_line", "held_in_candidate_line",
                "n_visible", "is_warm", "lift_sector", "lift_words"]

    def __init__(self, products, product_branch, smoothing: float = 3.0, inner_folds: int = 5, random_state: int = 0):
        self.products = list(products)
        self.product_branch = dict(product_branch)
        self.smoothing = smoothing
        self.inner_folds = inner_folds
        self.random_state = random_state
        self.index_ = {p: i for i, p in enumerate(self.products)}
        self.lines_ = sorted(set(self.product_branch.values()))
        line_idx = {l: i for i, l in enumerate(self.lines_)}
        self.cand_line_ = np.array([line_idx.get(self.product_branch.get(p, ""), len(self.lines_)) for p in self.products])

    # ------------------------------------------------------------------ statistics from training rows
    def _tables(self, df: pd.DataFrame, y: np.ndarray) -> dict:
        n_p = len(self.products)
        prior = (np.bincount(y, minlength=n_p) + 1.0) / (len(y) + n_p)
        m = self.smoothing

        def table(keys_per_row):
            acc = {}
            for keys, t in zip(keys_per_row, y):
                for k in keys:
                    acc.setdefault(k, np.zeros(n_p))[t] += 1
            return {k: (v + m * prior) / (v.sum() + m) for k, v in acc.items()}

        words = [set(_WORD_RE.findall(t)) for t in df["profile_text"]]
        vis = visible_matrix(df, self.index_)
        C = np.zeros((n_p, n_p))
        for r, t in enumerate(y):
            C[vis[r], t] += 1
        cooc = (C + 0.1 * prior) / (C.sum(axis=1, keepdims=True) + 0.1)
        return {"prior": prior, "sector": table([[s] for s in df["sector"]]),
                "category": table([[c] for c in df["category"]]), "words": table(words), "cooc": cooc}

    def _features(self, df: pd.DataFrame, tabs: dict) -> np.ndarray:
        prior = tabs["prior"]
        vis = visible_matrix(df, self.index_)
        out = np.zeros((len(df), len(self.products), len(self.FEATURES)))
        for r, (sector, category, text) in enumerate(zip(df["sector"], df["category"], df["profile_text"])):
            s = tabs["sector"].get(sector, prior)
            c = tabs["category"].get(category, prior)
            ws = [tabs["words"][w] for w in set(_WORD_RE.findall(text)) if w in tabs["words"]]
            w_mean = np.mean(ws, axis=0) if ws else prior
            w_max = np.max(ws, axis=0) if ws else prior
            co = vis[r].astype(float) @ tabs["cooc"] / vis[r].sum() if vis[r].any() else prior
            held_lines = np.bincount(self.cand_line_[vis[r]], minlength=len(self.lines_) + 1)
            out[r] = np.column_stack([
                np.log(prior), np.log(s), np.log(c), np.log(w_mean), np.log(w_max), np.log(co),
                self.cand_line_, held_lines[self.cand_line_], np.full(len(prior), vis[r].sum()),
                np.full(len(prior), float(vis[r].any())), np.log(s) - np.log(prior), np.log(w_mean) - np.log(prior)])
        return out

    # ------------------------------------------------------------------ fit / score
    def fit(self, df: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> "PairwiseRanker":
        y = np.asarray(y)
        F = np.zeros((len(df), len(self.products), len(self.FEATURES)))
        for itr, ite in GroupKFold(self.inner_folds).split(df, groups=groups):
            F[ite] = self._features(df.iloc[ite], self._tables(df.iloc[itr], y[itr]))
        self.tables_ = self._tables(df, y)
        mask = ~visible_matrix(df, self.index_)                 # products already held are never candidates
        target = np.arange(len(self.products))[None, :] == y[:, None]
        self.clf_ = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3, l2_regularization=1.0,
                                                   categorical_features=[self.FEATURES.index("candidate_line")],
                                                   random_state=self.random_state)
        self.clf_.fit(F[mask], target[mask])
        return self

    def predict_scores(self, df: pd.DataFrame) -> np.ndarray:
        F = self._features(df, self.tables_)
        p = self.clf_.predict_proba(F.reshape(-1, F.shape[2]))[:, 1].reshape(len(df), len(self.products))
        return p / p.sum(axis=1, keepdims=True)


class HybridRecommender:
    """Blend `w_lr * LR + (1 - w_lr) * pairwise`, then optional long-tail slot at position 3."""

    def __init__(self, products, product_branch, w_lr: float = 0.7, reserve_long_tail: bool = False,
                 C: float = 0.3, n_best_sellers: int = 3, random_state: int = 0):
        self.products = list(products)
        self.product_branch = dict(product_branch)
        self.w_lr = w_lr
        self.reserve_long_tail = reserve_long_tail
        self.C = C
        self.n_best_sellers = n_best_sellers
        self.random_state = random_state
        self.index_ = {p: i for i, p in enumerate(self.products)}
        self.lines_ = sorted(set(self.product_branch.values()))

    def fit(self, df: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> "HybridRecommender":
        df = add_profile_columns(df, self.product_branch)
        y = np.asarray(y)
        self.popularity_ = np.bincount(y, minlength=len(self.products)).astype(float)
        self.best_sellers_ = set(np.argsort(-self.popularity_)[: self.n_best_sellers].tolist())
        self.lr_ = None
        if self.w_lr > 0:
            self.lr_ = make_pipeline(build_preprocessor_v2(self.products, self.lines_),
                                     LogisticRegression(C=self.C, max_iter=4000))
            self.lr_.fit(df, y)
        self.pw_ = PairwiseRanker(self.products, self.product_branch, random_state=self.random_state).fit(df, y, groups) \
            if self.w_lr < 1 else None
        return self

    def predict_scores(self, df: pd.DataFrame) -> np.ndarray:
        df = add_profile_columns(df, self.product_branch)
        out = np.zeros((len(df), len(self.products)))
        if self.lr_ is not None:
            lr = np.zeros_like(out)
            lr[:, self.lr_.classes_] = self.lr_.predict_proba(df)
            out += self.w_lr * lr
        if self.pw_ is not None:
            out += (1 - self.w_lr) * self.pw_.predict_scores(df)
        return out + 1e-6 * self.popularity_ / self.popularity_.max()      # deterministic tie-break

    def ordered(self, scores: np.ndarray, held_mask: np.ndarray) -> np.ndarray:
        """Candidate indices best-first (held products removed), long-tail slot applied if enabled."""
        s = scores.astype(float).copy()
        s[held_mask] = -np.inf
        order = [int(i) for i in np.argsort(-s, kind="stable") if np.isfinite(s[i])]
        if self.reserve_long_tail and len(order) > 3 and all(i in self.best_sellers_ for i in order[:3]):
            j = next((j for j in range(3, len(order)) if order[j] not in self.best_sellers_), None)
            if j is not None:
                order.insert(2, order.pop(j))
        return np.array(order, dtype=int)
