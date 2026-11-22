"""Feature pipeline for the next-product (cross-sell) model.

Shared by notebooks 03 (demonstration) and 04 (cross-validated modeling) so that
the exact same encoding is used everywhere. Every encoder is *fitted* on the
training fold only, so no information leaks from test companies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

PROFILE_CATEGORICAL = ["category", "sector"]
HIGH_CARDINALITY = ["activity"]
NUMERIC = ["n_visible", "has_history", "max_contract_age", "sector_imputed"]


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replace each level by its frequency in the *training* data (unseen -> 0)."""

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = list(X.columns)
        self.maps_ = {c: X[c].value_counts(normalize=True).to_dict() for c in self.columns_}
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns_)
        return np.column_stack([X[c].map(self.maps_[c]).fillna(0.0).to_numpy(float) for c in self.columns_])

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{c}_freq" for c in self.columns_])


class PortfolioMultiHot(BaseEstimator, TransformerMixin):
    """'A;B' (visible products) -> multi-hot vector over the product catalogue."""

    def __init__(self, products=None):
        self.products = products

    def fit(self, X, y=None):
        self.index_ = {p: i for i, p in enumerate(self.products)}
        return self

    def transform(self, X):
        col = pd.DataFrame(X).iloc[:, 0].fillna("")
        out = np.zeros((len(col), len(self.products)), dtype=np.float32)
        for r, value in enumerate(col):
            for p in filter(None, str(value).split(";")):
                if p in self.index_:
                    out[r, self.index_[p]] = 1.0
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array([f"has::{p}" for p in self.products])


def build_preprocessor(products: list[str]) -> ColumnTransformer:
    """One-hot (category, sector) + frequency (activity) + multi-hot portfolio + numeric."""
    return ColumnTransformer([
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5, sparse_output=False),
         PROFILE_CATEGORICAL),
        ("freq", FrequencyEncoder(), HIGH_CARDINALITY),
        ("portfolio", PortfolioMultiHot(products), ["visible_portfolio"]),
        ("num", "passthrough", NUMERIC),
    ])


# --------------------------------------------------------------------------- validation 2 additions
# Two feature families added after validation 1 (evaluation notebook not yet included):
#   * the insurance *lines* of the visible portfolio (7 lines) - product-family information;
#   * the *words* of the sector and activity labels ("BATIMENT", "TRANSPORT", "CLINIQUE"...),
#     which let rare activities share signal with frequent ones instead of being reduced to a frequency.
from sklearn.feature_extraction.text import CountVectorizer

WORD_PATTERN = r"(?u)\b[A-Z]{4,}\b"      # labels are upper-case and accent-free after normalize_label


def add_profile_columns(df: pd.DataFrame, product_branch: dict) -> pd.DataFrame:
    """Derived inputs used by the v2 models: visible lines + free-text profile."""
    out = df.copy()
    out["visible_portfolio"] = out["visible_portfolio"].fillna("")
    out["visible_lines"] = out["visible_portfolio"].map(
        lambda v: ";".join(sorted({product_branch[p] for p in filter(None, v.split(";")) if p in product_branch})))
    out["profile_text"] = (out["sector"].fillna("") + " " + out["activity"].fillna("")).str.replace("UNKNOWN", "", regex=False)
    return out


def build_preprocessor_v2(products: list[str], lines: list[str]) -> ColumnTransformer:
    """v1 blocks + multi-hot of held lines + binary bag of profile words (min 2 companies)."""
    return ColumnTransformer([
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5, sparse_output=False),
         PROFILE_CATEGORICAL),
        ("freq", FrequencyEncoder(), HIGH_CARDINALITY),
        ("portfolio", PortfolioMultiHot(products), ["visible_portfolio"]),
        ("num", "passthrough", NUMERIC),
        ("lines", PortfolioMultiHot(lines), ["visible_lines"]),
        ("words", CountVectorizer(min_df=2, binary=True, token_pattern=WORD_PATTERN, lowercase=False), "profile_text"),
    ])
