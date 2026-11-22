"""Shared helpers for the NextDeal / BH Assurance notebooks.

Kept deliberately small: paths, a consistent chart style, text normalisation
and ranking metrics. Every notebook imports from here so figures and cleaning
rules stay identical across the project.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
SYNTHETIC = ROOT / "data" / "synthetic"      # generated evaluation sets, never presented as business data
FIGURES = ROOT / "figures"
REPORTS = ROOT / "reports"
LOGS = REPORTS / "logs"
MODELS = ROOT / "models"
for _p in (PROCESSED, SYNTHETIC, FIGURES, REPORTS, LOGS, MODELS):
    _p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- style
# Categorical slots are used in this fixed order (validated for colour-vision deficiency).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8983"
GRID = "#e6e5e0"
SURFACE = "#fcfcfb"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def set_style() -> None:
    """Matplotlib defaults shared by every figure of the project."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": GRID,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "text.color": INK,
        "legend.frameon": False,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
    })


def save(fig, name: str) -> Path:
    """Save a figure into /figures with tight bounding box and return its path."""
    path = FIGURES / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    return path


def subtitle(ax, text: str) -> None:
    """Grey one-line takeaway under the chart title."""
    ax.text(0, 1.015, text, transform=ax.transAxes, fontsize=9.5, color=INK_2, va="bottom")


# --------------------------------------------------------------------------- text
_PLACEHOLDERS = {"", "NON DEFINI", "NON DEFINIE", "NON DÉFINI", "NAN", "NONE", "NULL", "N/A", "-"}


def normalize_label(value):
    """Upper-case, strip accents and collapse whitespace; placeholders -> NaN.

    'STATION  DE SERVICE' -> 'STATION DE SERVICE', 'Non défini' -> NaN.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"\s+", " ", text).upper().strip()
    return np.nan if text in _PLACEHOLDERS else text


def split_multi(value, sep: str = ";") -> list[str]:
    """'A;B; A ' -> ['A', 'B'] (deduplicated, normalised, order kept)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    out = []
    for part in str(value).split(sep):
        norm = normalize_label(part)
        if isinstance(norm, str) and norm not in out:
            out.append(norm)
    return out


# --------------------------------------------------------------------------- metrics
def rank_of_true(scores: np.ndarray, true_idx: int, mask: np.ndarray | None = None) -> int:
    """1-based rank of the true item once masked (already-owned) items are removed."""
    s = scores.astype(float).copy()
    if mask is not None:
        s[mask] = -np.inf
    # ties broken pessimistically: equal scores count as ranked above the truth
    return int((s > s[true_idx]).sum() + (s == s[true_idx]).sum())


def ranking_report(ranks) -> dict:
    """HitRate@1/3/5, NDCG@3 and MRR from a list of 1-based ranks."""
    r = np.asarray(ranks, dtype=float)
    return {
        "HitRate@1": float(np.mean(r <= 1)),
        "HitRate@3": float(np.mean(r <= 3)),
        "HitRate@5": float(np.mean(r <= 5)),
        "NDCG@3": float(np.mean(np.where(r <= 3, 1 / np.log2(r + 1), 0.0))),
        "MRR": float(np.mean(1 / r)),
    }


# --------------------------------------------------------------------------- uncertainty
def mean_std(values, digits: int = 3) -> str:
    """'0.690 ± 0.034' over folds (population std, as in notebook 04)."""
    v = np.asarray(values, dtype=float)
    return f"{v.mean():.{digits}f} ± {v.std():.{digits}f}"


def cluster_bootstrap_ci(values, groups=None, n_boot: int = 2000, seed: int = 0, level: float = 0.95):
    """Percentile CI of the mean, resampling whole groups (clients) with replacement.

    Episodes of the same company are correlated, so the resampling unit is the company,
    not the episode. Without `groups` it falls back to an ordinary bootstrap.
    """
    v = np.asarray(values, dtype=float)
    g = np.arange(len(v)) if groups is None else np.asarray(groups)
    uniq, inv = np.unique(g, return_inverse=True)
    sums = np.bincount(inv, weights=v, minlength=len(uniq))
    counts = np.bincount(inv, minlength=len(uniq)).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    boots = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a))


def wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a proportion k / n (well behaved for small n or p near 0/1)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return float(centre - half), float(centre + half)


SYNTHETIC_BADGE = "SYNTHETIC / PROXY EVALUATION — pipeline check, not business evidence"
