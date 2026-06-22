"""
v2 evaluation upgrade — point-adjusted metrics, ensemble, and PR curves.

Three additions that make the anomaly evaluation rigorous and fair:
  * point_adjust  - the time-series-anomaly standard: if a detector flags ANY point inside a true
                    anomaly segment, the whole segment counts as detected. (Strict month-by-month
                    scoring unfairly punishes catching a 14-month crisis with a few sharp flags.)
  * ensemble      - combine detectors by voting; agreement raises precision.
  * pr_curve      - sweep the threshold to trace precision vs recall (and its area, PR-AUC).

Builds on `anomaly.score` (the P/R/F1/confusion helper). Importable; no plotting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .anomaly import score, Scores

try:                                   # NumPy 2.x renamed trapz -> trapezoid
    from numpy import trapezoid as _trapz
except ImportError:                    # NumPy < 2.0
    from numpy import trapz as _trapz


# --------------------------------------------------------------------------- point adjust
def point_adjust(pred: pd.Series, truth: pd.Series) -> pd.Series:
    """Point-adjustment: mark a whole true-anomaly segment detected if any point in it was flagged."""
    p = pred.reindex(truth.index).fillna(False).astype(bool).values.copy()
    t = truth.astype(bool).values
    n = len(t)
    i = 0
    while i < n:
        if t[i]:
            j = i
            while j < n and t[j]:
                j += 1
            if p[i:j].any():           # any hit inside the segment -> credit the whole segment
                p[i:j] = True
            i = j
        else:
            i += 1
    return pd.Series(p, index=truth.index)


def score_point_adjusted(pred: pd.Series, truth: pd.Series) -> Scores:
    """Precision/recall/F1/confusion after point-adjustment (the standard TS-anomaly metric)."""
    return score(point_adjust(pred, truth), truth)


# --------------------------------------------------------------------------- ensemble
def ensemble(flags_list: list[pd.Series], min_votes: int = 2) -> pd.Series:
    """Combine boolean detector outputs by voting: flag a month if >= min_votes detectors agree."""
    aligned = pd.concat([f.fillna(False).astype(int) for f in flags_list], axis=1)
    return aligned.sum(axis=1) >= min_votes


# --------------------------------------------------------------------------- PR curve
def pr_curve(scores: pd.Series, truth: pd.Series, n: int = 50,
             point_adjusted: bool = True) -> pd.DataFrame:
    """Sweep the threshold over a continuous anomaly score; return precision/recall at each."""
    s = scores.dropna()
    t = truth.reindex(s.index).astype(bool)
    qs = np.linspace(0.50, 0.999, n)
    rows = []
    for th in np.quantile(s.values, qs):
        pred = s > th
        sc = score_point_adjusted(pred, t) if point_adjusted else score(pred, t)
        rows.append({"threshold": float(th), "precision": sc.precision, "recall": sc.recall, "f1": sc.f1})
    return pd.DataFrame(rows)


def pr_auc(curve: pd.DataFrame) -> float:
    """Approximate area under the precision-recall curve (higher = better)."""
    d = curve.sort_values("recall")
    return float(_trapz(d["precision"].values, d["recall"].values))
