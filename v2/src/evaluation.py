"""
v2 evaluation upgrade — point-adjusted metrics, ensemble, PR curves, and statistical rigor.

Additions that make the anomaly evaluation rigorous and fair:
  * point_adjust  - the time-series-anomaly standard: if a detector flags ANY point inside a true
                    anomaly segment, the whole segment counts as detected. (Strict month-by-month
                    scoring unfairly punishes catching a 14-month crisis with a few sharp flags.)
  * ensemble      - combine detectors by voting; agreement raises precision.
  * pr_curve      - sweep the threshold to trace precision vs recall (and its area, PR-AUC).
  * bootstrap_ci  - percentile bootstrap confidence interval for precision/recall/F1 (ported from
                    v1: on a 292-point labelled set, a bare point estimate is not defensible).
  * mcnemar_test  - paired significance test for "is detector A really better than B, or is it
                    noise?" (ported from v1, where it distinguishes IsolationForest from z-score).

Builds on `anomaly.score` (the P/R/F1/confusion helper). Importable; no plotting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .anomaly import Scores, score

try:                                   # NumPy 2.x renamed trapz -> trapezoid
    from numpy import trapezoid as _trapz
except ImportError:                    # NumPy < 2.0
    from numpy import trapz as _trapz  # type: ignore[no-redef]


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


# --------------------------------------------------------------------------- statistical rigor
def _metric_from_bool(pred: np.ndarray, truth: np.ndarray, metric: str) -> float:
    """precision / recall / f1 from aligned boolean arrays (helper for bootstrap)."""
    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    if metric == "precision":
        return prec
    if metric == "recall":
        return rec
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def bootstrap_ci(pred: pd.Series, truth: pd.Series, metric: str = "recall",
                 n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Percentile bootstrap confidence interval for a detector's precision/recall/f1.

    Resamples the labelled months with replacement `n_boot` times, recomputing `metric` each
    time, and reports the point estimate plus the ``[alpha/2, 1-alpha/2]`` percentile interval
    (default 95%). On 292 monthly points with only 14 labelled as crisis months, a bare point
    estimate ("recall = 0.36") is not defensible without a stated uncertainty.
    """
    truth_b = truth.astype(bool)
    p = pred.reindex(truth.index).fillna(False).astype(bool).to_numpy()
    t = truth_b.to_numpy()
    point = _metric_from_bool(p, t, metric)
    rng = np.random.default_rng(seed)
    n = len(t)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[i] = _metric_from_bool(p[idx], t[idx], metric)
    return {
        "metric": metric,
        "point": float(point),
        "lower": float(np.quantile(boot, alpha / 2)),
        "upper": float(np.quantile(boot, 1 - alpha / 2)),
        "n_boot": n_boot,
        "n": int(n),   # labelled sample size — the driver of the interval width
    }


def mcnemar_test(pred_a: pd.Series, pred_b: pd.Series, truth: pd.Series) -> dict:
    """Exact McNemar test comparing two detectors on the same labelled months.

    Per month, each detector is 'correct' if its flag equals the truth. McNemar looks only at
    the *discordant* months (one detector right, the other wrong) and tests whether that split
    departs from 50/50 via an exact binomial test. Returns the discordant counts and a two-sided
    p-value (1.0 when the detectors never disagree) — the standard paired significance test for
    "is detector A really better than B, or is it noise?".
    """
    t = truth.astype(bool)
    a = pred_a.reindex(truth.index).fillna(False).astype(bool)
    b = pred_b.reindex(truth.index).fillna(False).astype(bool)
    correct_a = (a == t).to_numpy()
    correct_b = (b == t).to_numpy()
    only_a = int(np.sum(correct_a & ~correct_b))
    only_b = int(np.sum(~correct_a & correct_b))
    n = only_a + only_b
    if n == 0:
        p_value = 1.0
    else:
        from scipy.stats import binomtest
        p_value = float(binomtest(min(only_a, only_b), n, 0.5, alternative="two-sided").pvalue)
    return {"only_a_correct": only_a, "only_b_correct": only_b,
            "n_discordant": n, "p_value": p_value}
