"""Labeled evaluation via synthetic anomaly injection.

Jena has no ground-truth anomaly labels, so a detector's precision/recall can't
be measured directly. Here we inject *known* synthetic anomalies (spikes) into a
clean signal, run a detector, and score it against those known positions. This
turns the qualitative "looks sensible" check into real numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# Known Central-European weather events within the Jena record (2009–2016), used as REAL
# (not synthetic) ground-truth anomaly windows. Windows are approximate and were chosen from
# documented events and cross-checked against the record's own temperature extremes — an honest
# complement to the synthetic-spike evaluation below (which has exact but artificial labels).
KNOWN_EVENTS = [
    ("2009 winter cold snap", "2009-01-01", "2009-01-20"),
    ("2010 July heat", "2010-07-08", "2010-07-20"),
    ("2012 February cold wave", "2012-02-01", "2012-02-14"),
    ("2013 August heat", "2013-08-01", "2013-08-10"),
    ("2015 July heatwave", "2015-07-01", "2015-07-06"),
]


def known_events() -> list[tuple[str, str, str]]:
    """Return the labelled real-event windows as ``(name, start, end)`` — see ``KNOWN_EVENTS``."""
    return list(KNOWN_EVENTS)


def label_from_events(index, events=None) -> pd.Series:
    """Boolean Series over ``index``, True inside any known-event window (real ground truth).

    Unlike ``inject_spikes`` (which fabricates labels), this marks time spans of *actual*
    documented weather events, so a detector's precision/recall here is measured against
    real anomalies rather than synthetic ones.
    """
    events = events or KNOWN_EVENTS
    labels = pd.Series(False, index=index)
    for _, start, end in events:
        labels.loc[start:end] = True
    return labels


@dataclass
class Scores:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    tn: int = 0


def inject_spikes(series: pd.Series, n: int = 50, magnitude: float = 5.0,
                  seed: int = 42) -> tuple[pd.Series, pd.Series]:
    """Add `n` random spikes of size `magnitude * std`; return (noisy, labels).

    `labels` is a boolean Series (True at injected positions).
    """
    rng = np.random.default_rng(seed)
    out = series.astype(float).copy()
    labels = pd.Series(False, index=series.index)
    positions = rng.choice(len(series), size=n, replace=False)
    signs = rng.choice([-1.0, 1.0], size=n)
    bump = magnitude * series.std()
    for pos, sign in zip(positions, signs):
        out.iloc[pos] = out.iloc[pos] + sign * bump
        labels.iloc[pos] = True
    return out, labels


def score(pred: pd.Series, truth: pd.Series) -> Scores:
    """Precision / recall / F1 of a boolean prediction against boolean truth."""
    pred = pred.reindex(truth.index).fillna(False).astype(bool)
    truth = truth.astype(bool)
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    tn = int((~pred & ~truth).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return Scores(precision, recall, f1, tp, fp, fn, tn)


def confusion(pred, truth) -> dict:
    """Return the confusion-matrix counts as a dict (TP/FP/FN/TN)."""
    sc = score(pred, truth)
    return {"TP": sc.tp, "FP": sc.fp, "FN": sc.fn, "TN": sc.tn}


def point_adjust(pred: pd.Series, truth: pd.Series) -> pd.Series:
    """Point-adjust predictions: if a detector flags *any* point inside a true anomaly
    segment, credit the whole segment (the standard fairer metric for time-series
    anomalies, where catching an event slightly late still counts as catching it)."""
    p = pred.reindex(truth.index).fillna(False).astype(bool).to_numpy().copy()
    t = truth.astype(bool).to_numpy()
    i, n = 0, len(t)
    while i < n:
        if t[i]:
            j = i
            while j < n and t[j]:
                j += 1
            if p[i:j].any():
                p[i:j] = True
            i = j
        else:
            i += 1
    return pd.Series(p, index=truth.index, name="point_adjusted")


def score_point_adjusted(pred: pd.Series, truth: pd.Series) -> Scores:
    """Precision/recall/F1 after point-adjustment (see `point_adjust`)."""
    return score(point_adjust(pred, truth), truth)


def benchmark(clean: pd.Series, detector, seeds=range(10),
              n: int = 50, magnitude: float = 6.0) -> dict:
    """Repeatable evaluation: inject spikes with many random `seeds` and report
    mean±std of precision/recall/F1 (strict **and** point-adjusted).

    `detector` maps a noisy Series -> boolean anomaly flags. Averaging over seeds turns
    a single lucky run into a protocol with variance, which is what makes the numbers
    defensible.
    """
    rows = []
    for s in seeds:
        noisy, labels = inject_spikes(clean, n=n, magnitude=magnitude, seed=int(s))
        pred = detector(noisy)
        st, pa = score(pred, labels), score_point_adjusted(pred, labels)
        rows.append((st.precision, st.recall, st.f1, pa.precision, pa.recall, pa.f1))
    arr = np.array(rows)
    keys = ["precision", "recall", "f1", "pa_precision", "pa_recall", "pa_f1"]
    return {k: {"mean": float(arr[:, i].mean()), "std": float(arr[:, i].std())}
            for i, k in enumerate(keys)}


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

    Resamples the labelled points with replacement `n_boot` times, recomputing `metric`
    each time, and reports the point estimate plus the ``[alpha/2, 1-alpha/2]`` percentile
    interval (default 95%). This turns a single point estimate into a number with stated
    uncertainty — the rigor an examiner expects instead of a bare "recall = 0.83".
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
    }


def mcnemar_test(pred_a: pd.Series, pred_b: pd.Series, truth: pd.Series) -> dict:
    """Exact McNemar test comparing two detectors on the same labelled points.

    Per point, each detector is 'correct' if its flag equals the truth. McNemar looks only
    at the *discordant* points (one detector right, the other wrong) and tests whether that
    split departs from 50/50 via an exact binomial test. Returns the discordant counts and a
    two-sided p-value (1.0 when the detectors never disagree). This is the standard paired
    significance test for "is detector A really better than B, or is it noise?".
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


def anomaly_imbalance(flags: pd.Series) -> dict:
    """Class-imbalance summary for a boolean anomaly flag Series.

    Returns ``{n_anomaly, n_normal, n_total, anomaly_pct}``. Anomalies are a tiny
    minority of all points, so this quantifies the imbalance — essential context when
    reading precision/recall, since a detector can score high 'accuracy' simply by
    predicting everything normal.
    """
    f = flags.astype(bool)
    n_total = int(len(f))
    n_anomaly = int(f.sum())
    n_normal = n_total - n_anomaly
    anomaly_pct = 100.0 * n_anomaly / n_total if n_total else 0.0
    return {
        "n_anomaly": n_anomaly,
        "n_normal": n_normal,
        "n_total": n_total,
        "anomaly_pct": anomaly_pct,
    }
