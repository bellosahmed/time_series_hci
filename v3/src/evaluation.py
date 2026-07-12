"""Real-label evaluation for v3 — honest metrics for a rare, labelled anomaly problem.

Average precision (area under the PR curve) is the primary, threshold-free score for imbalanced
detection; `best_f1` reports the best achievable F1 and its operating point; `per_type_recall` breaks
recall down by attack type — the table that explains *why* a fusion of complementary channels helps.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def average_precision(scores, labels) -> float:
    """Area under the precision-recall curve (threshold-free; honest under heavy imbalance)."""
    return float(average_precision_score(labels, scores))


def best_f1(scores, labels) -> dict:
    """Best F1 over all thresholds and the precision/recall/threshold that achieve it."""
    p, r, t = precision_recall_curve(labels, scores)
    denom = p + r
    f1 = np.where(denom > 0, 2 * p * r / np.where(denom > 0, denom, 1), 0.0)
    i = int(np.nanargmax(f1))
    threshold = float(t[i]) if i < len(t) else float("inf")
    return {"f1": float(f1[i]), "precision": float(p[i]), "recall": float(r[i]), "threshold": threshold}


def point_adjust(pred, labels):
    """Point-adjust a binary prediction: if any window inside a true-attack segment is flagged, the
    whole segment counts as detected — the standard protocol for event-scale time-series anomalies."""
    pred = np.array(pred, dtype=int).copy()
    labels = np.asarray(labels)
    in_seg, start = False, 0
    for i, y in enumerate(labels):
        if y == 1 and not in_seg:
            in_seg, start = True, i
        elif y == 0 and in_seg:
            in_seg = False
            if pred[start:i].any():
                pred[start:i] = 1
    if in_seg and pred[start:].any():
        pred[start:] = 1
    return pred


def best_f1_point_adjusted(scores, labels) -> dict:
    """Best F1 over thresholds after point-adjusting each thresholded prediction."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    best = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "threshold": float("inf")}
    for t in np.unique(scores):
        pred = point_adjust((scores >= t).astype(int), labels)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        if f1 > best["f1"]:
            best = {"f1": f1, "precision": p, "recall": r, "threshold": float(t)}
    return best


def bootstrap_ci(scores, labels, metric: str = "ap", n: int = 1000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for a detection metric (default average precision) over windows."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    m = len(labels)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        if labels[idx].sum() == 0:
            continue                                  # skip degenerate resamples with no positives
        vals.append(average_precision(scores[idx], labels[idx]))
    if not vals:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def mcnemar_test(pred_a, pred_b, labels) -> dict:
    """Exact McNemar test comparing two binary predictions on the same labelled windows (ported
    from v1, where it distinguishes IsolationForest from z-score).

    Per window, each prediction is 'correct' if it equals the label. McNemar looks only at the
    *discordant* windows (one prediction right, the other wrong) and tests whether that split
    departs from 50/50 via an exact binomial test — a formal alternative to eyeballing whether two
    configs' bootstrap AP confidence intervals overlap. Returns the discordant counts and a
    two-sided p-value (1.0 when the two never disagree).
    """
    from scipy.stats import binomtest

    a = np.asarray(pred_a).astype(bool)
    b = np.asarray(pred_b).astype(bool)
    t = np.asarray(labels).astype(bool)
    correct_a = a == t
    correct_b = b == t
    only_a = int(np.sum(correct_a & ~correct_b))
    only_b = int(np.sum(~correct_a & correct_b))
    n = only_a + only_b
    if n == 0:
        p_value = 1.0
    else:
        p_value = float(binomtest(min(only_a, only_b), n, 0.5, alternative="two-sided").pvalue)
    return {"only_a_correct": only_a, "only_b_correct": only_b,
            "n_discordant": n, "p_value": p_value}


def per_type_recall(scores, labels, attack_type, threshold) -> dict:
    """Recall within each attack type at a fixed threshold — shows each channel's blind spots."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    attack_type = np.asarray(attack_type)
    pred = scores >= threshold
    out = {}
    for t in ("modification", "duplication", "mixed"):
        mask = (labels == 1) & (attack_type == t)
        if mask.sum():
            out[t] = float(pred[mask].mean())
    return out
