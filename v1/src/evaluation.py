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


@dataclass
class Scores:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


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
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return Scores(precision, recall, f1, tp, fp, fn)
