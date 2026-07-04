"""Tests for anomaly detectors + parameter-sensitivity sweeps."""

import numpy as np
import pandas as pd

from src import anomaly


def test_zscore_borderline_flags_near_threshold_only():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(0.0, 1.0, 1000))
    s.iloc[10] = 6.0   # ~6 sigma -> a clear anomaly (not borderline)
    s.iloc[30] = 3.5   # ~3.5 sigma -> borderline for threshold 4 (0.75*4=3.0 < 3.5 <= 4)
    thr = 4.0
    bl = anomaly.zscore_borderline(s, threshold=thr)
    an = anomaly.zscore_anomalies(s, threshold=thr)
    assert bl.dtype == bool
    assert bool(an.iloc[10]) and not bool(bl.iloc[10])   # anomaly, not borderline
    assert bool(bl.iloc[30]) and not bool(an.iloc[30])   # borderline, not anomaly
    assert not (bl & an).any()                            # the two sets are disjoint


def test_threshold_sensitivity_monotone():
    idx = pd.date_range("2024-01-01", periods=500, freq="h")
    s = pd.Series(np.random.default_rng(0).normal(0, 1, 500), index=idx)
    sens = anomaly.threshold_sensitivity(s, thresholds=(2.0, 3.0, 4.0))
    assert sens[2.0] >= sens[3.0] >= sens[4.0]      # higher threshold -> fewer flags


def test_contamination_sensitivity_monotone():
    idx = pd.date_range("2024-01-01", periods=300, freq="h")
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.normal(0, 1, 300), "b": rng.normal(0, 1, 300)}, index=idx)
    sens = anomaly.contamination_sensitivity(df, contaminations=(0.01, 0.05))
    assert sens[0.05] >= sens[0.01]
