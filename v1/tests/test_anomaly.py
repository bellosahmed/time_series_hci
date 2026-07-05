"""Tests for anomaly detectors + parameter-sensitivity sweeps."""

import numpy as np
import pandas as pd
import pytest

from src import anomaly


def test_matrix_profile_flags_injected_discord():
    pytest.importorskip("stumpy")
    # a clean repeating daily sine (period 24) so most subsequences have close neighbours...
    n = 24 * 30
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    s = pd.Series(np.sin(np.arange(n) * 2 * np.pi / 24), index=idx)
    # ...then corrupt one day into a flat/odd shape that repeats nowhere -> a discord
    s.iloc[24 * 15:24 * 16] = 5.0
    flags = anomaly.matrix_profile_anomalies(s, window=24)
    assert flags.dtype == bool and flags.index.equals(s.index)
    # a discord anchor should be flagged within the corrupted window (allow the window's span)
    window_flagged = flags.iloc[24 * 14:24 * 16].any()
    assert window_flagged and flags.sum() < n            # flags the discord, not everything


def test_matrix_profile_short_series_is_safe():
    pytest.importorskip("stumpy")
    s = pd.Series([1.0, 2.0, 3.0])
    flags = anomaly.matrix_profile_anomalies(s, window=24)   # window > len -> no crash
    assert flags.dtype == bool and not flags.any()


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
