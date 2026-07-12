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


def test_mahalanobis_flags_cross_sensor_inconsistency():
    # three highly-correlated sensors (share a latent signal) ...
    rng = np.random.default_rng(0)
    n = 500
    latent = rng.normal(0.0, 1.0, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    df = pd.DataFrame(
        {
            "s1": latent + rng.normal(0, 0.05, n),
            "s2": latent + rng.normal(0, 0.05, n),
            "s3": latent + rng.normal(0, 0.05, n),
        },
        index=idx,
    )
    # ... break the cross-sensor consistency at a few rows: push ONE sensor off its partners.
    # The offset (1.0) is within each sensor's own ~1-sigma range, so a per-sensor z-score barely
    # reacts, but jointly the row is far off the covariance -> Mahalanobis catches it.
    bad = [100, 250, 400]
    df.iloc[bad, df.columns.get_loc("s2")] += 1.0
    flags = anomaly.mahalanobis_anomalies(df, contamination=0.02)
    assert flags.dtype == bool and flags.index.equals(df.index)
    assert all(bool(flags.iloc[b]) for b in bad)          # the cross-sectional faults are caught
    assert 0 < flags.sum() < n                             # flags the faults, not everything


def test_mahalanobis_contamination_controls_count():
    rng = np.random.default_rng(1)
    n = 1000
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    df = pd.DataFrame(rng.normal(0, 1, (n, 4)), index=idx, columns=list("abcd"))
    few = anomaly.mahalanobis_anomalies(df, contamination=0.01).sum()
    more = anomaly.mahalanobis_anomalies(df, contamination=0.05).sum()
    assert more > few                                      # higher contamination -> more flags


def test_flatline_flags_frozen_sensor():
    rng = np.random.default_rng(0)
    n = 300
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    df = pd.DataFrame(
        {"s1": rng.normal(0, 1, n), "s2": rng.normal(0, 1, n)}, index=idx
    )
    # freeze s2 (stuck sensor) for a stretch -> its short-window variance collapses to 0
    j = df.columns.get_loc("s2")
    df.iloc[150:170, j] = df.iloc[150, j]
    flags = anomaly.flatline_anomalies(df, window=4)
    assert flags.dtype == bool and flags.index.equals(df.index)
    assert flags.iloc[150:170].sum() >= 10                 # most of the frozen stretch flagged
    assert flags.iloc[50:70].sum() <= 2                    # a normal stretch stays (mostly) clean


def test_flatline_clean_data_flags_few():
    rng = np.random.default_rng(2)
    n = 500
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    df = pd.DataFrame(rng.normal(0, 1, (n, 3)), index=idx, columns=list("abc"))
    flags = anomaly.flatline_anomalies(df, window=4)
    assert flags.dtype == bool
    assert flags.sum() < 0.05 * n                          # no freeze -> almost nothing flagged


def test_sensor_network_portfolio_catches_both_fault_types():
    # The notebook-05 recommendation: Mahalanobis (cross-sectional) + flat-line (variance) together
    # cover fault types that neither alone does. Build data with BOTH a cross-sensor spike and a
    # frozen-sensor stretch, and check the portfolio catches each.
    rng = np.random.default_rng(0)
    n = 600
    latent = rng.normal(0.0, 1.0, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    df = pd.DataFrame(
        {
            "s1": latent + rng.normal(0, 0.05, n),
            "s2": latent + rng.normal(0, 0.05, n),
            "s3": latent + rng.normal(0, 0.05, n),
        },
        index=idx,
    )
    cross = [120, 480]
    df.iloc[cross, df.columns.get_loc("s2")] += 1.0        # cross-sectional inconsistency (spike)
    freeze = slice(300, 320)
    df.iloc[freeze, df.columns.get_loc("s3")] = df.iloc[300, df.columns.get_loc("s3")]  # stuck sensor

    flags = anomaly.sensor_network_anomalies(df, contamination=0.02, window=4)
    assert flags.dtype == bool and flags.index.equals(df.index)
    assert all(bool(flags.iloc[c]) for c in cross)         # cross-sectional caught (Mahalanobis)
    assert flags.iloc[300:320].sum() >= 10                 # frozen stretch caught (flat-line)
    assert flags.sum() < n                                 # still selective, not everything


def test_network_culprit_names_the_faulted_sensor():
    rng = np.random.default_rng(0)
    n = 200
    latent = rng.normal(0.0, 1.0, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    df = pd.DataFrame(
        {
            "s1": latent + rng.normal(0, 0.05, n),
            "s2": latent + rng.normal(0, 0.05, n),
            "s3": latent + rng.normal(0, 0.05, n),
        },
        index=idx,
    )
    ts = idx[100]
    df.loc[ts, "s2"] += 1.0                                # push s2 off its correlated partners
    assert anomaly.network_culprit(df, ts) == "s2"         # attribution names the faulted sensor


def test_flatline_culprit_names_the_frozen_sensor():
    rng = np.random.default_rng(0)
    n = 300
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    df = pd.DataFrame({"s1": rng.normal(0, 1, n), "s2": rng.normal(0, 1, n)}, index=idx)
    j = df.columns.get_loc("s2")
    df.iloc[150:170, j] = df.iloc[150, j]                   # freeze s2
    ts = idx[165]                                           # inside the frozen stretch
    assert anomaly.flatline_culprit(df, ts, window=4) == "s2"


def test_flatline_culprit_none_when_nothing_is_frozen():
    rng = np.random.default_rng(2)
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    df = pd.DataFrame(rng.normal(0, 1, (n, 3)), index=idx, columns=list("abc"))
    assert anomaly.flatline_culprit(df, idx[100], window=4) is None
