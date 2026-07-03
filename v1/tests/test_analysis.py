"""Tests for analysis/evaluation/preprocess additions (supervisor-feedback work)."""
import numpy as np
import pandas as pd
import pytest

from src import analysis, anomaly, evaluation, preprocess


def _df(n=500, seed=0):
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "T (degC)": rng.normal(10, 5, n),
            "rh (%)": rng.normal(70, 10, n),
            "p (mbar)": rng.normal(1000, 3, n),
        },
        index=idx,
    )


def test_distribution_stats_columns_and_symmetry():
    stats = analysis.distribution_stats(_df())
    assert set(stats.columns) == {"mean", "median", "std", "skew", "kurtosis", "min", "max"}
    assert list(stats.index) == ["T (degC)", "rh (%)", "p (mbar)"]
    assert abs(stats.loc["T (degC)", "skew"]) < 0.5  # normal data ~symmetric


def test_distribution_stats_detects_right_skew():
    idx = pd.date_range("2024-01-01", periods=1000, freq="h")
    rng = np.random.default_rng(1)
    skewed = pd.DataFrame({"wv (m/s)": rng.exponential(2.0, 1000)}, index=idx)
    stats = analysis.distribution_stats(skewed)
    assert stats.loc["wv (m/s)", "skew"] > 1.0


def test_data_availability_exact_counts():
    idx = pd.date_range("2024-01-01", periods=10, freq="h")
    df = pd.DataFrame(
        {"x": [1.0, 2.0, np.nan, -9999.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}, index=idx
    )
    rep = analysis.data_availability(df)
    assert rep.loc["x", "n_total"] == 10
    assert rep.loc["x", "n_missing"] == 1
    assert rep.loc["x", "n_sentinel"] == 1
    assert rep.loc["x", "n_real"] == 8
    assert abs(rep.loc["x", "available_pct"] - 80.0) < 1e-9


def test_analyze_all_sensors_one_row_per_sensor():
    df = _df()
    out = analysis.analyze_all_sensors(df)
    assert list(out.index) == list(df.columns)
    assert {"skew", "kurtosis", "zscore_anomalies", "top_corr_sensor", "top_corr_r"} <= set(
        out.columns
    )
    assert (out["zscore_anomalies"] >= 0).all()
    # top-correlation partner must be a different, real sensor
    for col in df.columns:
        assert out.loc[col, "top_corr_sensor"] in df.columns
        assert out.loc[col, "top_corr_sensor"] != col


def test_anomaly_imbalance_counts():
    flags = pd.Series([True, False, False, False, True])
    imb = evaluation.anomaly_imbalance(flags)
    assert imb["n_anomaly"] == 2
    assert imb["n_normal"] == 3
    assert imb["n_total"] == 5
    assert abs(imb["anomaly_pct"] - 40.0) < 1e-9


def test_compare_imputation_interpolation_beats_mean_on_ramp():
    idx = pd.date_range("2024-01-01", periods=200, freq="h")
    ramp = pd.Series(np.linspace(0, 100, 200), index=idx)  # smooth trend
    res = preprocess.compare_imputation(ramp)
    assert np.isfinite(res.loc["interpolation", "mae"])
    assert res.loc["interpolation", "mae"] <= res.loc["mean_fill", "mae"]


# --- G1: circular statistics for wind direction ---

def test_circular_zscore_detects_true_outlier_that_linear_misses():
    idx = pd.date_range("2024-01-01", periods=200, freq="h")
    rng = np.random.default_rng(0)
    ang = (rng.normal(0, 8, 200)) % 360      # tight cluster near 0°, wraps to ~352-360
    ang[100] = 180.0                          # a genuine opposite-direction outlier
    ang[150] = 175.0
    s = pd.Series(ang, index=idx)
    circ = analysis.circular_zscore_anomalies(s, threshold=3.0)
    lin = anomaly.zscore_anomalies(s, threshold=3.0)
    assert bool(circ.iloc[100]) and bool(circ.iloc[150])   # circular catches real outliers
    assert int(lin.sum()) == 0                              # linear (mean~180) misses them


def test_circular_linear_corr_detects_and_ranges():
    idx = pd.date_range("2024-01-01", periods=500, freq="h")
    rng = np.random.default_rng(1)
    ang = rng.uniform(0, 360, 500)
    s = pd.Series(ang, index=idx)
    related = pd.Series(np.cos(np.deg2rad(ang)) * 10 + rng.normal(0, 0.5, 500), index=idx)
    unrelated = pd.Series(rng.normal(0, 1, 500), index=idx)
    r_rel = analysis.circular_linear_corr(s, related)
    r_unrel = analysis.circular_linear_corr(s, unrelated)
    assert 0.0 <= r_rel <= 1.0 and 0.0 <= r_unrel <= 1.0
    assert r_rel > 0.8
    assert r_unrel < 0.3


def test_analyze_all_sensors_wd_uses_circular():
    idx = pd.date_range("2024-01-01", periods=500, freq="h")
    rng = np.random.default_rng(2)
    ang = (rng.normal(0, 8, 500)) % 360
    df = pd.DataFrame({"T (degC)": rng.normal(10, 5, 500), "wd (deg)": ang}, index=idx)
    out = analysis.analyze_all_sensors(df)
    # wd correlation is circular-linear (non-negative, in [0, 1]) — not signed Pearson
    assert 0.0 <= out.loc["wd (deg)", "top_corr_r"] <= 1.0
    assert out.loc["wd (deg)", "top_corr_sensor"] == "T (degC)"


def test_compare_imputation_requires_datetime_index():
    s = pd.Series(np.linspace(0, 10, 50))     # RangeIndex, not DatetimeIndex
    with pytest.raises((TypeError, ValueError)):
        preprocess.compare_imputation(s)
