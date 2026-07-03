"""Tests for analysis/evaluation/preprocess additions (supervisor-feedback work)."""
import numpy as np
import pandas as pd

from src import analysis, evaluation, preprocess


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
