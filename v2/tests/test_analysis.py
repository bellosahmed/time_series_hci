"""Tests for v2 behaviour analysis. Run: cd code/v2 && python3.12 -m pytest tests/"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2))
from src import preprocess as pp
from src import analysis as an

DATA = V2.parent / "data" / "eu_trade_monthly.csv"


@pytest.fixture(scope="module")
def df():
    return pp.load_trade(DATA)


def test_trend_slope_positive_line():
    s = pd.Series(np.arange(50) * 2.0 + 5)        # slope is exactly 2
    assert abs(an.trend_slope(s) - 2.0) < 1e-9


def test_trend_slopes_returns_all_flows(df):
    sl = an.trend_slopes(df)
    assert len(sl) == 54
    assert sl.is_monotonic_increasing            # sorted ascending


def test_decompose_components_reconstruct(df):
    stl = an.decompose(df["DE_EXP"], period=12)
    recon = stl.trend + stl.seasonal + stl.resid
    pd.testing.assert_series_equal(recon, df["DE_EXP"], check_names=False)


def test_strengths_in_unit_range(df):
    stl = an.decompose(df["FR_EXP"], period=12)
    assert 0.0 <= an.trend_strength(stl) <= 1.0
    assert 0.0 <= an.seasonal_strength(stl) <= 1.0


def test_adf_detects_trend_vs_noise():
    rng = np.random.default_rng(0)
    trend = pd.Series(np.arange(200) + rng.normal(0, 1, 200))
    noise = pd.Series(rng.normal(0, 1, 200))
    assert an.adf_test(trend).stationary is False
    assert an.adf_test(noise).stationary is True


def test_correlations_shape(df):
    exp = [c for c in df.columns if c.endswith("_EXP")]
    c = an.correlations(df, exp, growth=True)
    assert c.shape == (27, 27)
    assert np.allclose(np.diag(c), 1.0)


def test_top_pairs_sorted(df):
    exp = [c for c in df.columns if c.endswith("_EXP")]
    tp = an.top_pairs(an.correlations(df, exp), 5)
    assert len(tp) == 5
    assert tp.is_monotonic_decreasing


def test_coverage_full(df):
    cov = an.coverage(df)
    assert (cov["n"] == 292).all()


def test_strength_handles_zero_variance_sum():
    # resid + component is constant (zero variance) when both are all-zero -- _strength must
    # hit its explicit zero-variance guard rather than dividing by zero.
    zero = pd.Series(np.zeros(50))
    assert an._strength(zero, zero) == 0.0


def test_rolling_stats_shape_and_window(df):
    r = an.rolling_stats(df["DE_EXP"], window=12)
    assert list(r.columns) == ["mean", "std"]
    assert r["mean"].notna().sum() == len(df) - 11   # first (window-1) rows are NaN


def test_adf_result_str_reports_verdict():
    stationary = an.ADFResult(stat=-5.0, pvalue=0.001, stationary=True)
    nonstationary = an.ADFResult(stat=-1.0, pvalue=0.8, stationary=False)
    assert "stationary" in str(stationary) and "NON-stationary" not in str(stationary)
    assert "NON-stationary" in str(nonstationary)


def test_acf_pacf_returns_arrays_of_expected_length(df):
    acf_vals, pacf_vals = an.acf_pacf(df["DE_EXP"], nlags=24)
    assert len(acf_vals) == 25 and len(pacf_vals) == 25   # nlags + the lag-0 value
    assert acf_vals[0] == pytest.approx(1.0)               # a series always correlates 1.0 with itself


def test_lead_lag_recovers_a_known_shift():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0, 1, 100).cumsum(), index=pd.date_range("2010-01-01", periods=100, freq="MS"))
    b = a.shift(2)   # b lags a by 2 months -> a should be found to LEAD b by 2
    best_lag, best_corr = an.lead_lag(a.iloc[2:], b.iloc[2:], max_lag=6)
    assert best_lag == 2
    assert best_corr > 0.9
