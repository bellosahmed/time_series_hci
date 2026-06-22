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
