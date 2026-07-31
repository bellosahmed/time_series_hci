"""Tests that `wd (deg)` (a circular/compass-bearing sensor) is routed through circular
statistics, not the plain linear stats that are meaningless on a wrapped angle."""

import numpy as np
import pandas as pd

from src import interface


def _wd_df(n=120, seed=0):
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    rng = np.random.default_rng(seed)
    ang = (rng.normal(0, 8, n)) % 360   # tight cluster near 0/360
    return pd.DataFrame(
        {"T (degC)": rng.normal(10, 5, n), "wd (deg)": ang}, index=idx
    )


def test_wd_mean_is_circular_not_linear():
    df = _wd_df()
    ans = interface.Interface(df).ask("average wind direction")
    assert ans.sensors == ["wd (deg)"]
    m = ans.value["mean"]
    # circular mean sits near 0/360, never the linear ~180 of a wrapped cluster
    assert min(m, 360 - m) < 30
    assert "prevailing" in ans.computed.lower() or "circular" in ans.computed.lower()


def test_wd_max_does_not_report_linear_extreme():
    df = _wd_df()
    ans = interface.Interface(df).ask("maximum wind direction")
    # a linear max on a wrapped angle (~359.x) is meaningless — must be redirected
    assert "prevailing" in ans.computed.lower() or "circular" in ans.computed.lower()


def test_wd_trend_is_refused_as_circular():
    ans = interface.Interface(_wd_df()).ask("is wind direction rising over time?")
    assert ans.intent == "trend"
    assert any(w in ans.computed.lower() for w in ("circular", "compass", "wrap"))


def test_wd_forecast_is_refused_as_circular():
    ans = interface.Interface(_wd_df()).ask("forecast the wind direction")
    assert ans.intent == "forecast"
    assert any(w in ans.computed.lower() for w in ("circular", "compass", "wrap"))


def test_wd_threshold_flags_circular_caveat():
    ans = interface.Interface(_wd_df()).ask("how often was wind direction above 200?")
    assert ans.intent == "threshold"
    assert any(w in ans.computed.lower() for w in ("circular", "compass", "wrap"))


def test_wd_correlation_is_circular_linear_unsigned():
    df = _wd_df()
    ans = interface.Interface(df).ask("relationship between wind direction and temperature")
    assert ans.intent == "correlation"
    assert 0.0 <= ans.value <= 1.0


def test_wd_summary_is_circular_not_linear():
    # "prevailing wind direction" carries no stat keyword, so it routes to `summary`. The summary
    # path must ALSO use circular statistics -- a linear min/max/mean/std on a wrapped compass angle
    # is meaningless (a tight cluster near 0/360 yields a linear mean of ~180, the exact wrong way).
    df = _wd_df()
    ans = interface.Interface(df).ask("prevailing wind direction")
    assert ans.sensors == ["wd (deg)"]
    assert ans.intent == "summary"
    # circular language, and NOT the linear "min=…, max=…, mean=…, std=…" summary format
    assert any(w in ans.computed.lower() for w in ("prevailing", "circular", "compass"))
    assert "min=" not in ans.computed.lower() and "max=" not in ans.computed.lower()
    # the circular mean sits near 0/360 for a wrapped cluster, never the linear ~180
    m = ans.value["mean"]
    assert min(m, 360 - m) < 30
