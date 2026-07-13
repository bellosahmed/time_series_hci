"""Tests for SARIMA forecasting (hourly_temperature / forecast / backtest)."""

import numpy as np
import pandas as pd

from src import forecasting


def _hourly_series(n=168):
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    rng = np.random.default_rng(0)
    hour_of_day = np.arange(n) % 24
    daily_cycle = 2.0 * np.sin(2 * np.pi * hour_of_day / 24)
    return pd.Series(20.0 + daily_cycle + rng.normal(0, 0.3, n), index=idx)


def test_hourly_temperature_resamples_and_fills():
    idx = pd.date_range("2024-01-01", periods=10, freq="10min")
    df = pd.DataFrame({"temperature": [20.0, np.nan, 21.0, 20.5, np.nan,
                                       19.5, 20.0, np.nan, 21.5, 20.0]}, index=idx)
    s = forecasting.hourly_temperature(df)
    assert not s.isna().any()


def test_forecast_returns_steps_rows_with_ordered_interval():
    series = _hourly_series()
    fc = forecasting.forecast(series, steps=12)
    assert len(fc) == 12
    assert (fc["lower"] <= fc["mean"]).all() and (fc["mean"] <= fc["upper"]).all()


def test_backtest_reports_all_metrics_and_uses_requested_folds():
    series = _hourly_series()
    result = forecasting.backtest(series, horizon=12, folds=2)
    assert result["folds"] == 2
    assert {"mae", "rmse", "baseline_mae", "mase"} <= set(result.keys())
    assert result["mae"] >= 0 and result["rmse"] >= 0
