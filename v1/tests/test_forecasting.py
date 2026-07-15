"""Tests for the ARIMA forecasting-based detector."""

import numpy as np
import pandas as pd
import pytest

from src import forecasting


def _signal_with_spikes():
    idx = pd.date_range("2024-01-01", periods=24 * 30, freq="h")
    s = pd.Series(10 + 5 * np.sin(2 * np.pi * np.arange(len(idx)) / 24), index=idx)
    rng = np.random.default_rng(1)
    pos = rng.choice(len(s), size=6, replace=False)
    s.iloc[pos] += 30.0
    return s, s.index[pos]


def test_arima_residuals_aligned():
    s, _ = _signal_with_spikes()
    resid = forecasting.arima_residuals(s, order=(1, 0, 1), deseasonalize_period=24)
    assert len(resid) == len(s)


def test_arima_flags_spikes():
    s, spikes = _signal_with_spikes()
    flags = forecasting.arima_anomalies(s, order=(1, 0, 1), threshold=3.0, deseasonalize_period=24)
    assert flags.dtype == bool
    assert flags.loc[spikes].sum() >= 3


def test_forecast_shape_and_bounds():
    idx = pd.date_range("2014-01-01", periods=24 * 400, freq="h")
    s = pd.Series(10 + 5 * np.sin(2 * np.pi * np.arange(len(idx)) / (24 * 365)), index=idx)
    fc = forecasting.forecast(s, steps=14)
    assert len(fc) == 14
    assert (fc["lower"] <= fc["mean"]).all() and (fc["mean"] <= fc["upper"]).all()
    assert fc.index[0] > s.index[-1]            # forecasts the future


def test_backtest_metrics():
    idx = pd.date_range("2014-01-01", periods=24 * 500, freq="h")
    s = pd.Series(10 + 5 * np.sin(2 * np.pi * np.arange(len(idx)) / (24 * 365)), index=idx)
    bt = forecasting.backtest(s, horizon=7, folds=3)
    assert bt["folds"] >= 1
    assert bt["mae"] >= 0 and bt["rmse"] >= 0 and "mase" in bt


def test_backtest_reports_per_fold_mase_and_a_confidence_interval():
    # A bare point-estimate MASE (the pre-existing behaviour) can't be compared for statistical
    # significance -- backtest() must also expose enough folds' worth of individual MASE values
    # to compute a mean/std/CI, the same rigor evaluation.benchmark() already gives anomaly
    # detection.
    idx = pd.date_range("2014-01-01", periods=24 * 900, freq="h")
    s = pd.Series(10 + 5 * np.sin(2 * np.pi * np.arange(len(idx)) / (24 * 365)), index=idx)
    bt = forecasting.backtest(s, horizon=7, folds=5)
    assert bt["folds"] >= 2, "need >=2 folds for this test to be meaningful"
    assert len(bt["per_fold_mase"]) == bt["folds"]
    assert all(m >= 0 for m in bt["per_fold_mase"])
    assert bt["mase_mean"] == pytest.approx(np.mean(bt["per_fold_mase"]))
    assert bt["mase_std"] == pytest.approx(np.std(bt["per_fold_mase"]))
    assert bt["mase_ci_low"] <= bt["mase_mean"] <= bt["mase_ci_high"]
