"""Tests for the ARIMA forecasting-based detector."""

import numpy as np
import pandas as pd

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
