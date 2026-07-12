"""Tests for v3's deep detectors — an LSTM autoencoder for the flow channel (never tried before in
v3) and a per-sensor LSTM forecaster for the value channel (v1's `LSTMForecaster`, applied per sensor
instead of a single target). Run: cd code/v3 && python3.12 -m pytest tests/"""
import numpy as np
import pandas as pd
import pytest

from src import deep


def _flow_df(n=60, seed=0):
    idx = pd.date_range("2019-10-21", periods=n, freq="5min")
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {"msg_rate": rng.normal(50, 5, n), "iat_mean": rng.normal(6, 1, n),
         "iat_std": rng.normal(2, 0.5, n), "dup_msgid_ratio": rng.uniform(0, 0.1, n)},
        index=idx,
    )


def _sensor_df(n=60, seed=0):
    idx = pd.date_range("2019-10-21", periods=n, freq="5min")
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {131: 20 + rng.normal(0, 0.5, n), 132: 22 + rng.normal(0, 0.5, n)}, index=idx
    )


def test_flow_lstm_ae_scores_length_and_finite():
    flow = _flow_df()
    s = deep.flow_lstm_ae_score(flow, window=6, epochs=5, seed=0)
    assert s.dropna().shape[0] > 0
    assert s.dropna().apply(lambda x: x == x).all()


def test_flow_lstm_ae_seeded_reproducibility():
    flow = _flow_df()
    s1 = deep.flow_lstm_ae_score(flow, window=6, epochs=5, seed=1)
    s2 = deep.flow_lstm_ae_score(flow, window=6, epochs=5, seed=1)
    assert (s1.dropna().round(6) == s2.dropna().round(6)).all()


def test_flow_lstm_ae_flags_a_clear_outlier_window():
    flow = _flow_df(n=80, seed=0)
    flow.iloc[40:44] = flow.iloc[40:44] * 5           # a clear burst
    s = deep.flow_lstm_ae_score(flow, window=6, epochs=30, seed=0)
    assert s.iloc[38:46].max() > s.drop(s.index[36:48]).median()   # burst region scores higher


def test_value_lstm_forecast_score_length_and_finite():
    sensors = _sensor_df()
    s = deep.value_lstm_forecast_score(sensors, window=12, epochs=5, seed=0)
    assert s.dropna().shape[0] > 0
    assert s.dropna().apply(lambda x: x == x).all()


def test_value_lstm_forecast_flags_a_modified_sensor():
    sensors = _sensor_df(n=100, seed=0)
    sensors.iloc[70, 0] = sensors.iloc[70, 0] + 15    # a clear out-of-pattern reading on sensor 131
    s = deep.value_lstm_forecast_score(sensors, window=12, epochs=20, seed=0)
    assert s.iloc[70] > s.drop(s.index[65:75]).median()
