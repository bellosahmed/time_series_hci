"""Tests for the deep detectors — CNN autoencoder, LSTM autoencoder, LSTM forecaster.

All are skipped automatically if PyTorch isn't installed. Each uses a clean daily sine wave
with a handful of injected spikes, and checks the detector flags at least some of them.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from src import deep


@pytest.fixture
def signal_with_spikes():
    """A daily sine wave (40 days, hourly) with 8 large spikes injected at known positions."""
    idx = pd.date_range("2024-01-01", periods=24 * 40, freq="h")
    s = pd.Series(10 + 8 * np.sin(2 * np.pi * np.arange(len(idx)) / 24), index=idx)
    rng = np.random.default_rng(0)
    pos = rng.choice(len(s), size=8, replace=False)
    s.iloc[pos] += 40.0
    return s, s.index[pos]


# --- CNN autoencoder ------------------------------------------------------- #
def test_windowing_shapes():
    s = pd.Series(np.arange(100.0), index=pd.date_range("2024-01-01", periods=100, freq="h"))
    X, spans = deep.make_windows(s, window=24)
    assert X.shape[1:] == (1, 24)
    assert len(spans) == X.shape[0]


def test_cnn_ae_flags_injected_spikes(signal_with_spikes):
    s, spikes = signal_with_spikes
    flags = deep.cnn_ae_anomalies(s, window=24, epochs=15, quantile=0.95)
    # at least some injected spikes should land in flagged windows
    assert flags.loc[spikes].sum() >= 4
    assert flags.dtype == bool


# --- LSTM autoencoder ------------------------------------------------------ #
def test_lstm_ae_scores_aligned(signal_with_spikes):
    s, _ = signal_with_spikes
    scores = deep.lstm_ae_scores(s, window=24, epochs=5)
    assert len(scores) == len(s)
    assert scores.notna().all()


def test_lstm_ae_flags_some_spikes(signal_with_spikes):
    s, spikes = signal_with_spikes
    flags = deep.lstm_ae_anomalies(s, window=24, epochs=8, quantile=0.95)
    assert flags.dtype == bool
    assert flags.loc[spikes].sum() >= 2


# --- LSTM forecaster ------------------------------------------------------- #
def test_lstm_forecast_flags_some_spikes(signal_with_spikes):
    s, spikes = signal_with_spikes
    flags = deep.lstm_forecast_anomalies(s, window=24, epochs=8, quantile=0.95)
    assert flags.dtype == bool
    # forecaster reacts at the spike (or the step right after it)
    near = spikes.union(spikes + pd.Timedelta(hours=1))
    assert flags.loc[flags.index.intersection(near)].sum() >= 2
