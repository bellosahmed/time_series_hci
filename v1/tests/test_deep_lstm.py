"""Tests for the LSTM autoencoder and LSTM forecaster (skipped without torch)."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from src import deep_lstm as dl


@pytest.fixture
def signal_with_spikes():
    idx = pd.date_range("2024-01-01", periods=24 * 40, freq="h")
    s = pd.Series(10 + 8 * np.sin(2 * np.pi * np.arange(len(idx)) / 24), index=idx)
    rng = np.random.default_rng(0)
    pos = rng.choice(len(s), size=8, replace=False)
    s.iloc[pos] += 40.0
    return s, s.index[pos]


def test_lstm_ae_scores_aligned(signal_with_spikes):
    s, _ = signal_with_spikes
    scores = dl.lstm_ae_scores(s, window=24, epochs=5)
    assert len(scores) == len(s)
    assert scores.notna().all()


def test_lstm_ae_flags_some_spikes(signal_with_spikes):
    s, spikes = signal_with_spikes
    flags = dl.lstm_ae_anomalies(s, window=24, epochs=8, quantile=0.95)
    assert flags.dtype == bool
    assert flags.loc[spikes].sum() >= 2


def test_lstm_forecast_flags_some_spikes(signal_with_spikes):
    s, spikes = signal_with_spikes
    flags = dl.lstm_forecast_anomalies(s, window=24, epochs=8, quantile=0.95)
    assert flags.dtype == bool
    # forecaster reacts at the spike (or the step right after it)
    near = spikes.union(spikes + pd.Timedelta(hours=1))
    assert flags.loc[flags.index.intersection(near)].sum() >= 2
