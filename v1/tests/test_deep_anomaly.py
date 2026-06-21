"""Test the 1D CNN Autoencoder on a controlled signal.

Skipped automatically if PyTorch isn't installed.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from src import deep_anomaly as da


def test_windowing_shapes():
    s = pd.Series(np.arange(100.0), index=pd.date_range("2024-01-01", periods=100, freq="h"))
    X, spans = da.make_windows(s, window=24)
    assert X.shape[1:] == (1, 24)
    assert len(spans) == X.shape[0]


def test_autoencoder_flags_injected_spikes():
    # clean daily sine + a few big spikes the AE shouldn't be able to reconstruct
    idx = pd.date_range("2024-01-01", periods=24 * 40, freq="h")
    s = pd.Series(10 + 8 * np.sin(2 * np.pi * np.arange(len(idx)) / 24), index=idx)
    rng = np.random.default_rng(0)
    spike_pos = rng.choice(len(s), size=8, replace=False)
    s.iloc[spike_pos] += 40.0

    flags = da.cnn_ae_anomalies(s, window=24, epochs=15, quantile=0.95)
    # at least some injected spikes should land in flagged windows
    spike_idx = s.index[spike_pos]
    assert flags.loc[spike_idx].sum() >= 4
    assert flags.dtype == bool
