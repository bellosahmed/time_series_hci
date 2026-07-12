"""Tests for v2's temporal deep detectors (LSTM-AE, CNN-AE, TCN-AE) — the sequence-aware upgrades
of the existing point-in-time `deep_anomaly.AE`, which never sees month-to-month order.
Run: cd code/v2 && python3.12 -m pytest tests/"""
import sys
from pathlib import Path

import pytest

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2))
from src import preprocess as pp
from src import anomaly as ad
from src import deep_temporal as dt

DATA = V2.parent / "data" / "eu_trade_monthly.csv"


@pytest.fixture(scope="module")
def feats():
    logged = pp.log_transform(pp.load_trade(DATA))
    return ad.systemic_features(logged)


@pytest.mark.parametrize("arch", ["lstm_ae", "cnn_ae", "tcn_ae"])
def test_scores_length_and_finite(feats, arch):
    s = dt.temporal_ae_scores(feats, arch=arch, window=6, epochs=5, seed=0)
    assert s.dropna().shape[0] > 0
    assert s.dropna().apply(lambda x: x == x).all()          # no NaNs where scored


@pytest.mark.parametrize("arch", ["lstm_ae", "cnn_ae", "tcn_ae"])
def test_flags_fraction(feats, arch):
    flags = dt.temporal_ae_anomalies(feats, arch=arch, window=6, epochs=5, quantile=0.95, seed=0)
    assert flags.dtype == bool
    assert 0 < flags.sum() < len(flags)


@pytest.mark.parametrize("arch", ["lstm_ae", "cnn_ae", "tcn_ae"])
def test_seeded_reproducibility(feats, arch):
    s1 = dt.temporal_ae_scores(feats, arch=arch, window=6, epochs=5, seed=1)
    s2 = dt.temporal_ae_scores(feats, arch=arch, window=6, epochs=5, seed=1)
    assert (s1.dropna().round(6) == s2.dropna().round(6)).all()


def test_unknown_arch_rejected(feats):
    with pytest.raises(ValueError):
        dt.temporal_ae_scores(feats, arch="not_a_real_arch", window=6, epochs=2, seed=0)


def test_windowing_shapes():
    import numpy as np
    X = np.arange(20 * 3, dtype="float32").reshape(20, 3)          # 20 timesteps, 3 features
    Xw, starts = dt._windows_from_array(X, window=5, step=5)
    assert Xw.shape == (4, 5, 3)                                    # 4 non-overlapping windows
    assert starts == [0, 5, 10, 15]
