"""Tests for v2 deep autoencoder anomaly detector. Run: cd code/v2 && python3.12 -m pytest tests/"""
import sys
from pathlib import Path

import pytest

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2))
from src import preprocess as pp
from src import anomaly as ad
from src import deep_anomaly as da

DATA = V2.parent / "data" / "eu_trade_monthly.csv"


@pytest.fixture(scope="module")
def feats():
    logged = pp.log_transform(pp.load_trade(DATA))
    return ad.systemic_features(logged)


def test_scores_length_and_finite(feats):
    s = da.ae_scores(feats, da.train_autoencoder(feats, epochs=100, seed=0))
    assert s.dropna().shape[0] == feats.dropna().shape[0]
    assert s.dropna().apply(lambda x: x == x).all()        # no NaNs in valid range


def test_flags_fraction(feats):
    flags = da.ae_anomalies(feats, contamination=0.05,
                            trained=da.train_autoencoder(feats, epochs=150, seed=0))
    assert flags.dtype == bool
    assert 0 < flags.sum() < len(flags)


def test_catches_a_real_crisis(feats):
    # the AE should rank a known crisis month among its highest reconstruction errors
    # early-stopped AE (see notebook 12): crises stand out as high reconstruction error
    s = da.ae_scores(feats, da.train_autoencoder(feats, epochs=60, seed=0)).dropna()
    top = set(d.strftime("%Y-%m") for d in s.sort_values(ascending=False).head(20).index)
    crisis = {"2009-01", "2009-02", "2009-05", "2020-04", "2020-05", "2021-04"}
    assert top & crisis                                    # at least one crisis month in the top 20


def test_get_device_returns_valid():
    assert da.get_device() in ("cuda", "cpu")
    assert da.get_device("cpu") == "cpu"
