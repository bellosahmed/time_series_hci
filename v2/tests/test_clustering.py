"""Tests for v2 country clustering. Run: cd code/v2 && python3.12 -m pytest tests/"""
import sys
from pathlib import Path

import pytest

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2))
from src import preprocess as pp
from src import clustering as cl

DATA = V2.parent / "data" / "eu_trade_monthly.csv"


@pytest.fixture(scope="module")
def feats():
    return cl.country_features(pp.load_trade(DATA))


def test_feature_shape(feats):
    assert feats.shape == (27, 5)
    assert list(feats.columns) == cl.FEATURES


def test_germany_is_largest(feats):
    assert feats["size_log"].idxmax() == "DE"            # Germany = biggest trader


def test_netherlands_surplus_positive(feats):
    assert feats.loc["NL", "balance_ratio"] > 0          # NL runs a surplus


def test_clusters_partition(feats):
    lab = cl.cluster_countries(feats, k=4)
    assert len(lab) == 27
    assert lab.nunique() == 4
    members = cl.cluster_members(lab)
    assert sum(len(v) for v in members.values()) == 27   # every country assigned once


def test_best_k_returns_scores(feats):
    sk = cl.best_k(feats, ks=range(2, 6))
    assert set(sk.index) == {2, 3, 4, 5}
    assert (sk.between(-1, 1)).all()


def test_profiles_shape(feats):
    lab = cl.cluster_countries(feats, k=3)
    prof = cl.cluster_profiles(feats, lab)
    assert prof.shape == (3, 5)
