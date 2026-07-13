"""Tests for day-type clustering (daily_features / kmeans_cluster / cluster_profiles)."""

import numpy as np
import pandas as pd

from src import clustering


def _week_df():
    """Seven days with two visibly different temperature/traffic regimes (hot+busy vs cool+quiet)."""
    rng = np.random.default_rng(0)
    frames = []
    hot_busy = {"mon", "tue", "wed"}
    for day in ("mon", "tue", "wed", "thu", "fri", "sat", "sun"):
        idx = pd.date_range("2024-01-01", periods=200, freq="1min") + pd.Timedelta(days=len(frames))
        base = 30.0 if day in hot_busy else 10.0
        temp = rng.normal(base, 1, 200)
        frames.append(pd.DataFrame({"temperature": temp, "day": day}, index=idx))
    return pd.concat(frames)


def test_daily_features_one_row_per_day():
    df = _week_df()
    feats = clustering.daily_features(df)
    assert len(feats) == 7
    assert {"temp_mean", "temp_std", "temp_min", "temp_max", "msg_rate_mean"} <= set(feats.columns)


def test_kmeans_cluster_separates_distinct_regimes():
    df = _week_df()
    feats = clustering.daily_features(df)
    labels = clustering.kmeans_cluster(feats, k=2)
    hot = set(labels[labels.index.isin(["mon", "tue", "wed"])])
    cool = set(labels[labels.index.isin(["thu", "fri", "sat", "sun"])])
    assert hot != cool          # the two visibly different regimes land in different clusters


def test_cluster_profiles_shape():
    df = _week_df()
    feats = clustering.daily_features(df)
    labels = clustering.kmeans_cluster(feats, k=2)
    profiles = clustering.cluster_profiles(feats, labels)
    assert profiles.shape[0] == labels.nunique()
    assert set(profiles.columns) == set(feats.columns)
