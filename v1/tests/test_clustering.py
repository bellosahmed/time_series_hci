"""Tests for the K-means day-type clustering (`src/clustering.py`)."""

import numpy as np
import pandas as pd

from src import clustering


def _df():
    # 40 days hourly: first 20 cold (~0 C), next 20 warm (~20 C) -> two clear day-types.
    idx = pd.date_range("2020-01-01", periods=24 * 40, freq="h")
    base = np.where(np.arange(len(idx)) < 24 * 20, 0.0, 20.0)
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {"T (degC)": base + rng.normal(0, 1, len(idx)),
         "rh (%)": 60 + rng.normal(0, 5, len(idx))},
        index=idx,
    )


def test_daily_features_one_row_per_day():
    f = clustering.daily_features(_df())
    assert len(f) == 40 and "mean_T (degC)" in f.columns and "max_rh (%)" in f.columns


def test_kmeans_profiles_and_labels():
    f = clustering.daily_features(_df())
    labels = clustering.kmeans_cluster(f, k=2)
    assert set(labels.unique()) <= {0, 1} and len(labels) == 40
    profiles = clustering.cluster_profiles(f, labels)
    names = clustering.day_type_labels(profiles)
    assert set(names.values()) == {"cold", "warm"}
    # the coldest cluster gets the 'cold' label (labels are temperature-ordered)
    coldest = min(names, key=lambda c: profiles.loc[c, "mean_T (degC)"])
    assert names[coldest] == "cold"


def test_cluster_membership_separates_cold_and_warm_days():
    f = clustering.daily_features(_df())
    labels = clustering.kmeans_cluster(f, k=2)
    cold_day = clustering.cluster_membership(labels, pd.Timestamp("2020-01-05"))  # cold half
    warm_day = clustering.cluster_membership(labels, pd.Timestamp("2020-01-30"))  # warm half
    assert cold_day != warm_day


def test_kmeans_selection_scores_each_k():
    f = clustering.daily_features(_df())          # data has a clear 2-cluster (cold/warm) structure
    sel = clustering.kmeans_selection(f, k_range=range(2, 6))
    assert list(sel.index) == [2, 3, 4, 5]
    assert sel["silhouette"].between(-1, 1).all()
    assert sel["inertia"].is_monotonic_decreasing          # inertia always falls as k rises
    assert sel["silhouette"].idxmax() == 2                  # best separation at the true k=2
