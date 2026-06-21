"""Clustering: group similar behaviour (normal vs abnormal, or similar periods).

Responsibilities (Phase 4):
- engineer window/feature vectors (e.g. daily summaries)
- K-means and/or hierarchical clustering
- interpret clusters; expose a membership lookup
"""

from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise each calendar day into a feature vector (mean/std/min/max per sensor).

    Turning a long series into one row per day gives interpretable units for
    clustering ("what kind of day was this?").
    """
    grouped = df.resample("1D")
    feats = pd.concat(
        {
            "mean": grouped.mean(),
            "std": grouped.std(),
            "min": grouped.min(),
            "max": grouped.max(),
        },
        axis=1,
    )
    feats.columns = [f"{stat}_{col}" for stat, col in feats.columns]
    return feats.dropna()


def kmeans_cluster(
    features: pd.DataFrame, k: int = 4, random_state: int = 42
) -> pd.Series:
    """Cluster feature vectors with K-means; return integer labels aligned to index."""
    X = StandardScaler().fit_transform(features.astype(float).to_numpy())
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)
    return pd.Series(labels, index=features.index, name="cluster")


def cluster_profiles(features: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Return the mean feature vector of each cluster (to interpret what it represents)."""
    return features.groupby(labels).mean()


def cluster_membership(labels: pd.Series, when) -> int:
    """Return the cluster label for a given timestamp/index value."""
    return int(labels.loc[when])
