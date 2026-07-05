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


def kmeans_selection(features: pd.DataFrame, k_range=None,
                     random_state: int = 42) -> pd.DataFrame:
    """Score K-means across a range of `k` to justify the chosen number of clusters.

    Returns a DataFrame indexed by `k` with two columns: **inertia** (within-cluster sum of squares —
    the "elbow"; always falls as k rises, look for the bend) and **silhouette** (−1..1 — cluster
    separation; higher is better). Used to justify k rather than picking it arbitrarily.
    """
    from sklearn.metrics import silhouette_score

    if k_range is None:
        k_range = range(2, 9)
    X = StandardScaler().fit_transform(features.astype(float).to_numpy())
    rows = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(X)
        rows[int(k)] = {"inertia": float(km.inertia_),
                        "silhouette": float(silhouette_score(X, km.labels_))}
    return pd.DataFrame(rows).T.rename_axis("k")


def cluster_membership(labels: pd.Series, when) -> int:
    """Return the cluster label for a given timestamp/index value."""
    return int(labels.loc[when])


def day_type_labels(profiles: pd.DataFrame, temp_col: str = "mean_T (degC)") -> dict[int, str]:
    """Map each cluster id to a human day-type label, ordered coldest→warmest by mean temperature.

    Makes clusters interpretable to a user ("a cold day", "a warm day") instead of an opaque
    integer. Falls back to the first feature column if the temperature mean is absent.
    """
    col = temp_col if temp_col in profiles.columns else profiles.columns[0]
    ordered = profiles[col].sort_values().index.tolist()  # cluster ids coldest -> warmest
    vocab = {
        1: ["typical"], 2: ["cold", "warm"], 3: ["cold", "mild", "warm"],
        4: ["cold", "cool", "mild", "warm"], 5: ["cold", "cool", "mild", "warm", "hot"],
    }
    names = vocab.get(len(ordered), [f"type-{i}" for i in range(len(ordered))])
    return {int(cid): names[rank] for rank, cid in enumerate(ordered)}
