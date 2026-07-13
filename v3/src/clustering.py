"""Day-type clustering for v3 — group the seven captured days by temperature/traffic behaviour.

Ported from v1's `daily_features`/`kmeans_cluster` pattern, adapted to v3's two channels: each day is
summarised by temperature stats (value channel) and message-rate stats (flow channel), then clustered.
With only seven days total this is a small-n exploratory grouping, not a statistically powered study —
and, tested against the real DAD week, it is an **honest negative result**: whole-day averages don't
separate attack days from clean days, because DAD's attacks last hours, not the whole day, so they
barely move a day-level mean. Kept as a tested, reproducible finding rather than hidden.
"""

from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise each calendar day into a feature vector (temperature stats + traffic rate)."""
    grouped = df.groupby("day")
    temp = grouped["temperature"].agg(["mean", "std", "min", "max"])
    temp.columns = [f"temp_{c}" for c in temp.columns]
    rate = grouped.apply(lambda g: g.resample("1min").size().mean(), include_groups=False)
    feats = temp.copy()
    feats["msg_rate_mean"] = rate
    return feats.dropna()


def kmeans_cluster(features: pd.DataFrame, k: int = 2, random_state: int = 42) -> pd.Series:
    """Cluster day feature vectors with K-means; return integer labels aligned to the day index."""
    X = StandardScaler().fit_transform(features.astype(float).to_numpy())
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)
    return pd.Series(labels, index=features.index, name="cluster")


def cluster_profiles(features: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Mean feature vector of each cluster, to interpret what it represents."""
    return features.groupby(labels).mean()
