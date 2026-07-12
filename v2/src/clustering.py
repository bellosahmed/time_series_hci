"""
v2 clustering — group COUNTRIES by their trade behaviour (Objective 4).

v1 clustered day-types; v2 clusters countries. Each country is summarised by a few interpretable
features (size, surplus/deficit tendency, growth, seasonality, volatility); K-means then groups
similar trading nations. Deterministic + importable; no plotting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .analysis import decompose, seasonal_strength, trend_slope
from .preprocess import countries, log_transform

FEATURES = ["size_log", "balance_ratio", "growth", "seasonality", "volatility"]


def country_features(df: pd.DataFrame) -> pd.DataFrame:
    """One row per country: size, balance tendency, growth, seasonality, volatility."""
    logged = log_transform(df)
    rows = {}
    for c in countries(df):
        exp, imp = df[f"{c}_EXP"], df[f"{c}_IMP"]
        tot, bal = exp + imp, exp - imp
        le = logged[f"{c}_EXP"]
        stl = decompose(le, period=12)
        rows[c] = {
            "size_log": float(np.log(tot.mean())),          # economic size (log avg total trade)
            "balance_ratio": float(bal.mean() / tot.mean()),# +surplus / -deficit, scale-free
            "growth": float(trend_slope(le) * 12),          # ~ annual log growth of exports
            "seasonality": float(seasonal_strength(stl)),   # how seasonal the exports are
            "volatility": float(le.diff().std()),           # month-to-month log volatility
        }
    return pd.DataFrame(rows).T[FEATURES]


def cluster_countries(features: pd.DataFrame, k: int = 4, random_state: int = 0) -> pd.Series:
    """K-means over standardised features. Returns a cluster label per country."""
    X = StandardScaler().fit_transform(features)
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(X)
    return pd.Series(km.labels_, index=features.index, name="cluster")


def best_k(features: pd.DataFrame, ks=None, random_state: int = 0) -> pd.Series:
    """Silhouette score for each k (higher = better-separated clusters)."""
    ks = ks if ks is not None else range(2, 7)
    X = StandardScaler().fit_transform(features)
    out = {}
    for k in ks:
        lab = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(X)
        out[k] = float(silhouette_score(X, lab))
    return pd.Series(out, name="silhouette")


def cluster_profiles(features: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Mean feature values per cluster (to interpret what each group represents)."""
    return features.assign(cluster=labels).groupby("cluster").mean()


def cluster_members(labels: pd.Series) -> dict[int, list[str]]:
    """Country codes in each cluster."""
    return {int(k): sorted(v.index.tolist()) for k, v in labels.groupby(labels)}
