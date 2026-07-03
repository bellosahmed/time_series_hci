"""Anomaly detection: flag abnormal behaviour, comparing methods.

Responsibilities (Phase 3):
- baseline: z-score on residuals / rolling-window thresholds
- classical ML: Isolation Forest on engineered features
- optional (later): 1D CNN, only if baseline + ML are insufficient

All detectors return a boolean Series (True = anomaly), aligned to the input index,
so they can be compared directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def zscore_anomalies(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Flag points whose global z-score exceeds `threshold` in absolute value."""
    s = series.astype(float)
    z = (s - s.mean()) / s.std(ddof=0)
    return (z.abs() > threshold).rename("zscore_anomaly")


def rolling_zscore_anomalies(
    series: pd.Series, window: int = 24, threshold: float = 3.0
) -> pd.Series:
    """Flag points far from their *local* rolling mean (robust to slow trends)."""
    s = series.astype(float)
    mean = s.rolling(window, min_periods=window // 2).mean()
    std = s.rolling(window, min_periods=window // 2).std(ddof=0)
    z = (s - mean) / std.replace(0, np.nan)
    return (z.abs() > threshold).fillna(False).rename("rolling_zscore_anomaly")


def isolation_forest_anomalies(
    features: pd.DataFrame, contamination: float = 0.01, random_state: int = 42
) -> pd.Series:
    """Fit Isolation Forest on (standardised) features; return a boolean Series.

    `features` may be one or many columns (multi-sensor). Rows scoring as outliers
    are marked True.
    """
    X = features.astype(float).to_numpy()
    X = StandardScaler().fit_transform(X)
    model = IsolationForest(contamination=contamination, random_state=random_state)
    labels = model.fit_predict(X)  # -1 = outlier, 1 = inlier
    return pd.Series(labels == -1, index=features.index, name="iforest_anomaly")


def lof_anomalies(
    features: pd.DataFrame, n_neighbors: int = 20, contamination: float = 0.01
) -> pd.Series:
    """Local Outlier Factor: flag points in low-density regions relative to their neighbours.

    LOF is a density-based detector — a point is anomalous if it sits in a sparser region
    than its `n_neighbors` nearest neighbours do. Complements Isolation Forest (which splits
    globally) by catching *local* outliers. Returns a boolean Series (True = anomaly).
    """
    from sklearn.neighbors import LocalOutlierFactor

    X = StandardScaler().fit_transform(features.astype(float).to_numpy())
    labels = LocalOutlierFactor(
        n_neighbors=n_neighbors, contamination=contamination
    ).fit_predict(X)  # -1 = outlier
    return pd.Series(labels == -1, index=features.index, name="lof_anomaly")


def ocsvm_anomalies(
    features: pd.DataFrame, nu: float = 0.01, gamma: str = "scale",
    max_fit: int = 15000, random_state: int = 42,
) -> pd.Series:
    """One-Class SVM: learn a boundary around the bulk of the data; points outside are anomalies.

    `nu` upper-bounds the fraction of outliers. A kernel One-Class SVM fits in ~O(n²), so on
    long series we fit on a random subsample of at most `max_fit` rows and then score *every*
    point — keeping it tractable on the full ~70k-row series while still labelling all of it.
    Returns a boolean Series (True = anomaly).
    """
    from sklearn.svm import OneClassSVM

    X = StandardScaler().fit_transform(features.astype(float).to_numpy())
    model = OneClassSVM(nu=nu, gamma=gamma)
    if len(X) > max_fit:
        rng = np.random.default_rng(random_state)
        model.fit(X[rng.choice(len(X), max_fit, replace=False)])
    else:
        model.fit(X)
    return pd.Series(model.predict(X) == -1, index=features.index, name="ocsvm_anomaly")


def threshold_sensitivity(series: pd.Series,
                          thresholds=(2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)) -> dict:
    """Anomaly count at each z-score threshold — evidence for the chosen value.

    Pass the (STL-residual) series. Lower thresholds flag more points (more false
    positives); higher thresholds flag fewer (risk missing real events). The default
    4.0 is the knee where only clear events survive on this data.
    """
    return {float(t): int(zscore_anomalies(series, threshold=t).sum()) for t in thresholds}


def contamination_sensitivity(features: pd.DataFrame,
                              contaminations=(0.005, 0.01, 0.02, 0.05)) -> dict:
    """Isolation-Forest anomaly count at each `contamination` prior (sweep to justify it)."""
    return {float(c): int(isolation_forest_anomalies(features, contamination=c).sum())
            for c in contaminations}


def anomaly_windows(flags: pd.Series) -> list[tuple]:
    """Collapse a boolean anomaly Series into (start, end) timestamp intervals.

    Thin wrapper over ``query_engine.get_anomaly_windows`` (the canonical grouping)
    so the interval logic lives in one place.
    """
    from .query_engine import get_anomaly_windows
    return [(w["start"], w["end"]) for w in get_anomaly_windows(flags)]
