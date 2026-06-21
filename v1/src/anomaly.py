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


def anomaly_windows(flags: pd.Series) -> list[tuple]:
    """Collapse a boolean anomaly Series into (start, end) timestamp intervals."""
    flags = flags.astype(bool)
    windows: list[tuple] = []
    start = None
    prev = None
    for ts, is_anom in flags.items():
        if is_anom and start is None:
            start = ts
        elif not is_anom and start is not None:
            windows.append((start, prev))
            start = None
        prev = ts
    if start is not None:
        windows.append((start, prev))
    return windows
