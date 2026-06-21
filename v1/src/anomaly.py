"""Anomaly detection: flag abnormal behaviour, comparing methods.

Responsibilities (Phase 3):
- baseline: z-score on residuals / rolling-window thresholds
- classical ML: Isolation Forest on engineered features
- optional (later): 1D CNN, only if baseline + ML are insufficient
Returns anomaly intervals and/or scores.
"""

from __future__ import annotations

import pandas as pd


def zscore_anomalies(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Flag points whose |z-score| exceeds threshold. TODO."""
    raise NotImplementedError


def isolation_forest_anomalies(features: pd.DataFrame, contamination: float = 0.01) -> pd.Series:
    """Fit Isolation Forest and return anomaly labels/scores. TODO."""
    raise NotImplementedError
