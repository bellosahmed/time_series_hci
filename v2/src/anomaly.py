"""
v2 anomaly detection for the EU monthly trade dataset.

Three detector families (chosen for a short, seasonal, multivariate monthly series):
  1. STATISTICAL  - z-score on the STL **residual** (per series).
  2. FORECASTING  - z-score on the **seasonal-naive error** (value vs same month last year).
  3. MACHINE-LEARNING (multivariate) - Isolation Forest over all countries' year-on-year
     growth, so it flags **systemic** months where many countries move abnormally at once.

Plus light evaluation helpers (label windows + precision/recall/F1/confusion) so detectors can
be scored against the real 2008 / COVID events. Importable + deterministic; no plotting.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .analysis import decompose
from .preprocess import flow_columns


# --------------------------------------------------------------------------- 1. statistical
def stl_residual_z(series: pd.Series, period: int = 12, robust: bool = True) -> pd.Series:
    """Standardised STL residual (how many std-devs from the de-seasonalised, de-trended norm)."""
    r = decompose(series, period=period, robust=robust).resid
    return (r - r.mean()) / r.std()


def zscore_anomalies(series: pd.Series, threshold: float = 3.0, period: int = 12) -> pd.Series:
    """Boolean mask: residual |z| exceeds threshold."""
    return stl_residual_z(series, period=period).abs() > threshold


# --------------------------------------------------------------------------- 2. forecasting
def seasonal_naive_z(series: pd.Series, period: int = 12) -> pd.Series:
    """Standardised seasonal-naive forecast error: actual - value `period` months ago."""
    err = series - series.shift(period)
    return (err - err.mean()) / err.std()


def seasonal_naive_anomalies(series: pd.Series, threshold: float = 3.0, period: int = 12) -> pd.Series:
    """Boolean mask from the seasonal-naive forecast error."""
    return (seasonal_naive_z(series, period=period).abs() > threshold).fillna(False)


# --------------------------------------------------------------------------- 3. ML multivariate
def systemic_features(logged: pd.DataFrame, period: int = 12) -> pd.DataFrame:
    """Year-on-year growth (log diff) of every flow column — the multivariate fingerprint."""
    return logged[flow_columns(logged)].diff(period).dropna()


def isoforest_anomalies(features: pd.DataFrame, contamination: float = 0.05,
                        random_state: int = 0) -> pd.Series:
    """Isolation Forest on a feature matrix (rows=months). True = anomalous month."""
    X = features.dropna()
    model = IsolationForest(contamination=contamination, random_state=random_state)
    pred = model.fit_predict(X)                    # -1 = anomaly, 1 = normal
    return pd.Series(pred == -1, index=X.index).reindex(features.index, fill_value=False)


def isoforest_scores(features: pd.DataFrame, random_state: int = 0) -> pd.Series:
    """Continuous anomaly score (higher = more anomalous) from Isolation Forest."""
    X = features.dropna()
    model = IsolationForest(random_state=random_state).fit(X)
    s = -model.score_samples(X)                    # flip so larger = more anomalous
    return pd.Series(s, index=X.index).reindex(features.index)


# --------------------------------------------------------------------------- evaluation
def label_windows(index: pd.DatetimeIndex, windows: list[tuple[str, str]]) -> pd.Series:
    """Boolean ground-truth labels: True inside any (start, end) window (inclusive)."""
    lab = pd.Series(False, index=index)
    for start, end in windows:
        lab.loc[start:end] = True
    return lab


@dataclass
class Scores:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    tn: int

    def __str__(self) -> str:
        return (f"P={self.precision:.2f} R={self.recall:.2f} F1={self.f1:.2f} "
                f"(TP={self.tp} FP={self.fp} FN={self.fn} TN={self.tn})")


def score(pred: pd.Series, truth: pd.Series) -> Scores:
    """Precision / recall / F1 / confusion of a boolean prediction vs boolean truth."""
    idx = pred.index.intersection(truth.index)
    p, t = pred.loc[idx].fillna(False).astype(bool), truth.loc[idx].astype(bool)
    tp = int((p & t).sum()); fp = int((p & ~t).sum())
    fn = int((~p & t).sum()); tn = int((~p & ~t).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return Scores(precision, recall, f1, tp, fp, fn, tn)
