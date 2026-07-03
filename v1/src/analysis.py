"""Behaviour analysis: describe how each sensor behaves over time.

Responsibilities (Phase 2):
- rolling mean / rolling std
- trend slope and STL decomposition (trend / seasonal / residual)
- correlation matrix across sensors
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


def rolling_stats(series: pd.Series, window: int = 24) -> pd.DataFrame:
    """Return rolling mean and std for a series (window in samples)."""
    return pd.DataFrame(
        {
            "mean": series.rolling(window, min_periods=1).mean(),
            "std": series.rolling(window, min_periods=1).std(),
        }
    )


def trend_slope(series: pd.Series) -> float:
    """Least-squares slope of the series vs time, in units per sample.

    Positive => the signal is rising over the period; negative => falling.
    """
    y = series.to_numpy(dtype=float)
    x = np.arange(len(y), dtype=float)
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return float("nan")
    slope = np.polyfit(x[mask], y[mask], 1)[0]
    return float(slope)


def decompose(series: pd.Series, period: int, robust: bool = True) -> pd.DataFrame:
    """STL-decompose a series into trend / seasonal / residual components.

    `period` is the number of samples in one seasonal cycle (e.g. 24 for a daily
    cycle on hourly data). Returns a DataFrame with columns
    [observed, trend, seasonal, resid].
    """
    series = series.astype(float).interpolate(limit_direction="both")
    result = STL(series, period=period, robust=robust).fit()
    return pd.DataFrame(
        {
            "observed": series,
            "trend": result.trend,
            "seasonal": result.seasonal,
            "resid": result.resid,
        }
    )


def correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Return the Pearson correlation matrix across all numeric sensors."""
    return df.corr(numeric_only=True)


def top_correlations(df: pd.DataFrame, target: str, n: int = 5) -> pd.Series:
    """Return the `n` sensors most correlated (by |r|) with `target`."""
    corr = correlations(df)[target].drop(target)
    return corr.reindex(corr.abs().sort_values(ascending=False).index).head(n)


def distribution_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-sensor distribution shape: mean, median, std, skew, kurtosis, min, max.

    Returns a DataFrame indexed by sensor name (one row per numeric column). ``skew``
    is Fisher-Pearson skewness — 0 means symmetric, >0 a right/long-upper tail, <0 a
    left tail. ``kurtosis`` is *excess* kurtosis — 0 is normal-tailed, >0 heavy-tailed
    (more outliers). Together they describe the *shape* of each sensor's distribution,
    the basis for judging skewness and heavy tails.
    """
    num = df.select_dtypes("number")
    return pd.DataFrame(
        {
            "mean": num.mean(),
            "median": num.median(),
            "std": num.std(),
            "skew": num.skew(),
            "kurtosis": num.kurt(),
            "min": num.min(),
            "max": num.max(),
        }
    )
