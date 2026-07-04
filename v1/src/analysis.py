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


# Sensors that are circular (angles) and must NOT be analysed with linear stats.
CIRCULAR_SENSORS = ("wd (deg)",)


def _circular_stats_rad(rad: np.ndarray) -> tuple[float, float]:
    """Return (circular mean, circular std) in radians for an angle array, ignoring NaN."""
    s = np.nanmean(np.sin(rad))
    c = np.nanmean(np.cos(rad))
    R = float(np.hypot(s, c))  # mean resultant length in [0, 1]
    mean_ang = float(np.arctan2(s, c))
    circ_std = float(np.sqrt(-2.0 * np.log(R))) if 0.0 < R < 1.0 else 0.0
    return mean_ang, circ_std


def circular_stats_deg(series_deg: pd.Series) -> dict:
    """Circular mean and standard deviation (in degrees) for an angular series.

    Wind direction wraps at 360°, so the ordinary mean/std are invalid — the linear mean
    of a cluster straddling 0°/360° lands near 180° (the exact wrong way, and the linear
    max/min just report ~359°/~0° regardless of the prevailing wind). This computes the
    *circular* mean (the prevailing direction) and circular std, ignoring NaN. Returns
    ``{"mean": deg in [0, 360), "std": deg}``.
    """
    rad = np.deg2rad(series_deg.astype(float).to_numpy())
    mean_ang, circ_std = _circular_stats_rad(rad)
    mean_deg = float(np.rad2deg(mean_ang)) % 360.0
    if mean_deg >= 360.0:  # guard tiny-negative angles rounding up to exactly 360
        mean_deg = 0.0
    return {"mean": mean_deg, "std": float(np.rad2deg(circ_std))}


_COMPASS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


def compass_point(deg: float) -> str:
    """Nearest 8-point compass label for an angle in degrees (0°=N, 90°=E, 180°=S, 270°=W)."""
    return _COMPASS[int((float(deg) % 360.0) / 45.0 + 0.5) % 8]


def circular_zscore_anomalies(series_deg: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Flag angular outliers in a circular (degrees) series such as wind direction.

    Wind direction wraps at 360°, so a linear z-score is invalid — 359° and 1° are 2°
    apart, not 358°, and the linear mean of a wrapped cluster is meaningless. This computes
    each point's angular deviation from the *circular* mean and scales it by the *circular*
    standard deviation, flagging ``|deviation| > threshold · circ_std``. Returns a boolean
    Series (True = angular outlier), aligned to the input index.
    """
    r = np.deg2rad(series_deg.astype(float).to_numpy())
    mean_ang, circ_std = _circular_stats_rad(r)
    if circ_std <= 0:
        flags = np.zeros(len(r), dtype=bool)
    else:
        diff = np.arctan2(np.sin(r - mean_ang), np.cos(r - mean_ang))  # signed, in [-π, π]
        flags = np.abs(diff) > threshold * circ_std
    flags = flags & ~np.isnan(r)
    return pd.Series(flags, index=series_deg.index, name="circular_zscore_anomaly")


def circular_linear_corr(deg_series: pd.Series, linear_series: pd.Series) -> float:
    """Mardia circular–linear correlation between an angle (degrees) and a linear variable.

    Returns a value in ``[0, 1]`` (0 = unrelated, 1 = perfectly related), or NaN if
    undefined. Unlike Pearson's r on raw degrees, this is invariant to the 0/360° wrap and
    to the choice of zero direction, so it reports the real strength of association between
    wind direction and a linear sensor. The coefficient is unsigned (a circular variable has
    no single sign of relationship).
    """
    r = np.deg2rad(np.asarray(deg_series, dtype=float))
    x = np.asarray(linear_series, dtype=float)
    mask = ~(np.isnan(r) | np.isnan(x))
    r, x = r[mask], x[mask]
    if len(x) < 3:
        return float("nan")
    sin_r, cos_r = np.sin(r), np.cos(r)
    rxs = np.corrcoef(x, sin_r)[0, 1]
    rxc = np.corrcoef(x, cos_r)[0, 1]
    rsc = np.corrcoef(sin_r, cos_r)[0, 1]
    denom = 1.0 - rsc**2
    if denom <= 0:
        return float("nan")
    r2 = (rxc**2 + rxs**2 - 2.0 * rxc * rxs * rsc) / denom
    return float(np.sqrt(max(r2, 0.0)))


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


def data_availability(
    raw_df: pd.DataFrame, sentinels: tuple[float, ...] = (-9999.0,)
) -> pd.DataFrame:
    """Per-sensor data-availability report, computed on the RAW (pre-clean) frame.

    Returns a DataFrame indexed by sensor with columns:
      ``n_total``      — number of rows
      ``n_missing``    — NaN count (genuinely absent readings)
      ``n_sentinel``   — count of sentinel error codes (e.g. -9999 on wind sensors)
      ``n_real``       — ``n_total - n_missing - n_sentinel`` (usable readings)
      ``available_pct``— ``100 * n_real / n_total``
    This quantifies how much of each sensor is real versus imputed *before* cleaning —
    the honest denominator for any downstream statistic.
    """
    num = raw_df.select_dtypes("number")
    n_total = len(num)
    n_missing = num.isna().sum().astype(int)
    n_sentinel = num.isin(list(sentinels)).sum().astype(int)
    n_real = (n_total - n_missing - n_sentinel).astype(int)
    return pd.DataFrame(
        {
            "n_total": n_total,
            "n_missing": n_missing,
            "n_sentinel": n_sentinel,
            "n_real": n_real,
            "available_pct": 100.0 * n_real / n_total,
        }
    )


def analyze_all_sensors(
    df: pd.DataFrame, threshold: float = 3.0, circular: tuple[str, ...] = CIRCULAR_SENSORS
) -> pd.DataFrame:
    """Run the core analysis across EVERY sensor (not just temperature).

    Returns a DataFrame indexed by sensor with one row each:
      ``skew``, ``kurtosis``   — distribution shape (from ``distribution_stats``)
      ``zscore_anomalies``     — count of outliers at ``threshold`` (circular for angles)
      ``top_corr_sensor``      — the other sensor it associates most strongly with
      ``top_corr_r``           — that correlation (signed Pearson for linear sensors;
                                 unsigned circular-linear coefficient in [0, 1] for angles)
    Columns named in ``circular`` (default wind direction) are analysed with circular
    statistics — a linear z-score/Pearson on an angle that wraps at 360° is invalid. This
    is the artifact that proves humidity, pressure, wind and the rest were all verified —
    the answer to "verify the other sensors".
    """
    from . import anomaly

    stats = distribution_stats(df)
    num = df.select_dtypes("number")
    corr = correlations(num)  # compute the full matrix once, then index per sensor
    circ_set = set(circular)
    rows = {}
    for col in num.columns:
        if col in circ_set:
            zcount = int(circular_zscore_anomalies(num[col], threshold=threshold).sum())
            partners = {
                other: circular_linear_corr(num[col], num[other])
                for other in num.columns
                if other != col
            }
            partners = {k: v for k, v in partners.items() if pd.notna(v)}
            if partners:
                top_sensor = max(partners, key=partners.get)
                top_r = float(partners[top_sensor])
            else:
                top_sensor, top_r = None, float("nan")
        else:
            zcount = int(anomaly.zscore_anomalies(num[col], threshold=threshold).sum())
            c = corr[col].drop(col)
            c = c.reindex(c.abs().sort_values(ascending=False).index)
            top_sensor = c.index[0] if len(c) else None
            top_r = float(c.iloc[0]) if len(c) else float("nan")
        rows[col] = {
            "skew": float(stats.loc[col, "skew"]),
            "kurtosis": float(stats.loc[col, "kurtosis"]),
            "zscore_anomalies": zcount,
            "top_corr_sensor": top_sensor,
            "top_corr_r": top_r,
        }
    return pd.DataFrame.from_dict(rows, orient="index")
