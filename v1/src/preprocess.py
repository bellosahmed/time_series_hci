"""Preprocessing: load, clean, align, and interpolate time-series data.

Responsibilities (Phase 1):
- parse and sort timestamps; set a datetime index
- handle missing values (interpolate / forward-fill) and report how many were missing
- resample / align multiple sensors to a common frequency

Written generically so it works on any time-series CSV, with sensible defaults
for the Jena Climate dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Jena Climate defaults
JENA_TIME_COL = "Date Time"
JENA_TIME_FMT = "%d.%m.%Y %H:%M:%S"


def load_data(
    path: str,
    time_col: str = JENA_TIME_COL,
    time_format: str | None = JENA_TIME_FMT,
) -> pd.DataFrame:
    """Load a time-series CSV and return a DataFrame indexed by a sorted datetime.

    Parameters
    ----------
    path : path to the CSV file.
    time_col : name of the timestamp column.
    time_format : optional strptime format for fast/strict parsing.
    """
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. This dataset is gitignored, so it is not in the repo — "
            "download the CSV (Jena Climate 2009-2016 from Kaggle / the MPI-BGC site) and place it "
            "under code/data/. See the README for details."
        )
    df = pd.read_csv(path)
    if time_col not in df.columns:
        raise KeyError(f"Time column {time_col!r} not found. Columns: {list(df.columns)}")
    df[time_col] = pd.to_datetime(df[time_col], format=time_format)
    df = df.set_index(time_col).sort_index()
    return df


def missing_report(df: pd.DataFrame) -> pd.Series:
    """Return the count of missing values per column (only columns with any)."""
    missing = df.isna().sum()
    return missing[missing > 0].sort_values(ascending=False)


def clean(
    df: pd.DataFrame,
    method: str = "time",
    sentinels: tuple[float, ...] = (-9999.0,),
) -> pd.DataFrame:
    """Remove duplicate timestamps, drop sentinel error codes, and interpolate gaps.

    The Jena wind columns use ``-9999`` to mark a failed reading; left in place these
    sentinels wreck means and correlations, so we convert them (and any other listed
    sentinels) to NaN first.

    `method="time"` interpolates respecting the spacing of the datetime index.
    Any leading/trailing gaps interpolation can't fill are forward/back filled.
    """
    df = df[~df.index.duplicated(keep="first")].sort_index()
    if sentinels:
        df = df.replace(list(sentinels), np.nan)
    df = df.interpolate(method=method, limit_direction="both")
    df = df.ffill().bfill()
    return df


def align(df: pd.DataFrame, freq: str = "1h", agg: str = "mean") -> pd.DataFrame:
    """Resample all sensors to a common frequency (default hourly mean).

    Resampling to hourly keeps the multi-year series tractable for analysis while
    preserving daily and seasonal structure.
    """
    resampler = df.resample(freq)
    out = getattr(resampler, agg)()
    return out.interpolate(method="time", limit_direction="both")


def infer_interval(df: pd.DataFrame) -> pd.Timedelta:
    """Return the most common spacing between consecutive timestamps."""
    return df.index.to_series().diff().mode().iloc[0]


def compare_imputation(
    series: pd.Series, method: str = "time", frac: float = 0.1, seed: int = 42
) -> pd.DataFrame:
    """Justify the interpolation choice by holding out known points and refilling.

    Randomly masks ``frac`` of the finite points, then refills them two ways:
      - interpolation (``method``, time-aware by default) — follows the local trend
      - mean-fill — replaces every gap with the series mean (ignores the trend)
    Returns a DataFrame indexed ``["interpolation", "mean_fill"]`` with a ``mae`` column:
    the mean absolute error against the held-out truth. Lower is better; on a trending
    signal interpolation wins, which is *why* the pipeline interpolates (following the
    graph's trend) rather than flattening gaps to an average.

    Requires a ``DatetimeIndex`` when ``method="time"`` (time-aware interpolation needs the
    timestamps); raises ``TypeError`` otherwise instead of a cryptic pandas error.
    """
    if method == "time" and not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError(
            "compare_imputation(method='time') requires a DatetimeIndex; "
            f"got {type(series.index).__name__}."
        )
    s = series.astype(float)
    finite_idx = s.dropna().index
    rng = np.random.default_rng(seed)
    n_mask = max(1, int(len(finite_idx) * frac))
    mask_idx = rng.choice(finite_idx.to_numpy(), size=n_mask, replace=False)
    held = s.copy()
    held.loc[mask_idx] = np.nan
    interp = held.interpolate(method=method, limit_direction="both")
    meanfill = held.fillna(s.mean())
    true_vals = s.loc[mask_idx]
    mae_interp = float((interp.loc[mask_idx] - true_vals).abs().mean())
    mae_mean = float((meanfill.loc[mask_idx] - true_vals).abs().mean())
    return pd.DataFrame({"mae": {"interpolation": mae_interp, "mean_fill": mae_mean}})
