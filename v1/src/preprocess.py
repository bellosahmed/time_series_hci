"""Preprocessing: load, clean, align, and interpolate time-series data.

Responsibilities (Phase 1):
- parse and sort timestamps; set a datetime index
- handle missing values (interpolate / forward-fill) and report how many were missing
- resample / align multiple sensors to a common frequency

Written generically so it works on any time-series CSV, with sensible defaults
for the Jena Climate dataset.
"""

from __future__ import annotations

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
