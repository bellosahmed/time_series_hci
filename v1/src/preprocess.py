"""Preprocessing: load, clean, align, and interpolate time-series data.

Responsibilities (Phase 1):
- parse and sort timestamps; set a datetime index
- handle missing values (interpolate / forward-fill) and report how many were missing
- resample / align multiple sensors to a common frequency
"""

from __future__ import annotations

import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load a time-series dataset from `path` into a DataFrame. TODO: implement."""
    raise NotImplementedError


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Sort timestamps, set datetime index, interpolate missing values. TODO."""
    raise NotImplementedError


def align(df: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    """Resample/align sensors to a common frequency. TODO."""
    raise NotImplementedError


def missing_report(df: pd.DataFrame) -> dict:
    """Return a count of missing values per column. TODO."""
    raise NotImplementedError
