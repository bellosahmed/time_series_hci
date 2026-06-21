"""Exact query engine — the TRUTH layer.

Responsibilities (Phase 5):
Compute exact answers deterministically. The LLM must NEVER compute these values.
- max/min, mean/std/range
- correlations
- missing-data count
- anomaly windows
- cluster membership
- per-sensor summaries

Each function returns a structured result (value + sensor + time range) so the
explanation layer can describe it without recomputing anything.
"""

from __future__ import annotations

import pandas as pd


def get_max(df: pd.DataFrame, sensor: str):
    """Exact maximum of `sensor` (value + timestamp). TODO."""
    raise NotImplementedError


def get_min(df: pd.DataFrame, sensor: str):
    """Exact minimum of `sensor` (value + timestamp). TODO."""
    raise NotImplementedError


def get_mean_std(df: pd.DataFrame, sensor: str):
    """Exact mean and std of `sensor`. TODO."""
    raise NotImplementedError


def get_correlation(df: pd.DataFrame, a: str, b: str) -> float:
    """Exact correlation between two sensors. TODO."""
    raise NotImplementedError


def count_missing(df: pd.DataFrame, sensor: str) -> int:
    """Exact count of missing values for `sensor`. TODO."""
    raise NotImplementedError


def sensor_summary(df: pd.DataFrame, sensor: str) -> dict:
    """Structured summary (min/max/mean/std/missing) for `sensor`. TODO."""
    raise NotImplementedError
