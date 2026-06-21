"""Behaviour analysis: describe how each sensor behaves over time.

Responsibilities (Phase 2):
- rolling mean / rolling std
- trend slope and STL decomposition (trend / seasonal / residual)
- correlation matrix across sensors
"""

from __future__ import annotations

import pandas as pd


def rolling_stats(series: pd.Series, window: int = 24) -> pd.DataFrame:
    """Return rolling mean and std for a series. TODO."""
    raise NotImplementedError


def decompose(series: pd.Series, period: int) -> object:
    """STL-decompose a series into trend/seasonal/residual. TODO."""
    raise NotImplementedError


def correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Return the correlation matrix across sensors. TODO."""
    raise NotImplementedError
