"""Shared pytest fixtures/helpers for synthetic hourly sensor data.

Several test modules build a small hourly-indexed DataFrame of a few normally-distributed
sensor columns; this factors out that shared shape. Tests with a genuinely different data
shape (hand-crafted deterministic values, a seasonal signal, an injected spike, a bimodal
day-type pattern, non-weather column names) build their own DataFrame directly rather than
forcing it through this helper — the shape isn't actually shared there, only the surface
syntax of "date_range + rng.normal + DataFrame" is.
"""

import numpy as np
import pandas as pd
import pytest


def hourly_df(n, sensors, seed=0, start="2024-01-01"):
    """Build an hourly-indexed DataFrame of normally-distributed sensor columns.

    `sensors` maps column name -> (mean, std).
    """
    idx = pd.date_range(start, periods=n, freq="h")
    rng = np.random.default_rng(seed)
    data = {name: rng.normal(mean, std, n) for name, (mean, std) in sensors.items()}
    return pd.DataFrame(data, index=idx)


# Shared across the test_interface_*.py files: a small two-sensor frame and a full year of
# two-sensor data. Modules needing a different shape (e.g. test_agent.py's seasonal-cycle
# fixture, test_query_engine.py's hand-crafted deterministic one) define their own `df` locally,
# which shadows these for that module.
@pytest.fixture
def df():
    return hourly_df(48, {"T (degC)": (10, 5), "rh (%)": (70, 10)})


@pytest.fixture
def year_df():
    idx = pd.date_range("2016-01-01 00:00", "2017-01-01 00:00", freq="h")
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {"T (degC)": rng.normal(5, 3, len(idx)), "wv (m/s)": rng.normal(3, 1, len(idx))},
        index=idx,
    )
