"""Tests for multi-sensor reasoning (describe_moment / co_movement)."""

import numpy as np
import pandas as pd

from src import reasoning


def _df():
    idx = pd.date_range("2024-01-01", periods=48, freq="h")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {"T (degC)": rng.normal(10, 5, 48), "rh (%)": rng.normal(70, 10, 48)},
        index=idx,
    )


def test_sensor_zscores_sorted_by_magnitude():
    df = _df()
    z = reasoning.sensor_zscores(df, df.index[0])
    mags = z.abs().to_numpy()
    assert (mags[:-1] >= mags[1:]).all()  # descending |z|


def test_describe_moment_verdict_single_sensor():
    # one sensor spikes alone -> "sensor glitch" verdict
    idx = pd.date_range("2024-01-01", periods=200, freq="h")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"T (degC)": rng.normal(10, 1, 200),
                       "rh (%)": rng.normal(70, 1, 200),
                       "p (mbar)": rng.normal(1000, 1, 200)}, index=idx)
    df.loc[idx[50], "T (degC)"] += 20          # only temperature moves
    text = reasoning.describe_moment(df, idx[50])
    assert "sensor glitch" in text


def test_co_movement_names_the_actual_target():
    """The explanation must reference the queried sensor, not a hardcoded one."""
    df = _df()
    text = reasoning.co_movement(df, df.index[5], target="rh (%)")
    assert "rh (%)" in text
    assert "temperature" not in text  # regression: was hardcoded to "temperature"
