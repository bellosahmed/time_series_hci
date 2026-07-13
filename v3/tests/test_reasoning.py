"""Tests for the two-channel reasoning layer (value_zscores / flow_zscores / describe_window)."""

import numpy as np
import pandas as pd

from src import reasoning


def _channels():
    idx = pd.date_range("2024-01-01", periods=200, freq="5min")
    rng = np.random.default_rng(0)
    sensors = pd.DataFrame(
        {101.0: rng.normal(20, 1, 200), 102.0: rng.normal(20, 1, 200)}, index=idx
    )
    flow = pd.DataFrame(
        {"msg_rate": rng.normal(50, 5, 200), "iat_mean": rng.normal(1.0, 0.1, 200)}, index=idx
    )
    return sensors, flow


def test_value_zscores_sorted_by_magnitude():
    sensors, _ = _channels()
    z = reasoning.value_zscores(sensors, sensors.index[0])
    mags = z.abs().to_numpy()
    assert (mags[:-1] >= mags[1:]).all()


def test_flow_zscores_sorted_by_magnitude():
    _, flow = _channels()
    z = reasoning.flow_zscores(flow, flow.index[0])
    mags = z.abs().to_numpy()
    assert (mags[:-1] >= mags[1:]).all()


def test_describe_window_names_value_channel_driver():
    sensors, flow = _channels()
    spike_at = sensors.index[50]
    sensors.loc[spike_at, 101.0] += 15          # only the value channel spikes
    text = reasoning.describe_window(sensors, flow, spike_at)
    assert "101.0" in text and "value channel" in text


def test_describe_window_names_flow_channel_driver():
    sensors, flow = _channels()
    spike_at = flow.index[80]
    flow.loc[spike_at, "msg_rate"] += 100        # only the flow channel spikes
    text = reasoning.describe_window(sensors, flow, spike_at)
    assert "msg_rate" in text and "flow channel" in text


def test_describe_window_null_when_nothing_unusual():
    sensors, flow = _channels()
    calm_at = sensors.index[10]
    text = reasoning.describe_window(sensors, flow, calm_at, threshold=8.0)  # unreachable threshold
    assert "neither channel" in text
