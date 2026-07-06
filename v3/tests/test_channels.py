"""Tests for the two detector channels (value = temperature residual, flow = network features)."""
import pandas as pd

from src import channels


def test_value_score_flags_the_modified_window():
    idx = pd.date_range("2019-10-21", periods=5, freq="5min")
    sensors = pd.DataFrame({131: [20.0] * 5, 132: [21, 21, 99, 21, 21.0]}, index=idx)
    s = channels.value_score(sensors)
    assert s.idxmax() == idx[2]                       # the extreme temperature window scores highest


def test_flow_score_flags_the_outlier_window():
    idx = pd.date_range("2019-10-21", periods=20, freq="5min")
    flow = pd.DataFrame({"msg_rate": [10.0] * 20, "iat_mean": [30.0] * 20}, index=idx)
    flow.iloc[5, 0] = 500.0                            # a burst (duplication-like)
    s = channels.flow_score(flow)
    assert s.idxmax() == idx[5]
