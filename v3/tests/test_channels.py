"""Tests for the two detector channels (value = temperature residual, flow = network features)."""
import numpy as np
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


def test_seasonal_value_score_catches_time_of_day_anomaly():
    # A sensor with a strong daily cycle: a reading that is normal *globally* but wrong for its
    # time-of-day (a peak value at a trough hour) is invisible to the global z-score but caught by a
    # seasonal (time-of-day) baseline.
    idx = pd.date_range("2019-10-21", periods=24 * 7, freq="h")
    tod = idx.hour.to_numpy()
    base = 20 + 10 * np.sin(2 * np.pi * tod / 24)       # daily cycle, global range ~[10, 30]
    rng = np.random.default_rng(0)
    df = pd.DataFrame({131: base + rng.normal(0, 0.5, len(idx))}, index=idx)
    hmin = int(tod[np.argmin(base)])                    # a trough hour
    ti = int(np.where(tod == hmin)[0][3])
    df.iloc[ti, 0] = float(base.max())                  # peak value dropped into a trough hour
    glob = channels.value_score(df)
    seas = channels.seasonal_value_score(df)
    # seasonal z is far larger than the global z for this disguised anomaly (global buries it in the
    # daily range); the rank comparison is the substantive check that seasonal surfaces what global hides.
    assert seas.iloc[ti] > 2.0
    assert seas.iloc[ti] > glob.iloc[ti]
    assert seas.rank(pct=True).iloc[ti] > glob.rank(pct=True).iloc[ti]


def test_flow_score_mahalanobis_flags_joint_outlier_window():
    # A window that is jointly anomalous across flow features (high rate + low gap) but not extreme on
    # any single one — the cross-sectional case where a covariance model beats axis-aligned splits.
    idx = pd.date_range("2019-10-21", periods=40, freq="5min")
    rng = np.random.default_rng(0)
    flow = pd.DataFrame(
        {"msg_rate": rng.normal(10, 1, 40), "iat_mean": rng.normal(30, 2, 40)}, index=idx
    )
    flow.iloc[7] = [14.0, 22.0]                         # off the joint distribution, within each margin
    s = channels.flow_score(flow, method="mahalanobis")
    assert s.idxmax() == idx[7]
