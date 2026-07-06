"""Tests for v3 feature extraction — the VALUE channel (per-sensor temperature) and the FLOW
channel (windowed network-traffic features). Unit tests use tiny hand-built frames (fast,
deterministic); a couple of checks run on the real week (via the shared `week` fixture)."""
import numpy as np
import pandas as pd

from src import features


def _msgs(index, **cols):
    n = len(index)
    base = {"clientid": [131] * n, "msgid": list(range(n)), "temperature": [np.nan] * n,
            "label": [0] * n, "attack_type": ["none"] * n, "src_ip": ["x"] * n}
    base.update(cols)
    return pd.DataFrame(base, index=pd.to_datetime(index))


def test_sensor_series_pivots_one_column_per_sensor():
    df = _msgs(["2019-10-21 00:00:00", "2019-10-21 00:00:30", "2019-10-21 00:05:00"],
              clientid=[131, 132, 131], temperature=[10.0, 20.0, 12.0])
    s = features.sensor_series(df, freq="5min")
    assert 131 in s.columns and 132 in s.columns          # one column per sensor
    assert s[131].iloc[0] == 10.0 and s[132].iloc[0] == 20.0
    assert s[131].iloc[1] == 12.0                          # second 5-min window


def test_sensor_series_fill_interpolates_gaps():
    # two readings 10 min apart -> the middle 5-min window is a gap; fill=True interpolates it
    df = _msgs(["2019-10-21 00:00:00", "2019-10-21 00:10:00"], clientid=[131, 131],
               temperature=[10.0, 20.0])
    raw = features.sensor_series(df, "5min")
    filled = features.sensor_series(df, "5min", fill=True)
    assert raw[131].isna().any()                      # the gap exists...
    assert filled[131].isna().sum() == 0 and abs(filled[131].iloc[1] - 15.0) < 1e-6   # ...and is filled


def test_flow_features_counts_and_timing():
    df = _msgs(["2019-10-21 00:00:00", "2019-10-21 00:00:10", "2019-10-21 00:00:40",
                "2019-10-21 00:01:05"],
               temperature=[10.0, np.nan, 20.0, np.nan])
    f = features.flow_features(df, window="1min")
    w0 = pd.Timestamp("2019-10-21 00:00:00")
    assert f.loc[w0, "msg_rate"] == 3                      # 3 packets in the first minute
    assert f.loc[w0, "publish_rate"] == 2                  # 2 carry a temperature
    assert abs(f.loc[w0, "iat_mean"] - 20.0) < 1e-6        # gaps 10s & 30s → mean 20s


def test_flow_features_dup_ratio_flags_replay():
    df = _msgs(["2019-10-21 00:00:00", "2019-10-21 00:00:01", "2019-10-21 00:00:02"],
               clientid=[131, 131, 131], msgid=[5, 5, 6])   # (131,5) repeated → 1 of 3 is a dup
    f = features.flow_features(df, window="1min")
    assert abs(f.iloc[0]["dup_msgid_ratio"] - 1 / 3) < 1e-9


def test_window_labels_mark_attack_and_type():
    df = _msgs(["2019-10-23 02:10:00", "2019-10-23 02:10:30", "2019-10-23 00:00:00"],
               label=[1, 0, 0], attack_type=["modification", "none", "none"])
    wl = features.window_labels(df, window="1min")
    assert wl.loc[pd.Timestamp("2019-10-23 02:10:00"), "label"] == 1
    assert wl.loc[pd.Timestamp("2019-10-23 02:10:00"), "attack_type"] == "modification"
    assert wl.loc[pd.Timestamp("2019-10-23 00:00:00"), "label"] == 0


# ---- real-data structural checks (shared week fixture) ----

def test_flow_features_on_real_week_are_finite(week):
    f = features.flow_features(week, window="1min")
    assert isinstance(f.index, pd.DatetimeIndex)
    assert np.isfinite(f.to_numpy()).all()                 # no NaN/inf after windowing
    # exact invariants: every packet is counted in exactly one window
    assert f["msg_rate"].sum() == len(week)
    assert f["publish_rate"].sum() == int(week["temperature"].notna().sum())  # 32,288 readings


def test_window_labels_align_with_real_attacks(week):
    wl = features.window_labels(week, window="1min")
    assert wl["label"].sum() >= 1                          # some attack windows exist
    assert set(wl.loc[wl["label"] == 1, "attack_type"]) <= {"modification", "duplication", "mixed"}
