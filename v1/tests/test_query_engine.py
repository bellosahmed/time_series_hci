"""Unit tests for the query engine (the truth layer).

These confirm the exact computations match pandas ground truth on a small,
fully-controlled synthetic dataset. This is the core trust guarantee: the engine
must never disagree with the underlying data.
"""

import numpy as np
import pandas as pd
import pytest

from src import query_engine as qe


@pytest.fixture
def df():
    idx = pd.date_range("2024-01-01", periods=10, freq="h")
    return pd.DataFrame(
        {
            "temp": [1.0, 2.0, 3.0, np.nan, 5.0, 10.0, -4.0, 3.0, 2.0, 1.0],
            "humid": [50, 51, 49, 48, 47, 30, 80, 52, 53, 50],
        },
        index=idx,
    )


def test_get_max_matches_pandas(df):
    res = qe.get_max(df, "temp")
    assert res["value"] == df["temp"].max()
    assert res["when"] == df["temp"].idxmax()


def test_get_min_matches_pandas(df):
    res = qe.get_min(df, "temp")
    assert res["value"] == df["temp"].min()
    assert res["when"] == df["temp"].idxmin()


def test_query_engine_functions_all_return_plain_dicts():
    # Locks in a consistent return-type contract: every query_engine function returns a plain
    # dict (get_max/get_min used to return a bespoke `Extremum` dataclass, forcing callers to
    # special-case them -- see interface.py's chart-building code, which no longer needs to).
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=pd.date_range("2024-01-01", periods=3, freq="h"))
    for result in (
        qe.get_max(df, "x"), qe.get_min(df, "x"), qe.get_mean_std(df, "x"),
        qe.threshold_events(df, "x", ">", 1.5), qe.weather_report(df, ["x"]).get("x", {}),
        qe.sensor_summary(df, "x"),
    ):
        assert isinstance(result, dict)


def test_mean_std_matches_pandas(df):
    res = qe.get_mean_std(df, "temp")
    assert res["mean"] == pytest.approx(df["temp"].mean())
    assert res["std"] == pytest.approx(df["temp"].std())
    assert res["range"] == pytest.approx(df["temp"].max() - df["temp"].min())


def test_correlation_matches_pandas(df):
    assert qe.get_correlation(df, "temp", "humid") == pytest.approx(
        df["temp"].corr(df["humid"])
    )


def test_count_missing(df):
    assert qe.count_missing(df, "temp") == 1
    assert qe.count_missing(df, "humid") == 0
    assert qe.count_missing(df) == {"temp": 1, "humid": 0}


def test_unknown_sensor_raises(df):
    with pytest.raises(KeyError):
        qe.get_max(df, "does_not_exist")


def test_anomaly_windows_grouping():
    idx = pd.date_range("2024-01-01", periods=6, freq="h")
    flags = pd.Series([False, True, True, False, True, False], index=idx)
    windows = qe.get_anomaly_windows(flags)
    assert len(windows) == 2
    assert windows[0]["n_points"] == 2
    assert windows[0]["start"] == idx[1]
    assert windows[0]["end"] == idx[2]
    assert windows[1]["n_points"] == 1


def test_sensor_summary_keys(df):
    s = qe.sensor_summary(df, "temp")
    for key in ("mean", "std", "min", "max", "range", "missing", "max_at", "min_at"):
        assert key in s
