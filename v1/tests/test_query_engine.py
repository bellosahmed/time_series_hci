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


def test_correlation_breakdown_pearson_and_r_squared_match_pandas():
    idx = pd.date_range("2024-01-01", periods=365, freq="D")
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0, 1, 365), index=idx)
    b = -0.6 * a + rng.normal(0, 1, 365)
    df = pd.DataFrame({"a": a, "b": b}, index=idx)
    bd = qe.correlation_breakdown(df, "a", "b")
    expected_pearson = df["a"].corr(df["b"])
    assert bd["pearson"] == pytest.approx(expected_pearson)
    assert bd["r_squared"] == pytest.approx(expected_pearson ** 2)
    assert bd["spearman"] == pytest.approx(df["a"].rank().corr(df["b"].rank()))


def test_correlation_breakdown_seasonal_split_isolates_the_driving_season():
    # a is strongly related to b ONLY in summer months (Jun-Aug); flat/unrelated the rest of
    # the year -- the whole-record Pearson should be a moderate figure, but the seasonal split
    # must reveal that it's really a summer-only effect.
    idx = pd.date_range("2020-01-01", "2022-12-31", freq="D")
    rng = np.random.default_rng(1)
    a = pd.Series(rng.normal(0, 1, len(idx)), index=idx)
    is_summer = idx.month.isin((6, 7, 8))
    b = pd.Series(rng.normal(0, 1, len(idx)), index=idx)
    b[is_summer] = -a[is_summer] + rng.normal(0, 0.1, int(is_summer.sum()))
    df = pd.DataFrame({"a": a, "b": b}, index=idx)
    bd = qe.correlation_breakdown(df, "a", "b")
    assert set(bd["by_season"]) == {"winter", "spring", "summer", "autumn"}
    assert bd["by_season"]["summer"] < -0.9          # strong, engineered relationship
    assert abs(bd["by_season"]["winter"]) < 0.3       # no real relationship the rest of the year


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


# --- slice_window: shared by Interface and Agent, which each used to carry a near-duplicate copy ---

def test_slice_window_no_bounds_returns_whole_df_and_empty_label(df):
    data, where = qe.slice_window(df, None, None)
    assert data is df
    assert where == ""


def test_slice_window_with_external_label_on_hit(df):
    data, where = qe.slice_window(df, "2024-01-01 02:00", "2024-01-01 05:00", label="a window")
    assert len(data) == 4
    assert where == " in a window"


def test_slice_window_with_external_label_on_miss_returns_none(df):
    data, where = qe.slice_window(df, "2099-01-01", "2099-01-02", label="the year 2099")
    assert data is None
    assert where == "the year 2099"


def test_slice_window_auto_label_two_sided(df):
    data, where = qe.slice_window(df, "2024-01-01 02:00", "2024-01-01 05:00")
    assert len(data) == 4
    assert where == " in 2024-01-01 02:00 to 2024-01-01 05:00"


def test_slice_window_auto_label_one_sided_start_only(df):
    data, where = qe.slice_window(df, "2024-01-01 05:00", None)
    assert where == " from 2024-01-01 05:00"


def test_slice_window_auto_label_one_sided_end_only(df):
    data, where = qe.slice_window(df, None, "2024-01-01 05:00")
    assert where == " up to 2024-01-01 05:00"


def test_slice_window_auto_label_on_miss_also_returns_none(df):
    # Previously agent.py's own copy of this logic (`_scope`) did NOT do this -- it silently
    # returned an empty DataFrame, which crashed downstream query_engine calls (e.g. get_max on
    # an empty Series raises ValueError). This is a real fix, not just deduplication.
    data, where = qe.slice_window(df, "2099-01-01", "2099-01-02")
    assert data is None
    assert where == "2099-01-01 to 2099-01-02"


# --- rank_calendar_days / find_analog_periods ("recommend" intent's truth layer) -------------

@pytest.fixture
def year_temp_df():
    # One value per day for a full year, following a smooth seasonal sine (summer peak, winter
    # trough), so "hottest"/"coldest" and the honest-threshold check have unambiguous answers.
    idx = pd.date_range("2020-01-01", periods=366, freq="D")
    base = 10 + 15 * np.sin((idx.dayofyear - 80) / 365 * 2 * np.pi)
    return pd.DataFrame({"T (degC)": base}, index=idx)


def test_rank_calendar_days_hottest_matches_pandas_idxmax(year_temp_df):
    result = qe.rank_calendar_days(year_temp_df, year_temp_df, "T (degC)", ascending=False, top_n=3)
    daily = year_temp_df["T (degC)"].resample("1D").mean()
    assert result["candidates"][0]["date"] == daily.idxmax()
    assert result["candidates"][0]["value"] == pytest.approx(daily.max())
    assert len(result["candidates"]) == 3


def test_rank_calendar_days_coldest_matches_pandas_idxmin(year_temp_df):
    result = qe.rank_calendar_days(year_temp_df, year_temp_df, "T (degC)", ascending=True, top_n=1)
    daily = year_temp_df["T (degC)"].resample("1D").mean()
    assert result["candidates"][0]["date"] == daily.idxmin()
    assert result["candidates"][0]["value"] == pytest.approx(daily.min())


def test_rank_calendar_days_meets_threshold_when_scope_is_genuinely_extreme(year_temp_df):
    summer = year_temp_df[(year_temp_df.index.month == 7)]
    result = qe.rank_calendar_days(summer, year_temp_df, "T (degC)", ascending=False, top_n=3)
    assert result["meets_threshold"] is True


def test_rank_calendar_days_does_not_meet_threshold_when_scope_is_never_extreme(year_temp_df):
    # December in this synthetic year is cold -- "hottest days in December" must be flagged as
    # not genuinely hot rather than silently presented as a real recommendation.
    december = year_temp_df[year_temp_df.index.month == 12]
    result = qe.rank_calendar_days(december, year_temp_df, "T (degC)", ascending=False, top_n=3)
    assert result["meets_threshold"] is False
    assert result["threshold"] == pytest.approx(
        year_temp_df["T (degC)"].resample("1D").mean().quantile(0.75))


def test_rank_calendar_days_empty_scope_returns_no_candidates(year_temp_df):
    empty = year_temp_df.iloc[0:0]
    result = qe.rank_calendar_days(empty, year_temp_df, "T (degC)", ascending=False)
    assert result == {"candidates": [], "meets_threshold": False, "threshold": None}


def test_find_analog_periods_target_day_matches_itself_excluded(year_temp_df):
    # A day right next to the target should be its closest analog (smooth sinusoid -> monotone
    # distance with time-offset), and the target's own bucket must never appear in candidates.
    target = pd.Timestamp("2020-07-01")
    result = qe.find_analog_periods(year_temp_df, ["T (degC)"], target, target, freq="D", top_n=5)
    whens = [c["when"] for c in result["candidates"]]
    assert target not in whens
    assert len(whens) == 5
    # closest candidate should be one of the immediate neighbouring days
    assert min(abs((w - target).days) for w in whens) <= 2


def test_find_analog_periods_year_granularity_labels_whole_years():
    idx = pd.date_range("2009-01-01", "2016-12-31", freq="D")
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"T (degC)": 10 + rng.normal(0, 1, len(idx))}, index=idx)
    result = qe.find_analog_periods(df, ["T (degC)"], pd.Timestamp("2013-01-01"),
                                     pd.Timestamp("2013-12-31"), freq="YS", top_n=3)
    assert result["target"] is not None
    assert len(result["candidates"]) == 3
    assert all(c["when"].year != 2013 for c in result["candidates"])


def test_find_analog_periods_no_data_in_target_range_returns_empty():
    idx = pd.date_range("2020-01-01", periods=30, freq="D")
    df = pd.DataFrame({"T (degC)": range(30)}, index=idx)
    result = qe.find_analog_periods(df, ["T (degC)"], pd.Timestamp("2099-01-01"),
                                     pd.Timestamp("2099-01-02"), freq="D")
    assert result == {"target": None, "candidates": []}


def test_find_analog_periods_search_bound_restricts_candidates(year_temp_df):
    # Without a search bound, analogs to a July day can come from anywhere in the record; with
    # one, they must stay inside it -- the "stay within April 2013" fix.
    target = pd.Timestamp("2020-07-01")
    unrestricted = qe.find_analog_periods(year_temp_df, ["T (degC)"], target, target, freq="D",
                                           top_n=40)
    restricted = qe.find_analog_periods(year_temp_df, ["T (degC)"], target, target, freq="D",
                                         top_n=40, search_start="2020-07-01", search_end="2020-07-31")
    assert all(pd.Timestamp("2020-07-01") <= c["when"] <= pd.Timestamp("2020-07-31")
               for c in restricted["candidates"])
    assert len(restricted["candidates"]) == 30          # every other July day, none outside it
    assert len(unrestricted["candidates"]) == 40          # top_n cap, drawn from the whole year


def test_find_analog_periods_search_bound_with_no_other_data_in_range_returns_empty():
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    df = pd.DataFrame({"T (degC)": range(10)}, index=idx)
    target = pd.Timestamp("2020-01-01")
    result = qe.find_analog_periods(df, ["T (degC)"], target, target, freq="D",
                                     search_start="2020-01-01", search_end="2020-01-01")
    assert result["candidates"] == []
