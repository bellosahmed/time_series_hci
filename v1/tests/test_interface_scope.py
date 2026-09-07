"""Tests for time/date scope resolution: does the interface find the right WINDOW of data,
and does that scope stay correctly isolated per sensor."""

import numpy as np
import pandas as pd
import pytest

from src import interface

# All 14 dataset columns except `wd (deg)`, a circular (compass-bearing) sensor whose "mean" is
# deliberately routed through circular statistics, not a plain average -- covered separately in
# test_interface_wind_circular.py. This list is for linear-mean cross-sensor checks.
LINEAR_SENSOR_COLUMNS = [
    "p (mbar)", "T (degC)", "Tpot (K)", "Tdew (degC)", "rh (%)", "VPmax (mbar)",
    "VPact (mbar)", "VPdef (mbar)", "sh (g/kg)", "H2OC (mmol/mol)", "rho (g/m**3)",
    "wv (m/s)", "max. wv (m/s)",
]


@pytest.fixture
def multi_sensor_df():
    """All 13 linear sensors, each a distinct, well-separated series (loc spaced 100 apart) so
    any cross-sensor mixup shows up as an obviously wrong mean, not a rounding-level coincidence.
    """
    idx = pd.date_range("2016-01-01 00:00", "2017-01-01 00:00", freq="h")
    rng = np.random.default_rng(2)
    data = {col: rng.normal(loc=100 * (i + 1), scale=1, size=len(idx))
            for i, col in enumerate(LINEAR_SENSOR_COLUMNS)}
    return pd.DataFrame(data, index=idx)


# "max. wv (m/s)" (wind gust) literally contains the word "max", which trips the "max" intent
# keyword ahead of "mean" if the raw column name is used in a question -- a real user would say
# "wind gust", the natural alias, which has no such collision. Use that phrasing here instead of
# the raw column name so the test reflects realistic usage rather than a self-inflicted collision.
def _mention(sensor: str) -> str:
    return "wind gust" if sensor == "max. wv (m/s)" else sensor


@pytest.mark.parametrize("sensor", LINEAR_SENSOR_COLUMNS)
def test_each_sensor_answer_matches_its_own_column_independently(multi_sensor_df, sensor):
    iface = interface.Interface(multi_sensor_df)
    ans = iface.ask(f"what was the mean {_mention(sensor)} last month?")
    expected = multi_sensor_df.loc["2016-12-01":"2016-12-31", sensor].mean()
    assert ans.value["mean"] == pytest.approx(expected)


def test_no_cross_sensor_bleed_when_every_sensor_is_asked_in_sequence(multi_sensor_df):
    # Ask about every sensor in turn on the SAME Interface (shared conversation history) -- that's
    # the realistic risk for cross-contamination, not a fresh instance per sensor.
    iface = interface.Interface(multi_sensor_df)
    seen = {}
    for sensor in LINEAR_SENSOR_COLUMNS:
        ans = iface.ask(f"what was the mean {_mention(sensor)} last month?")
        seen[sensor] = ans.value["mean"]
    for sensor in LINEAR_SENSOR_COLUMNS:
        expected = multi_sensor_df.loc["2016-12-01":"2016-12-31", sensor].mean()
        assert seen[sensor] == pytest.approx(expected), \
            f"{sensor}'s answer does not match its own independent computation -- possible bleed"
    # each sensor's mean is well-separated (loc spacing of 100) -- a real mixup would land ~100x
    # away from its neighbour, never a rounding-level discrepancy
    values = sorted(seen.values())
    assert all(values[i + 1] - values[i] > 50 for i in range(len(values) - 1))


def test_interface_anchors_last_month_to_dataset_end(year_df):
    iface = interface.Interface(year_df)
    assert iface._anchor == year_df.index.max()
    ans = iface.ask("what was the mean temperature last month?")
    expected = year_df.loc["2016-12-01":"2016-12-31"]
    assert ans.value["mean"] == pytest.approx(expected["T (degC)"].mean())


def test_interface_ordinal_month_with_year_beats_bare_year_default(year_df):
    iface = interface.Interface(year_df)
    ans = iface.ask("what was the mean temperature in the second month 2016?")
    expected = year_df.loc["2016-02-01":"2016-02-29"]
    assert ans.value["mean"] == pytest.approx(expected["T (degC)"].mean())
    assert ans.value["mean"] != pytest.approx(year_df["T (degC)"].mean())


def test_interface_hour_submask_narrows_a_named_period(year_df):
    iface = interface.Interface(year_df)
    ans = iface.ask("what was the mean temperature last month at 3pm?")
    scoped = year_df.loc["2016-12-01":"2016-12-31"]
    expected = scoped[scoped.index.hour == 15]
    assert ans.value["mean"] == pytest.approx(expected["T (degC)"].mean())


def test_interface_last_hour_is_the_final_clock_hour_of_the_record(year_df):
    iface = interface.Interface(year_df)
    ans = iface.ask("what was the mean temperature last hour?")
    expected = year_df.loc["2016-12-31 23:00":"2016-12-31 23:59:59.999999"]
    assert ans.value["mean"] == pytest.approx(expected["T (degC)"].mean())


def test_interface_now_is_the_single_latest_reading(year_df):
    iface = interface.Interface(year_df)
    ans = iface.ask("what was the mean temperature now?")
    assert ans.value["mean"] == pytest.approx(year_df["T (degC)"].iloc[-1])


def test_interface_night_bucket_wraps_past_midnight(year_df):
    iface = interface.Interface(year_df)
    ans = iface.ask("what was the mean wind speed at night in december 2016?")
    scoped = year_df.loc["2016-12-01":"2016-12-31"]
    expected = scoped[(scoped.index.hour >= 21) | (scoped.index.hour < 6)]
    assert ans.value["mean"] == pytest.approx(expected["wv (m/s)"].mean())


def test_interface_hour_only_question_scopes_across_whole_record(year_df):
    iface = interface.Interface(year_df)
    ans = iface.ask("what was the mean temperature at 3pm?")
    expected = year_df[year_df.index.hour == 15]
    assert ans.value["mean"] == pytest.approx(expected["T (degC)"].mean())


def test_resolve_period():
    assert interface.resolve_period("hottest in 2013")[:2] == ("2013-01-01", "2013-12-31")
    assert interface.resolve_period("average in july 2015")[:2] == ("2015-07-01", "2015-07-31")
    assert interface.resolve_period("between 2012 and 2014")[:2] == ("2012-01-01", "2014-12-31")
    assert interface.resolve_period("hottest ever")[0] is None


def test_resolve_period_repeated_identical_year_is_not_treated_as_a_range():
    # "6 April 2013 in the month of April 2013" names 2013 TWICE (once in the date, once in the
    # restated scope) -- before the fix this hit the >=2-years "range" branch and returned the
    # whole year 2013 instead of just April, silently discarding the month.
    q = "similar to 6 april 2013 in the month of april 2013"
    assert interface.resolve_period(q)[:2] == ("2013-04-01", "2013-04-30")
    # a genuine range of two DIFFERENT years must still work
    assert interface.resolve_period("compare 2012 to 2014")[:2] == ("2012-01-01", "2014-12-31")


def test_resolve_period_seasons():
    # summer = Jun–Aug of the year
    assert interface.resolve_period("hottest in summer 2012")[:2] == ("2012-06-01", "2012-08-31")
    # winter rolls back to the previous December
    start, end, label = interface.resolve_period("coldest in winter 2013")
    assert start == "2012-12-01" and end == "2013-02-28" and "winter" in label


def test_ask_daytype_classifies_specific_day():
    # a warming ramp across 60 days so K-means finds distinct day-types
    idx = pd.date_range("2013-01-01", periods=24 * 60, freq="h")
    base = 20.0 * np.arange(len(idx)) / len(idx)
    df = pd.DataFrame({"T (degC)": base + np.random.default_rng(0).normal(0, 1, len(idx))}, index=idx)
    ans = interface.Interface(df).ask("what kind of day was 2 January 2013?")
    assert ans.intent == "daytype"                      # clustering is now reachable via the interface
    assert isinstance(ans.value, dict) and "label" in ans.value
    assert ans.value["label"] in ans.text


def test_resolve_threshold():
    assert interface.resolve_threshold("how often above 30 degrees?") == (">", 30.0)
    assert interface.resolve_threshold("colder than -5 in 2014") == ("<", -5.0)
    assert interface.resolve_threshold("what is the average?") is None  # no comparison
    # scope year after the threshold is ignored (the number follows the comparison word)
    assert interface.resolve_threshold("how often above 30 in 2013?") == (">", 30.0)
    # year-range edge case fixed: a genuine threshold in 1900-2100 right after "above" is kept
    assert interface.resolve_threshold("times above 2000") == (">", 2000.0)
    # but a year that is only a scope (not directly after the comparison word) is not a threshold
    assert interface.resolve_threshold("above average in 2015") is None


def test_ask_scopes_to_year():
    # strictly increasing temp across 2013 -> 2014; the all-time max is in 2014.
    idx = pd.date_range("2013-06-01", periods=400, freq="D")
    df = pd.DataFrame({"T (degC)": np.arange(400, dtype=float)}, index=idx)
    ans = interface.Interface(df).ask("what was the hottest temperature in 2013?")
    assert ans.intent == "max"
    assert ans.value["value"] == df.loc["2013", "T (degC)"].max()   # the 2013 max, not global
    assert ans.value["value"] < df["T (degC)"].max()               # proves it didn't use the whole record
    assert "2013" in ans.text


def test_resolve_day():
    assert interface.resolve_day("tell me the weather in 26 january")[:3] == (1, 26, None)
    assert interface.resolve_day("weather on 5 july 2015")[:3] == (7, 5, 2015)
    assert interface.resolve_day("conditions on 2015-01-26")[:3] == (1, 26, 2015)
    assert interface.resolve_day("what was the average?") is None


def test_resolve_day_finds_the_dated_month_even_when_a_second_dayless_month_sorts_earlier():
    # "5 September 2012 in June 2016": MONTHS iterates in calendar order, so "june" (month 6)
    # is checked before "september" (month 9) -- before the fix, finding "june" with no day
    # attached made the whole function give up (`break`) instead of continuing on to find
    # "september", which DOES have a day next to it. Must still find September 5, 2012, not None.
    q = "recommend a day to go out in 5 september 2012 in june 2016"
    assert interface.resolve_day(q)[:3] == (9, 5, 2012)
    # a bare month with genuinely no day anywhere must still correctly return None
    assert interface.resolve_day("tell me about march") is None


def test_ask_weather_report():
    idx = pd.date_range("2015-01-25", periods=72, freq="h")     # 3 days incl 26 Jan
    df = pd.DataFrame({"T (degC)": np.linspace(-5, 5, 72),
                       "rh (%)": np.linspace(60, 90, 72)}, index=idx)
    ans = interface.Interface(df).ask("tell me the weather on 26 january 2015")
    assert ans.intent == "weather"
    assert "T (degC)" in ans.value and "rh (%)" in ans.value
    assert "26 January 2015" in ans.text


def test_ask_threshold_count():
    idx = pd.date_range("2014-01-01", periods=10, freq="D")
    df = pd.DataFrame({"T (degC)": np.arange(0, 50, 5, dtype=float)}, index=idx)  # 0..45
    ans = interface.Interface(df).ask("how many times was it above 22 degrees?")
    assert ans.intent == "threshold"
    assert ans.value["count"] == 5   # 25,30,35,40,45


def test_resolve_compare():
    assert interface.resolve_compare("compare 2013 and 2014")[:2] == (2013, 2014)
    assert interface.resolve_compare("was 2013 hotter than 2014?")[2] == "max"
    assert interface.resolve_compare("average in 2013") is None


def test_ask_compare():
    idx = pd.date_range("2013-01-01", periods=730, freq="D")     # 2013 + 2014
    t = np.where(idx.year == 2014, 15.0, 5.0)                     # 2014 warmer
    df = pd.DataFrame({"T (degC)": t}, index=idx)
    ans = interface.Interface(df).ask("was 2013 hotter than 2014?")
    assert ans.intent == "compare"
    assert "2014" in ans.text and "2013" in ans.text


def test_compare_uses_raw_extrema_like_max_intent():
    # 'was 2013 hotter than 2014' must report the SAME peak the max intent would (computed on the
    # native-resolution `raw` frame), not the hourly-mean max -- otherwise the same fact ("2013's
    # highest temperature") gets two different numbers depending on how it's asked.
    raw_idx = pd.date_range("2013-01-01", "2014-12-31 23:00", freq="h")
    base = np.where(raw_idx.year == 2013, 20.0, 10.0)
    raw = pd.DataFrame({"T (degC)": base}, index=raw_idx)
    # a single sub-hourly-looking spike only present in the raw frame (survives no averaging here,
    # but we simulate the smoothing gap by giving the hourly frame a lower peak explicitly)
    raw.iloc[100, 0] = 35.0            # true 2013 peak lives in raw
    hourly = raw.resample("2h").mean()  # coarser grid -> its max is the averaged-down value
    iface = interface.Interface(hourly, raw=raw)
    ans = iface.ask("was 2013 hotter than 2014?")
    assert ans.intent == "compare"
    # 2013's reported peak must equal the RAW max (35.0), not the coarser frame's smoothed peak
    assert ans.value["a"] == raw.loc["2013", "T (degC)"].max() == 35.0
    assert ans.value["a"] != hourly.loc["2013", "T (degC)"].max()


def _multiyear_df():
    idx = pd.date_range("2014-01-01", "2015-12-31 23:00", freq="h")
    n = len(idx)
    return pd.DataFrame({"T (degC)": [float(i % 40) for i in range(n)],
                         "rh (%)": [40.0 + (i % 50) for i in range(n)]}, index=idx)


def test_bare_month_no_year_asks_for_year():
    # A month with no year is under-specified: without a year it would silently use the whole record
    # (June's max would show the all-time July peak). Ask which year instead of guessing.
    iface = interface.Interface(_multiyear_df())
    for q in ["hottest temperature in June", "average humidity in June",
              "how many hours was the temperature above 30 in June"]:
        ans = iface.ask(q)
        iface.history = []
        assert ans.value is None, f"{q!r} should not be answered on a guessed scope"
        assert "which year" in ans.text.lower() and "June" in ans.text


def test_month_with_year_still_answers():
    iface = interface.Interface(_multiyear_df())
    ans = iface.ask("hottest temperature in June 2015")
    assert ans.value is not None                      # a real scope -> answered
    assert "2015" in str(ans.understood) or "June 2015" in str(ans.understood)


def test_bare_day_still_resolves_across_years():
    # A bare DAY ("26 January") already means that day across all years -> must NOT ask for a year.
    iface = interface.Interface(_multiyear_df())
    ans = iface.ask("temperature on 26 January")
    assert "which year" not in ans.text.lower()


def test_offtopic_with_month_is_not_asked_for_year():
    # "my cat in June" names no sensor/intent -> it must decline, not ask which year's June.
    iface = interface.Interface(_multiyear_df())
    ans = iface.ask("what is the name of my cat in June")
    assert "which year" not in ans.text.lower()


def test_recommend_bare_month_still_ranks_across_years():
    # The recommend intent deliberately ranks a bare month across all years -> not redirected.
    iface = interface.Interface(_multiyear_df())
    ans = iface.ask("recommend the hottest days in June")
    assert ans.intent == "recommend"
    assert "which year" not in ans.text.lower()


def test_ask_trend_rising():
    idx = pd.date_range("2014-01-01", periods=24 * 60, freq="h")
    df = pd.DataFrame({"T (degC)": np.linspace(0, 10, len(idx))}, index=idx)  # steadily up
    ans = interface.Interface(df).ask("is temperature rising in 2014?")
    assert ans.intent == "trend"
    assert ans.value > 0


def test_ask_forecast():
    idx = pd.date_range("2014-01-01", periods=24 * 400, freq="h")
    df = pd.DataFrame({"T (degC)": np.linspace(0, 12, len(idx))}, index=idx)
    ans = interface.Interface(df).ask("what will the temperature be next week?")
    assert ans.intent == "forecast"
    assert ans.value is not None and len(ans.value) == 14


# Regression coverage for docs/archive/MULTI_SENSOR_ANSWERS_PLAN.md: naming more than one sensor on a
# single-statistic question used to silently keep only sensors[0] and drop the rest with no
# signal to the user, even though the router had correctly resolved every named sensor.

def test_mean_with_two_sensors_answers_both_not_just_the_first(multi_sensor_df):
    iface = interface.Interface(multi_sensor_df)
    ans = iface.ask("what was the mean p (mbar) and T (degC) last month?")
    expected_p = multi_sensor_df.loc["2016-12-01":"2016-12-31", "p (mbar)"].mean()
    expected_t = multi_sensor_df.loc["2016-12-01":"2016-12-31", "T (degC)"].mean()
    assert "p (mbar)" in ans.text and "T (degC)" in ans.text
    assert f"{expected_p:.2f}" in ans.text
    assert f"{expected_t:.2f}" in ans.text
    assert set(ans.sensors) == {"p (mbar)", "T (degC)"}


def test_multi_sensor_answer_matches_the_single_sensor_answer_exactly(multi_sensor_df):
    # The multi-sensor path must call the identical query_engine function as the single-sensor
    # path, not a second implementation of the same statistic -- so the two can never disagree.
    iface = interface.Interface(multi_sensor_df)
    solo = iface.ask("what was the max p (mbar) last month?")
    combined = iface.ask("what was the max p (mbar) and T (degC) last month?")
    assert f"{solo.value['value']:.2f}" in combined.text


def test_multi_sensor_threshold_is_not_mislabelled_as_a_vague_summary(multi_sensor_df):
    # Regression: `intent` can still hold the "summary" default (no statistic keyword found)
    # when `thr` is what actually routes the question, and the multi-sensor helper used to pass
    # that stale "summary" straight through to `_reply`, which made `_assumption_notes` wrongly
    # append "no specific statistic named" onto an answer that named one via `thr`.
    iface = interface.Interface(multi_sensor_df)
    ans = iface.ask("how often was p (mbar) and T (degC) above 100 last month?")
    assert ans.intent == "threshold"
    assert "no specific statistic named" not in ans.text
    assert "p (mbar) was above 100" in ans.text
    assert "T (degC) was above 100" in ans.text


def test_correlation_with_three_sensors_stays_pairwise_and_says_so(multi_sensor_df):
    iface = interface.Interface(multi_sensor_df)
    ans = iface.ask("how do p (mbar), T (degC) and Tpot (K) relate?")
    assert ans.intent == "correlation"
    assert ans.sensors == ["p (mbar)", "T (degC)"]          # first two, not silently all three
    assert "compares two sensors at a time" in ans.text
    assert "Tpot (K)" in ans.text                            # the dropped one is still named
