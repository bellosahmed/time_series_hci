"""Tests for the new time-constraint parsers: relative/ordinal periods and hour-of-day."""

import pandas as pd

from src.interface_parsing import (
    hour_mask,
    resolve_any_period,
    resolve_hour,
    resolve_ordinal_unit_in_year,
    resolve_relative_period,
)

ANCHOR = pd.Timestamp("2017-01-01 00:10:00")


def test_relative_now():
    assert resolve_relative_period("what is it now", ANCHOR) == (str(ANCHOR), str(ANCHOR), "now")


def test_relative_this_hour():
    start, end, label = resolve_relative_period("temperature this hour", ANCHOR)
    assert start == str(ANCHOR.floor("h"))
    assert end == str(ANCHOR)
    assert label == "this hour"


def test_relative_last_hour():
    start, end, label = resolve_relative_period("temperature last hour", ANCHOR)
    assert start == str(ANCHOR.floor("h") - pd.Timedelta(hours=1))
    assert end == str(ANCHOR.floor("h") - pd.Timedelta(microseconds=1))
    assert label == "last hour"


def test_relative_this_week():
    start, end, label = resolve_relative_period("wind speed this week", ANCHOR)
    assert start == str(ANCHOR - pd.Timedelta(days=7))
    assert end == str(ANCHOR)
    assert label == "this week"


def test_relative_last_week():
    start, end, label = resolve_relative_period("wind speed last week", ANCHOR)
    assert start == str(ANCHOR - pd.Timedelta(days=14))
    assert end == str(ANCHOR - pd.Timedelta(days=7) - pd.Timedelta(microseconds=1))
    assert label == "last week"


def test_relative_last_month_is_december_2016():
    assert resolve_relative_period("temperature last month", ANCHOR)[:2] == ("2016-12-01", "2016-12-31")


def test_relative_this_month_is_partial_january():
    start, end, _label = resolve_relative_period("temperature this month", ANCHOR)
    assert start == "2017-01-01 00:00:00"
    assert end == str(ANCHOR)


def test_relative_last_year_is_2016():
    # ANCHOR is 2017-01-01 00:10 -- "last year" is the last full calendar year, 2016.
    assert resolve_relative_period("temperature last year", ANCHOR)[:2] == ("2016-01-01", "2016-12-31")


def test_relative_this_year_is_partial_2017_to_date():
    # ANCHOR's own year is 2017 (a 10-minute stub), consistent with "this month" resolving to a
    # partial January 2017 rather than December 2016.
    start, end, _label = resolve_relative_period("temperature this year", ANCHOR)
    assert start == "2017-01-01 00:00:00"
    assert end == str(ANCHOR)


def test_relative_period_yields_nothing_when_a_year_is_present():
    assert resolve_relative_period("temperature last month 2013", ANCHOR) == (None, None, "")


def test_relative_period_yields_nothing_for_unrelated_question():
    assert resolve_relative_period("what was the hottest day?", ANCHOR) == (None, None, "")


def test_ordinal_last_month_of_year_is_december():
    assert resolve_ordinal_unit_in_year("what was the wind speed last month 2013")[:2] == \
        ("2013-12-01", "2013-12-31")


def test_ordinal_word_month_of_year():
    assert resolve_ordinal_unit_in_year("temperature in the second month 2016")[:2] == \
        ("2016-02-01", "2016-02-29")  # 2016 is a leap year, Feb has 29 days


def test_ordinal_numeric_week_of_year():
    expected_start = pd.Timestamp("2013-01-01") + pd.Timedelta(days=33 * 7)
    start, _end, label = resolve_ordinal_unit_in_year("what happened in the 34th week of 2013")
    assert start == str(expected_start.date())
    assert label == "week 34 of 2013"


def test_ordinal_last_week_of_year_ends_on_dec_31():
    _start, end, _label = resolve_ordinal_unit_in_year("last week of 2013")
    assert end == "2013-12-31"


def test_ordinal_last_day_of_year():
    assert resolve_ordinal_unit_in_year("temperature on the last day of 2013")[:2] == \
        ("2013-12-31", "2013-12-31")


def test_ordinal_first_hour_of_year():
    start, _end, _label = resolve_ordinal_unit_in_year("what was the wind on the first hour of 2013")
    assert start == "2013-01-01 00:00:00"


def test_ordinal_yields_nothing_without_a_year():
    assert resolve_ordinal_unit_in_year("last month") == (None, None, "")


def test_ordinal_yields_nothing_for_a_bare_year():
    assert resolve_ordinal_unit_in_year("temperature in 2013") == (None, None, "")


def test_resolve_hour_pm_clock_time():
    assert resolve_hour("temperature at 3pm")[:2] == (15, 16)


def test_resolve_hour_am_clock_time():
    assert resolve_hour("wind speed at 9am")[:2] == (9, 10)


def test_resolve_hour_24h_clock():
    assert resolve_hour("temperature at 15:00")[:2] == (15, 16)


def test_resolve_hour_oclock():
    assert resolve_hour("temperature at 3 o'clock")[:2] == (3, 4)


def test_resolve_hour_noon_and_midnight():
    assert resolve_hour("temperature at noon")[:2] == (12, 13)
    assert resolve_hour("temperature at midnight")[:2] == (0, 1)


def test_resolve_hour_time_of_day_buckets():
    assert resolve_hour("temperature this morning")[:2] == (6, 12)
    assert resolve_hour("temperature this afternoon")[:2] == (12, 18)
    assert resolve_hour("temperature this evening")[:2] == (18, 21)
    assert resolve_hour("temperature at night")[:2] == (21, 6)


def test_resolve_hour_yields_nothing_for_unrelated_question():
    assert resolve_hour("what was the hottest day?") == (None, None, "")


def test_hour_mask_normal_range():
    idx = pd.DatetimeIndex([f"2016-01-01 {h:02d}:00" for h in range(24)])
    mask = hour_mask(idx, 15, 16)
    assert list(idx[mask].hour) == [15]


def test_hour_mask_wraps_past_midnight():
    idx = pd.DatetimeIndex([f"2016-01-01 {h:02d}:00" for h in range(24)])
    mask = hour_mask(idx, 21, 6)
    assert sorted(idx[mask].hour) == [0, 1, 2, 3, 4, 5, 21, 22, 23]


def test_resolve_any_period_prefers_ordinal_over_bare_year_default():
    start, end, _label = resolve_any_period("second month 2016", ANCHOR)
    assert (start, end) == ("2016-02-01", "2016-02-29")


def test_resolve_any_period_falls_back_to_explicit_period():
    assert resolve_any_period("hottest in 2013", ANCHOR)[:2] == ("2013-01-01", "2013-12-31")


def test_resolve_any_period_works_without_an_anchor_for_explicit_period():
    assert resolve_any_period("hottest in 2013") == ("2013-01-01", "2013-12-31", "2013")


def test_resolve_any_period_yields_nothing_for_relative_phrase_without_an_anchor():
    assert resolve_any_period("last month") == (None, None, "")


def test_resolve_any_period_falls_back_to_relative_with_an_anchor():
    assert resolve_any_period("last month", ANCHOR)[:2] == ("2016-12-01", "2016-12-31")


def test_resolve_hour_rejects_out_of_range_12h_clock_time():
    # "13pm" is not a valid 12-hour-clock time (only 1-12 are); must not silently accept it.
    assert resolve_hour("temperature at 13pm") == (None, None, "")
    assert resolve_hour("temperature at 19pm") == (None, None, "")


def test_resolve_hour_still_accepts_valid_12h_boundary_values():
    assert resolve_hour("temperature at 12pm")[:2] == (12, 13)
    assert resolve_hour("temperature at 1am")[:2] == (1, 2)


def test_relative_period_refuses_chained_ordinal_construct():
    # "third week of last year" is a chained ordinal+relative construct that isn't fully
    # resolvable -- must not silently answer using only "last year" and dropping "third week".
    assert resolve_relative_period("third week of last year", ANCHOR) == (None, None, "")
    assert resolve_relative_period("second month of this year", ANCHOR) == (None, None, "")


def test_relative_period_still_resolves_plain_last_year_without_a_chained_prefix():
    assert resolve_relative_period("temperature last year", ANCHOR)[:2] == ("2016-01-01", "2016-12-31")
