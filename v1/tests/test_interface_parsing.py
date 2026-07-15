"""Tests for the new time-constraint parsers: relative/ordinal periods and hour-of-day."""

import pandas as pd

from src.interface_parsing import resolve_relative_period

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
