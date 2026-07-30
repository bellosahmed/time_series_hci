"""Tests for the 'best guess + confirm note' behaviour: when a question under-specifies its
sensor, statistic, or period (or matches two intents), the interface still answers but appends a
short note flagging the assumption so the user can refine it. Fully-specified questions get no note.
"""

import numpy as np
import pandas as pd

from src import interface


def _year_df():
    idx = pd.date_range("2015-01-01", "2016-01-01", freq="h")
    rng = np.random.default_rng(0)
    return pd.DataFrame({"T (degC)": rng.normal(10, 5, len(idx)),
                         "rh (%)": rng.normal(70, 10, len(idx)),
                         "p (mbar)": rng.normal(1013, 8, len(idx))}, index=idx)


def _note(ans):
    return ans.text.split("ℹ️")[1].strip() if "ℹ️" in ans.text else None


def test_no_sensor_named_answers_but_flags_the_temperature_assumption():
    ans = interface.Interface(_year_df()).ask("what was the maximum in 2015")
    assert ans.intent == "max" and ans.value is not None       # still answered
    assert ans.sensors == ["T (degC)"]
    assert "temperature" in _note(ans)
    assert "temperature" in [a for a in ans.assumptions][0]


def test_no_statistic_named_gives_overview_and_says_so():
    ans = interface.Interface(_year_df()).ask("temperature in 2015")
    assert ans.intent == "summary"
    assert "overview" in _note(ans).lower()


def test_no_period_named_flags_whole_record():
    ans = interface.Interface(_year_df()).ask("hottest temperature")
    assert ans.intent == "max"
    assert "whole" in _note(ans).lower() and "2015" in _note(ans)


def test_two_intents_matching_is_flagged():
    ans = interface.Interface(_year_df()).ask("maximum temperature trend in 2015")
    assert _note(ans) is not None
    assert "trend" in _note(ans)


def test_fully_specified_question_gets_no_note():
    ans = interface.Interface(_year_df()).ask("maximum temperature in 2015")
    assert _note(ans) is None
    assert ans.assumptions == []


def test_temperature_implying_word_suppresses_the_sensor_note():
    # "warmest" means temperature even though it isn't a sensor alias -> no 'assuming temperature'.
    ans = interface.Interface(_year_df()).ask("warmest day in 2015")
    note = _note(ans)
    assert note is None or "assuming you mean" not in note
