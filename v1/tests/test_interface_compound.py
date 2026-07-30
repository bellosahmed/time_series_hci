"""Compound-question handling: a question that asks two things ("max temp in 2015 AND min humidity
in 2013") is answered as two independent, correctly-scoped sub-answers — not silently reduced to the
first part with a merged scope. The legitimate single-question uses of "and" (correlation between two
sensors, comparison of two periods, "between X and Y" ranges) must NOT be split.
"""

import numpy as np
import pandas as pd

from src import interface


def _df():
    idx = pd.date_range("2012-01-01", "2016-01-01", freq="h")
    rng = np.random.default_rng(0)
    return pd.DataFrame({"T (degC)": rng.normal(10, 8, len(idx)),
                         "rh (%)": rng.normal(70, 12, len(idx)),
                         "p (mbar)": rng.normal(1013, 8, len(idx)),
                         "Tdew (degC)": rng.normal(5, 6, len(idx))}, index=idx)


def _ask(q):
    iface = interface.Interface(_df())
    iface.history = []
    return iface.ask(q)


def test_two_intents_two_sensors_is_split_and_both_answered():
    ans = _ask("what was the max temperature in 2015 and the min humidity in 2013?")
    assert ans.intent == "multi"
    assert isinstance(ans.value, list) and len(ans.value) == 2
    kinds = {v["kind"] for v in ans.value}
    sensors = {v["sensor"] for v in ans.value}
    assert kinds == {"max", "min"}
    assert sensors == {"T (degC)", "rh (%)"}
    # each part keeps its OWN scope (the bug was a merged 2013-2015 window)
    assert ans.value[0]["when"].year == 2015
    assert ans.value[1]["when"].year == 2013


def test_each_part_computed_independently_and_appears_in_text():
    ans = _ask("average humidity in 2015 and standard deviation of dew point in 2014")
    assert ans.intent == "multi"
    assert "rh (%)" in ans.text and "Tdew (degC)" in ans.text
    assert ans.text.strip().startswith("1.") and "\n2." in ans.text


def test_correlation_between_two_sensors_is_not_split():
    ans = _ask("correlation between temperature and humidity")
    assert ans.intent == "correlation"


def test_compare_two_periods_is_not_split():
    ans = _ask("compare 2012 and 2015")
    assert ans.intent == "compare"


def test_between_x_and_y_range_is_not_split():
    ans = _ask("mean pressure between 2012 and 2014")
    assert ans.intent == "mean"


def test_single_question_is_unchanged():
    ans = _ask("what was the max temperature in 2015")
    assert ans.intent == "max"
