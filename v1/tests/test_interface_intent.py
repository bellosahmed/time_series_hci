"""Tests for intent/sensor detection: does the interface figure out WHAT is being asked."""

import pandas as pd

from src import interface


def test_detect_intent():
    assert interface.detect_intent("what was the highest temperature?") == "max"
    assert interface.detect_intent("lowest reading") == "min"
    assert interface.detect_intent("average humidity") == "mean"
    assert interface.detect_intent("correlation between temp and humidity") == "correlation"
    assert interface.detect_intent("any unusual readings?") == "anomaly"
    assert interface.detect_intent("hello there") is None
    # word-boundary: short keywords must not match inside larger words
    assert interface.detect_intent("almost done") is None        # not "most" -> max
    assert interface.detect_intent("please stop now") is None    # not "top" -> max


def test_extreme_value_questions_route_to_max_min_not_anomaly():
    # Found by the query-routing coverage benchmark's independent-LLM-judge cross-check
    # (docs/evaluation/06-query-routing-coverage.md, §3): an independent judge read "hottest/
    # lowest ___ ever recorded" as `anomaly` three separate times, conflating "an extreme value"
    # with "an anomalous one". Confirmed this system's own deterministic taxonomy is NOT
    # ambiguous -- "record high"/"record low" are max/min keywords, not anomaly ones, and there
    # is no keyword overlap -- so this is a genuine semantic-interpretation difference between an
    # outside reader and this system's design, not a routing bug. Locked in here so it stays that
    # way; see the doc for the fuller discussion of why the ambiguity is real but not a defect.
    assert interface.detect_intent("hottest it got in summer 2013") == "max"
    assert interface.detect_intent("lowest humidity recorded") == "min"
    assert interface.detect_intent("hottest temperature ever recorded") == "max"
    assert interface.detect_intent("lowest reading ever recorded") == "min"


def test_causal_question_gives_correlation_with_causation_caveat(df):
    ans = interface.Interface(df).ask("does humidity cause the temperature to change?")
    assert ans.intent == "correlation"          # falls back to the honest measure
    assert "causation" in ans.text.lower()      # explicit correlation-not-causation caveat


def test_causal_question_without_pair_explains_no_causation(df):
    ans = interface.Interface(df).ask("what causes temperature spikes?")
    assert any(w in ans.text.lower() for w in ("causation", "correlation"))
    assert "couldn't map" not in ans.text       # not the generic fallback


def test_intent_paraphrase_coverage():
    # Realistic paraphrasings must route correctly — guards against keyword brittleness
    # regressing (each of these was a miss before the synonym lists were widened).
    cases = {
        "how hot did it get?": "max", "what was the record high?": "max", "peak temperature": "max",
        "how cold did it get?": "min", "record low": "min", "coldest day": "min",
        "what is the usual humidity?": "mean", "typical temperature": "mean",
        "how does temperature link to humidity?": "correlation",
        "is temperature associated with density?": "correlation",
        "how are temperature and humidity connected?": "correlation",
        "is it getting warmer?": "trend", "has it been warming over the years?": "trend",
        "anything weird in 2014?": "anomaly", "were there strange readings?": "anomaly",
        "any outliers?": "anomaly", "what is the outlook for next week?": "forecast",
        "how much does it vary?": "spread", "any gaps in the data?": "missing",
    }
    for q, expected in cases.items():
        assert interface.detect_intent(q) == expected, q


def test_resolve_sensors(df):
    cols = list(df.columns)
    assert interface.resolve_sensors("max temperature", cols) == ["T (degC)"]
    assert "rh (%)" in interface.resolve_sensors("humidity vs temperature", cols)


def test_bare_sensor_question_defaults_to_summary_not_fallback(df):
    # No intent keyword at all ("mean"/"max"/etc.) -- previously fell through to the generic
    # "I couldn't map that" message even though the sensor+scope resolved perfectly.
    ans = interface.Interface(df).ask("what was the temperature at 3pm on 1 january 2024?")
    assert "couldn't map" not in ans.text
    assert ans.intent == "summary"
    assert ans.sensors == ["T (degC)"]


def test_bare_sensor_question_with_no_scope_also_gets_a_real_answer(df):
    ans = interface.Interface(df).ask("the humidity")  # no intent keyword, no time scope either
    assert "couldn't map" not in ans.text
    assert ans.intent == "summary"


def test_question_with_neither_intent_nor_sensor_still_gets_the_fallback(df):
    # The fallback must still fire when there's truly nothing to work with -- the new default
    # only kicks in once a sensor was actually named, not for every unroutable question.
    ans = interface.Interface(df).ask("blah blah nonsense")
    assert "couldn't map" in ans.text


def test_ask_max_matches_engine(df):
    ans = interface.Interface(df).ask("what was the max temperature?")
    assert ans.intent == "max"
    assert ans.value["value"] == df["T (degC)"].max()
    assert "Computed" in ans.text


def test_ask_correlation(df):
    ans = interface.Interface(df).ask("correlation between temperature and humidity")
    assert ans.intent == "correlation"
    assert len(ans.sensors) == 2


def test_missing_sensor():
    idx = pd.date_range("2014-01-01", periods=10, freq="h")
    df = pd.DataFrame({"T (degC)": range(10)}, index=idx)         # no humidity column
    ans = interface.Interface(df).ask("what is the average humidity?")
    assert "humidity" in ans.text.lower() and "don't track" in ans.text.lower()
