"""Tests for intent/sensor detection: does the interface figure out WHAT is being asked."""

import pandas as pd

from src import interface, rag_system


class _StubRetriever:
    """A domain classifier that always returns the same snippet (or none), for offline tests
    of the semantic fallback -- exercises both branches without loading a real encoder."""

    def __init__(self, snippet):
        self._snip = snippet

    def retrieve(self, query, k=1):
        return [self._snip] if self._snip is not None else []


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


def test_is_causal_query_word_boundary():
    # "cause" is a literal substring of "because" -- a plain `w in question` check used to flag
    # ANY question containing "because" (even bare, with no "of") as causal, hijacking it into
    # the correlation branch instead of answering what was actually asked.
    assert interface.is_causal_query("what causes the temperature to rise")
    assert interface.is_causal_query("it was cold because of low pressure")   # "because of" -> legit
    assert not interface.is_causal_query("I stayed in because it was cold")   # bare "because"
    assert not interface.is_causal_query("the excause was cold weather")     # nonsense substring


def test_bare_because_no_longer_hijacks_an_unrelated_question(df):
    # the concrete bug: a bare "because" (mid-sentence, no causal claim intended) used to hijack
    # ANY question into the correlation branch, silently ignoring what was actually asked.
    ans = interface.Interface(df).ask(
        "I went out because the weather was nice, what was the average temperature?")
    assert ans.intent == "mean"
    assert ans.intent != "correlation"


def test_recommend_intent_beats_keyword_overlap_with_max_and_weather():
    # "recommend" is checked first in INTENTS specifically so these don't fall through to
    # "max" (via "hottest") or "weather" (via "weather") instead.
    assert interface.detect_intent("recommend the hottest days in June") == "recommend"
    assert interface.detect_intent("can you recommend the weather for a specific day") == "recommend"
    assert interface.detect_intent("suggest some dates similar to today") == "recommend"


def test_resolve_direction_maps_adjectives_to_sensor_and_direction():
    assert interface.resolve_direction("recommend hot days") == ("T (degC)", False)
    assert interface.resolve_direction("recommend the coldest week") == ("T (degC)", True)
    assert interface.resolve_direction("recommend the driest month") == ("rh (%)", True)
    assert interface.resolve_direction("recommend a calm day") == ("wv (m/s)", True)
    assert interface.resolve_direction("recommend something") is None


def test_resolve_recurring_month_only_fires_without_a_year():
    assert interface.resolve_recurring_month("hot days in June") == 6
    assert interface.resolve_recurring_month("hot days in June 2015") is None
    assert interface.resolve_recurring_month("recommend something") is None


def test_resolve_generic_direction_covers_sensors_with_no_adjective():
    cols = ["p (mbar)", "Tdew (degC)", "T (degC)"]
    assert interface.resolve_generic_direction("recommend the highest pressure days", cols) \
        == ("p (mbar)", False)
    assert interface.resolve_generic_direction("days with the lowest dew point", cols) \
        == ("Tdew (degC)", True)


def test_resolve_generic_direction_needs_both_a_sensor_and_a_direction_word():
    cols = ["p (mbar)"]
    assert interface.resolve_generic_direction("recommend the highest days", cols) is None  # no sensor
    assert interface.resolve_generic_direction("recommend some pressure days", cols) is None  # no direction


def test_resolve_secondary_period_finds_a_different_month_and_year():
    # "5 September 2012 in November 2012": a genuinely different month than the target's own.
    q = "recommend a day to go out in 5 september 2012 in november 2012"
    assert interface.resolve_secondary_period(q, 9, 2012)[:2] == ("2012-11-01", "2012-11-30")
    # different month AND different year, word order doesn't matter (found by value, not position)
    q2 = "recommend a day like 5 september 2012 in june 2016"
    assert interface.resolve_secondary_period(q2, 9, 2012)[:2] == ("2016-06-01", "2016-06-30")
    # same month restated (the earlier bug report) still works
    q3 = "similar to 6 april 2013 in the month of april 2013"
    assert interface.resolve_secondary_period(q3, 4, 2013)[:2] == ("2013-04-01", "2013-04-30")
    # nothing else named -> no secondary period
    assert interface.resolve_secondary_period("similar to 13 february 2013", 2, 2013) is None


def test_split_target_clause_finds_the_similar_to_target():
    # "January 2010 similar to June 2010": resolve_period alone would pick January by dict
    # order (jan sorts before jun) -- the marker clause makes June unambiguously the target.
    q = "recommend a day to go out in january 2010 similar to june 2010"
    clause = interface.split_target_clause(q)
    assert clause is not None
    assert interface.resolve_period(clause)[:2] == ("2010-06-01", "2010-06-30")


def test_split_target_clause_returns_none_without_a_marker():
    assert interface.split_target_clause("recommend a day like 5 september 2012 in june 2016") is None
    assert interface.split_target_clause("recommend hot days in june") is None


def test_recommend_intent_fires_on_plain_similar_conditions_phrasing():
    # "similar conditions"/"similar weather" (no "to") and "when next"/"next time"/"how soon"
    # weren't recommend keywords at all -- these questions used to fall through to a different
    # intent entirely (e.g. "weather", via the word "weather" alone) with "similar conditions"
    # or "when next" silently doing nothing.
    q = "I liked the weather on 13th february 2013, when next do we get similar conditions?"
    assert interface.detect_intent(q) == "recommend"
    assert interface.detect_intent("recommend a day with similar conditions to today") == "recommend"


def test_is_forward_looking_query():
    assert interface.is_forward_looking_query("when next do we get similar conditions?")
    assert interface.is_forward_looking_query("how soon will conditions be similar to this?")
    assert interface.is_forward_looking_query("next time we get weather like this")
    assert not interface.is_forward_looking_query("recommend other dates similar to 13 february 2013")


def test_resolve_explicit_scope_restriction():
    # a month named twice (once in the date, once as a restated scope) -> restrict
    assert interface.resolve_explicit_scope_restriction(
        "similar to 6 april 2013 in the month of april 2013") is True
    # an explicit restriction phrase, even without repeating the month -> restrict
    assert interface.resolve_explicit_scope_restriction(
        "recommend days similar to 6 april 2013 staying within that month") is True
    # a bare date, named once, no restriction phrase -> search the whole record
    assert interface.resolve_explicit_scope_restriction(
        "recommend other dates similar to 13 february 2013") is False


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


def test_offtopic_question_gets_polite_redirect(df):
    # A question with no semantic relation to the weather data (dense score below threshold)
    # must be redirected politely, never shown a raw error or answered.
    stub = _StubRetriever(rag_system.Snippet("T (degC)", "", 0.05))  # below DOMAIN_THRESHOLD
    ans = interface.Interface(df, domain_retriever=stub).ask("what is my dog's name")
    assert ans.intent is None
    assert ans.sensors == []
    assert "weather dataset" in ans.text        # redirected to the dataset's scope
    assert "rephrase" in ans.text.lower()       # and invited to ask again
    assert "couldn't map" not in ans.text


def test_ontopic_unroutable_question_gets_semantic_suggestion(df):
    # A question that reaches the fallback but is semantically about a sensor (score above
    # threshold) gets a targeted, sensor-specific suggestion -- not a generic error.
    stub = _StubRetriever(rag_system.Snippet("rh (%)", "", 0.40))  # above DOMAIN_THRESHOLD
    ans = interface.Interface(df, domain_retriever=stub).ask("xyzzy foobar plugh")
    assert ans.intent is None
    assert "humidity" in ans.text            # friendly name mapped from rh (%)
    assert "couldn't map" not in ans.text


def test_empty_semantic_match_is_treated_as_offtopic(df):
    # No snippet at all (e.g. TF-IDF fallback returns nothing) -> off-topic redirect.
    ans = interface.Interface(df, domain_retriever=_StubRetriever(None)).ask("blah blah nonsense")
    assert ans.intent is None
    assert "weather dataset" in ans.text


def test_ask_max_matches_engine(df):
    ans = interface.Interface(df).ask("what was the max temperature?")
    assert ans.intent == "max"
    assert ans.value["value"] == df["T (degC)"].max()
    assert "Computed" in ans.text


def test_ask_correlation(df):
    ans = interface.Interface(df).ask("correlation between temperature and humidity")
    assert ans.intent == "correlation"
    assert len(ans.sensors) == 2


def test_correlation_strength_labels_the_magnitude_not_just_the_number():
    assert interface.correlation_strength(0.05) == "very weak"
    assert interface.correlation_strength(-0.25) == "weak"      # sign doesn't affect magnitude
    assert interface.correlation_strength(0.45) == "moderate"
    assert interface.correlation_strength(-0.65) == "strong"
    assert interface.correlation_strength(0.95) == "very strong"
    # boundary values belong to the higher bin
    assert interface.correlation_strength(0.80) == "very strong"
    assert interface.correlation_strength(0.60) == "strong"


def test_ask_correlation_answer_includes_a_strength_and_sign_label(df):
    ans = interface.Interface(df).ask("correlation between temperature and humidity")
    r = ans.value
    expected_strength = interface.correlation_strength(r)
    expected_sign = "positive" if r > 0 else "negative"
    assert f"{expected_strength} {expected_sign} correlation" in ans.computed


def test_is_correlation_breakdown_query():
    assert interface.is_correlation_breakdown_query("why is that correlation only moderate?")
    assert interface.is_correlation_breakdown_query("explain that correlation between T and rh")
    assert interface.is_correlation_breakdown_query("break down the correlation between T and p")
    assert not interface.is_correlation_breakdown_query("what is the correlation between T and rh?")


def test_ask_correlation_breakdown_gives_r_squared_rank_and_seasonal_split(year_df):
    ans = interface.Interface(year_df).ask(
        "why is the correlation between temperature and wind speed only weak?")
    assert ans.intent == "correlation"
    assert "Breakdown:" in ans.computed
    assert "R²=" in ans.computed
    assert "rank correlation=" in ans.computed
    assert "winter" in ans.computed and "summer" in ans.computed


def test_ask_correlation_without_breakdown_trigger_has_no_breakdown_text(year_df):
    # plain correlation questions must be unaffected -- no trigger phrase, no breakdown section.
    ans = interface.Interface(year_df).ask("what is the correlation between temperature and wind speed?")
    assert "Breakdown:" not in ans.computed


def test_missing_sensor():
    idx = pd.date_range("2014-01-01", periods=10, freq="h")
    df = pd.DataFrame({"T (degC)": range(10)}, index=idx)         # no humidity column
    ans = interface.Interface(df).ask("what is the average humidity?")
    assert "humidity" in ans.text.lower() and "don't track" in ans.text.lower()
