"""Tests for the NL interface routing and the labeled-evaluation scorer."""

import numpy as np
import pandas as pd
import pytest

from src import evaluation, interface


@pytest.fixture
def df():
    idx = pd.date_range("2024-01-01", periods=48, freq="h")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {"T (degC)": rng.normal(10, 5, 48), "rh (%)": rng.normal(70, 10, 48)},
        index=idx,
    )


def test_followup_is_rewritten_and_routed(df, monkeypatch):
    from src import llm
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: True)
    monkeypatch.setattr(llm, "plan", lambda *a, **k: None)
    monkeypatch.setattr(llm, "explain", lambda *a, **k: "phrased")
    monkeypatch.setattr(llm, "rewrite_followup", lambda q, h, **k: "what was the max temperature?")
    iface = interface.Interface(df, use_llm=True)
    iface.history = [{"question": "average temperature", "computed": "mean ..."}]
    ans = iface.ask("and the peak?")
    assert ans.intent == "max"                       # routed via the rewritten question
    assert ans.value.value == df["T (degC)"].max()   # exact number still from the query engine
    assert ans.question == "and the peak?"           # original wording preserved for display
    assert ans.resolved == "what was the max temperature?"


def test_standalone_question_is_not_rewritten(df, monkeypatch):
    from src import llm
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: True)
    monkeypatch.setattr(llm, "explain", lambda *a, **k: "phrased")
    calls = {"n": 0}

    def spy(q, h, **k):
        calls["n"] += 1
        return "what was the min temperature?"

    monkeypatch.setattr(llm, "rewrite_followup", spy)
    iface = interface.Interface(df, use_llm=True)
    iface.history = [{"question": "average humidity", "computed": "..."}]
    ans = iface.ask("max temperature")
    assert calls["n"] == 0            # intent+sensor => self-contained, rewrite never invoked
    assert ans.intent == "max"


def test_unroutable_rewrite_is_rejected(df, monkeypatch):
    from src import llm
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: True)
    monkeypatch.setattr(llm, "plan", lambda *a, **k: None)
    monkeypatch.setattr(llm, "explain", lambda *a, **k: "phrased")
    monkeypatch.setattr(llm, "rewrite_followup", lambda q, h, **k: "hello there my friend")
    iface = interface.Interface(df, use_llm=True)
    iface.history = [{"question": "max temperature", "computed": "max ..."}]
    ans = iface.ask("and?")
    assert ans.resolved is None   # gibberish rewrite rejected; original kept, not silently applied


def test_history_accumulates_across_turns(df):
    iface = interface.Interface(df)
    iface.ask("max temperature")
    iface.ask("min temperature")
    assert len(iface.history) == 2
    assert iface.history[-1]["question"] == "min temperature"


def test_llm_rewriter_not_called_when_off_but_memory_still_works(df, monkeypatch):
    from src import llm
    calls = {"n": 0}
    monkeypatch.setattr(llm, "rewrite_followup", lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), "x")[1])
    iface = interface.Interface(df, use_llm=False)
    iface.history = [{"question": "max temperature", "resolved": "max temperature", "computed": "..."}]
    ans = iface.ask("what about the min?")
    assert calls["n"] == 0          # the LLM rewriter is never called when the model is off
    assert ans.intent == "min"      # ...but deterministic memory now resolves the follow-up offline
    assert ans.resolved is not None


def test_legend_question_skips_rewrite(df, monkeypatch):
    from src import llm
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: True)
    monkeypatch.setattr(llm, "explain", lambda *a, **k: "phrased")

    def boom(*a, **k):
        raise AssertionError("a legend question must not be rewritten")

    monkeypatch.setattr(llm, "rewrite_followup", boom)
    iface = interface.Interface(df, use_llm=True)
    iface.history = [{"question": "show anomalies", "computed": "..."}]
    assert iface.ask("what is the red dot?").intent == "legend"


def test_resolve_followup_carries_intent_and_swaps_year():
    # an elliptical follow-up naming a new year -> carry the prior intent/sensor, swap the scope
    hist = [{"question": "hottest in 2013", "resolved": "hottest in 2013", "computed": "max ..."}]
    out = interface.resolve_followup("what about 2014?", hist, ["T (degC)", "rh (%)"])
    assert out is not None
    assert interface.detect_intent(out) == "max"          # intent carried from the previous turn
    start, _end, _lbl = interface.resolve_period(out)
    assert start.startswith("2014")                       # scope swapped to the new year


def test_resolve_followup_swaps_sensor_keeping_intent():
    hist = [{"question": "average temperature", "resolved": "average temperature", "computed": "mean ..."}]
    out = interface.resolve_followup("what about humidity?", hist, ["T (degC)", "rh (%)"])
    assert out is not None
    assert interface.detect_intent(out) == "mean"                              # intent kept
    assert "rh (%)" in interface.resolve_sensors(out, ["T (degC)", "rh (%)"])   # sensor swapped


def test_resolve_followup_returns_none_when_nothing_new():
    # nothing to carry a new scope onto -> no deterministic rewrite (leave it to the LLM/fallback)
    hist = [{"question": "max temperature", "resolved": "max temperature", "computed": "max ..."}]
    assert interface.resolve_followup("and?", hist, ["T (degC)", "rh (%)"]) is None


def test_followup_resolved_without_llm(monkeypatch):
    # THE bug fix: conversational memory must work with the LLM OFF (the default offline mode).
    from src import llm
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: False)
    idx = pd.date_range("2013-01-01", periods=2 * 365 * 24, freq="h")
    rng = np.random.default_rng(0)
    two_year = pd.DataFrame({"T (degC)": rng.normal(10, 8, len(idx))}, index=idx)
    iface = interface.Interface(two_year, use_llm=False)
    iface.ask("what was the hottest in 2013?")
    ans = iface.ask("what about 2014?")
    assert ans.intent == "max"                       # intent carried across the turn, no LLM
    assert ans.resolved and "2014" in ans.resolved   # scope swapped to the new year
    assert ans.value.value == two_year[two_year.index.year == 2014]["T (degC)"].max()


def test_is_followup_cue_detects_continuations():
    assert interface.is_followup_cue("and the coldest?")
    assert interface.is_followup_cue("what about 2014?")
    assert not interface.is_followup_cue("what was the hottest in 2013?")


def _three_year_df(seed):
    idx = pd.date_range("2013-01-01", periods=3 * 365 * 24, freq="h")   # 2013–2015
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"T (degC)": rng.normal(10, 8, len(idx))}, index=idx)


def test_followup_carries_scope_across_intent_swap_without_llm(monkeypatch):
    # a cue follow-up that swaps intent but names no new year should inherit the prior scope
    from src import llm
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: False)
    df3 = _three_year_df(1)
    iface = interface.Interface(df3, use_llm=False)
    iface.ask("what was the hottest in 2015?")
    ans = iface.ask("and the coldest?")                  # cue, no new period -> inherit 2015
    assert ans.intent == "min"
    assert ans.resolved and "2015" in ans.resolved
    assert ans.value.value == df3[df3.index.year == 2015]["T (degC)"].min()   # 2015, not whole record


def test_fresh_noncue_question_ignores_prior_scope_without_llm(monkeypatch):
    # no follow-up cue -> a fresh whole-record question; the prior 2015 scope must NOT bleed in
    from src import llm
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: False)
    df3 = _three_year_df(2)
    iface = interface.Interface(df3, use_llm=False)
    iface.ask("what was the hottest in 2015?")
    ans = iface.ask("what was the coldest?")             # no cue -> whole record
    assert ans.value.value == df3["T (degC)"].min()


def test_resolve_legend_query_maps_markers():
    r = interface.resolve_legend_query
    assert r("what is the red dot?") == "anomaly"
    assert r("what does the dotted circle mean?") == "borderline"
    assert r("what's the dashed amber ring?") == "borderline"
    assert r("what is the orange dot?") == "max"
    assert r("what is the blue dot?") == "min"
    assert r("what does the shaded band mean?") == "forecast"
    assert r("explain the legend") == "all"
    # not a legend question -> None (normal routing)
    assert r("what was the max temperature?") is None
    assert r("hottest in 2013") is None


def test_interface_answers_legend_question_not_fallback(df):
    ans = interface.Interface(df).ask("what is the red dot?")
    assert ans.intent == "legend"
    assert "anomaly" in ans.text.lower()
    assert "couldn't map" not in ans.text  # not the default fallback


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


def test_ask_max_matches_engine(df):
    ans = interface.Interface(df).ask("what was the max temperature?")
    assert ans.intent == "max"
    assert ans.value.value == df["T (degC)"].max()
    assert "Computed" in ans.text


def test_ask_correlation(df):
    ans = interface.Interface(df).ask("correlation between temperature and humidity")
    assert ans.intent == "correlation"
    assert len(ans.sensors) == 2


def _big_df(n=600, seed=0):
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"T (degC)": rng.normal(10, 3, n)}, index=idx)


def test_anomaly_answer_exposes_borderline_list():
    ans = interface.Interface(_big_df()).ask("any unusual readings?")
    assert ans.intent == "anomaly"
    assert isinstance(ans.borderline, list)  # near-threshold windows are exposed


def test_chart_renders_borderline_marks(df):
    iface = interface.Interface(df)
    ts = df.index[5]
    ans = interface.Answer("q", "anomaly", "t", value=[], sensors=["T (degC)"])
    ans.borderline = [{"start": ts, "end": ts, "n_points": 1}]
    chart = iface._chart_for(ans)
    assert "borderline" in {m["kind"] for m in chart["marks"]}


def test_llm_error_falls_back_to_template_and_logs(df, monkeypatch, caplog):
    # A dead/erroring LLM must not fail silently (the G2 "silently dead LLM" class of bug):
    # the answer falls back to the template AND the failure is logged, not swallowed.
    from src import llm
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: True)
    monkeypatch.setattr(llm, "plan", lambda *a, **k: None)  # isolate the phrasing path

    def boom(*a, **k):
        raise RuntimeError("ollama exploded")

    monkeypatch.setattr(llm, "explain", boom)
    iface = interface.Interface(df, use_llm=True)
    with caplog.at_level("WARNING"):
        ans = iface.ask("max temperature")
    assert "Computed" in ans.text  # template fallback still produced a correct answer
    assert any("ollama exploded" in r.getMessage() for r in caplog.records)


def test_llm_inventing_a_number_is_rejected_for_template(df, monkeypatch, caplog):
    # If the LLM fabricates a figure absent from the computed facts, the interface must
    # reject its phrasing and fall back to the trustworthy template — numbers stay exact.
    from src import llm
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: True)
    monkeypatch.setattr(llm, "plan", lambda *a, **k: None)
    monkeypatch.setattr(llm, "explain", lambda *a, **k: "The max was 999.99 degrees.")
    iface = interface.Interface(df, use_llm=True)
    with caplog.at_level("WARNING"):
        ans = iface.ask("max temperature")
    assert "999.99" not in ans.text        # fabricated number never reaches the user
    assert "Computed" in ans.text          # template fallback used instead
    assert any("invented" in r.getMessage().lower() for r in caplog.records)


def test_llm_faithful_phrasing_is_kept(df, monkeypatch):
    # When the LLM only rewords the exact facts (no invented numbers), keep its phrasing.
    from src import llm
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: True)
    monkeypatch.setattr(llm, "plan", lambda *a, **k: None)
    monkeypatch.setattr(llm, "explain", lambda *a, **k: "Nicely worded, no figures here.")
    ans = interface.Interface(df, use_llm=True).ask("max temperature")
    assert ans.text == "Nicely worded, no figures here."


def _wd_df(n=120, seed=0):
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    rng = np.random.default_rng(seed)
    ang = (rng.normal(0, 8, n)) % 360   # tight cluster near 0/360
    return pd.DataFrame(
        {"T (degC)": rng.normal(10, 5, n), "wd (deg)": ang}, index=idx
    )


def test_wd_mean_is_circular_not_linear():
    df = _wd_df()
    ans = interface.Interface(df).ask("average wind direction")
    assert ans.sensors == ["wd (deg)"]
    m = ans.value["mean"]
    # circular mean sits near 0/360, never the linear ~180 of a wrapped cluster
    assert min(m, 360 - m) < 30
    assert "prevailing" in ans.computed.lower() or "circular" in ans.computed.lower()


def test_wd_max_does_not_report_linear_extreme():
    df = _wd_df()
    ans = interface.Interface(df).ask("maximum wind direction")
    # a linear max on a wrapped angle (~359.x) is meaningless — must be redirected
    assert "prevailing" in ans.computed.lower() or "circular" in ans.computed.lower()


def test_wd_trend_is_refused_as_circular():
    ans = interface.Interface(_wd_df()).ask("is wind direction rising over time?")
    assert ans.intent == "trend"
    assert any(w in ans.computed.lower() for w in ("circular", "compass", "wrap"))


def test_wd_forecast_is_refused_as_circular():
    ans = interface.Interface(_wd_df()).ask("forecast the wind direction")
    assert ans.intent == "forecast"
    assert any(w in ans.computed.lower() for w in ("circular", "compass", "wrap"))


def test_wd_threshold_flags_circular_caveat():
    ans = interface.Interface(_wd_df()).ask("how often was wind direction above 200?")
    assert ans.intent == "threshold"
    assert any(w in ans.computed.lower() for w in ("circular", "compass", "wrap"))


def test_wd_correlation_is_circular_linear_unsigned():
    df = _wd_df()
    ans = interface.Interface(df).ask("relationship between wind direction and temperature")
    assert ans.intent == "correlation"
    assert 0.0 <= ans.value <= 1.0


def test_resolve_period():
    assert interface.resolve_period("hottest in 2013")[:2] == ("2013-01-01", "2013-12-31")
    assert interface.resolve_period("average in july 2015")[:2] == ("2015-07-01", "2015-07-31")
    assert interface.resolve_period("between 2012 and 2014")[:2] == ("2012-01-01", "2014-12-31")
    assert interface.resolve_period("hottest ever")[0] is None


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
    assert ans.value.value == df.loc["2013", "T (degC)"].max()   # the 2013 max, not global
    assert ans.value.value < df["T (degC)"].max()               # proves it didn't use the whole record
    assert "2013" in ans.text


def test_resolve_day():
    assert interface.resolve_day("tell me the weather in 26 january")[:3] == (1, 26, None)
    assert interface.resolve_day("weather on 5 july 2015")[:3] == (7, 5, 2015)
    assert interface.resolve_day("conditions on 2015-01-26")[:3] == (1, 26, 2015)
    assert interface.resolve_day("what was the average?") is None


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


def test_missing_sensor():
    idx = pd.date_range("2014-01-01", periods=10, freq="h")
    df = pd.DataFrame({"T (degC)": range(10)}, index=idx)         # no humidity column
    ans = interface.Interface(df).ask("what is the average humidity?")
    assert "humidity" in ans.text.lower() and "don't track" in ans.text.lower()


def test_anomaly_answer_has_reasoning():
    idx = pd.date_range("2014-01-01", periods=24 * 40, freq="h")
    rng = np.random.default_rng(0)
    t = pd.Series(10 + 5 * np.sin(2 * np.pi * np.arange(len(idx)) / 24) + rng.normal(0, 0.5, len(idx)),
                  index=idx)
    h = pd.Series(70 - 2 * (t - t.mean()) + rng.normal(0, 1, len(idx)), index=idx)
    t.iloc[500] += 30  # a clear spike -> at least one anomaly window
    df = pd.DataFrame({"T (degC)": t, "rh (%)": h})
    ans = interface.Interface(df).ask("any unusual temperature readings?")
    assert ans.intent == "anomaly"
    assert ans.inferred is not None and "σ" in ans.inferred   # multi-sensor reasoning attached


def test_inject_and_score_perfect():
    s = pd.Series(np.zeros(100), index=pd.date_range("2024-01-01", periods=100, freq="h"))
    noisy, labels = evaluation.inject_spikes(s, n=10, magnitude=5.0)
    assert labels.sum() == 10
    # a perfect detector = the labels themselves
    sc = evaluation.score(labels, labels)
    assert sc.precision == 1.0 and sc.recall == 1.0 and sc.f1 == 1.0


def test_score_zero_when_no_overlap():
    idx = pd.date_range("2024-01-01", periods=10, freq="h")
    truth = pd.Series([True] * 5 + [False] * 5, index=idx)
    pred = pd.Series([False] * 5 + [True] * 5, index=idx)
    sc = evaluation.score(pred, truth)
    assert sc.tp == 0 and sc.precision == 0.0 and sc.recall == 0.0


def test_benchmark_aggregates_over_seeds():
    from src import anomaly
    idx = pd.date_range("2024-01-01", periods=500, freq="h")
    clean = pd.Series(np.sin(np.arange(500) / 5.0), index=idx)        # std > 0 so spikes are real
    res = evaluation.benchmark(clean, lambda noisy: anomaly.zscore_anomalies(noisy, threshold=4.0),
                               seeds=range(3), n=10, magnitude=8.0)
    for k in ("precision", "recall", "f1", "pa_recall", "pa_f1"):
        assert 0.0 <= res[k]["mean"] <= 1.0
    assert res["recall"]["mean"] > 0                                  # big spikes get caught


def test_point_adjust_credits_segment():
    idx = pd.date_range("2024-01-01", periods=10, freq="h")
    truth = pd.Series([False, True, True, True, False, False, False, False, False, False], index=idx)
    pred = pd.Series([False, False, True, False, False, False, False, False, False, False], index=idx)
    adj = evaluation.point_adjust(pred, truth)
    assert adj.iloc[1] and adj.iloc[2] and adj.iloc[3]            # whole segment credited
    assert evaluation.score_point_adjusted(pred, truth).recall == 1.0


JENA_COLUMNS = [
    "p (mbar)", "T (degC)", "Tpot (K)", "Tdew (degC)", "rh (%)", "VPmax (mbar)",
    "VPact (mbar)", "VPdef (mbar)", "sh (g/kg)", "H2OC (mmol/mol)", "rho (g/m**3)",
    "wv (m/s)", "max. wv (m/s)", "wd (deg)",
]


def test_all_14_sensors_have_plain_english_aliases():
    cols = JENA_COLUMNS
    phrases = {
        "pressure": "p (mbar)",
        "temperature": "T (degC)",
        "potential temperature": "Tpot (K)",
        "dew point": "Tdew (degC)",
        "humidity": "rh (%)",
        "saturation vapour pressure": "VPmax (mbar)",
        "actual vapour pressure": "VPact (mbar)",
        "vapour pressure deficit": "VPdef (mbar)",
        "specific humidity": "sh (g/kg)",
        "water vapour concentration": "H2OC (mmol/mol)",
        "air density": "rho (g/m**3)",
        "wind speed": "wv (m/s)",
        "wind gust": "max. wv (m/s)",
        "wind direction": "wd (deg)",
    }
    for phrase, col in phrases.items():
        got = interface.resolve_sensors(phrase, cols)
        assert got and got[0] == col, f"{phrase!r} -> {got}, expected {col} first"


def test_specific_alias_beats_generic():
    cols = JENA_COLUMNS
    assert interface.resolve_sensors("wind direction", cols)[0] == "wd (deg)"      # not wind speed
    assert interface.resolve_sensors("vapour pressure", cols)[0] == "VPact (mbar)"  # not pressure
    assert interface.resolve_sensors("specific humidity", cols)[0] == "sh (g/kg)"   # not humidity
