"""Tests for conversational continuity: follow-up rewriting/resolution, the legend intent,
and the explainability ("understood") trace."""

import numpy as np
import pandas as pd

from src import interface


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
    assert ans.value["value"] == df["T (degC)"].max()   # exact number still from the query engine
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


def test_selfcontained_threshold_not_hijacked_by_history(df, monkeypatch):
    # A threshold question ("how many times ... above 12") is self-contained but its intent is
    # derived from resolve_threshold, not detect_intent -- so _needs_context must still treat it as
    # complete. With prior history present it must NOT be rewritten into the previous turn's intent.
    from src import llm
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: False)   # deterministic follow-up path
    calls = {"n": 0}

    def spy(q, h, cols):
        calls["n"] += 1
        return "mean temperature"

    monkeypatch.setattr(interface, "resolve_followup", spy)
    iface = interface.Interface(df, use_llm=False)
    iface.history = [{"question": "average temperature", "resolved": "average temperature",
                      "computed": "mean T (degC) was 10.00"}]
    ans = iface.ask("how many times was the temperature above 12")
    assert calls["n"] == 0            # self-contained => deterministic follow-up never consulted
    assert ans.intent == "threshold"  # not inherited "mean" from the prior turn
    assert isinstance(ans.value, dict) and "count" in ans.value


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


def test_ask_exposes_an_understanding_trace_for_transparency(year_df):
    iface = interface.Interface(year_df)
    ans = iface.ask("what was the mean temperature last month at 3pm?")
    assert ans.understood is not None
    assert "T (degC)" in ans.understood
    assert "December 2016" in ans.understood
    assert "3pm" in ans.understood


def test_understanding_trace_reflects_only_the_deterministic_parse_not_llm(year_df, monkeypatch):
    # The trace must never trigger a second LLM call -- it's built from the pure parsing
    # functions only, independent of whatever _understand's (possibly LLM-backed) plan step did.
    from src import llm
    calls = {"n": 0}

    def fake_plan(*a, **k):
        calls["n"] += 1
        return None
    monkeypatch.setattr(llm, "plan", fake_plan)
    monkeypatch.setattr(llm, "is_available", lambda: True)
    iface = interface.Interface(year_df, use_llm=True)
    iface.ask("what was the mean temperature last month?")
    assert calls["n"] == 1  # only _understand's own plan() call, not a second one for the trace


def test_understanding_trace_says_something_when_nothing_was_named(year_df):
    ans = interface.Interface(year_df).ask("blah blah nonsense")
    assert ans.understood is not None


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
    assert ans.value["value"] == two_year[two_year.index.year == 2014]["T (degC)"].max()


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
    assert ans.value["value"] == df3[df3.index.year == 2015]["T (degC)"].min()   # 2015, not whole record


def test_fresh_noncue_question_ignores_prior_scope_without_llm(monkeypatch):
    # no follow-up cue -> a fresh whole-record question; the prior 2015 scope must NOT bleed in
    from src import llm
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: False)
    df3 = _three_year_df(2)
    iface = interface.Interface(df3, use_llm=False)
    iface.ask("what was the hottest in 2015?")
    ans = iface.ask("what was the coldest?")             # no cue -> whole record
    assert ans.value["value"] == df3["T (degC)"].min()


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
    # marker cues must match on word boundaries, not as raw substrings: "ring" inside "during"
    # (a common date phrasing) must NOT trip the legend explainer.
    assert r("what was the maximum temperature during July 2015?") is None
    assert r("maximum temperature during in July 2015 ?") is None


def test_interface_answers_legend_question_not_fallback(df):
    ans = interface.Interface(df).ask("what is the red dot?")
    assert ans.intent == "legend"
    assert "anomaly" in ans.text.lower()
    assert "couldn't map" not in ans.text  # not the default fallback
