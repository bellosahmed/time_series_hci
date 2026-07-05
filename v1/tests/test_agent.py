"""Tests for the agentic, still-numberless multi-step tool-use layer (`src/agent.py`).

All of these run **offline** (no Ollama): the executor is pure Python, the LLM planning and
synthesis are monkeypatched. The recurring assertion is the project's core guarantee — every
tool result equals the pandas ground truth, and a hallucinated synthesis is rejected.
"""

import numpy as np
import pandas as pd
import pytest

from src import agent, llm


@pytest.fixture
def df():
    # ~2 years of hourly data (covers full summers of 2013 and 2014) with a seasonal cycle.
    idx = pd.date_range("2013-01-01", periods=24 * 760, freq="h")
    rng = np.random.default_rng(0)
    t = 10 + 8 * np.sin(np.arange(len(idx)) * 2 * np.pi / (24 * 365)) + rng.normal(0, 2, len(idx))
    rh = 70 - 0.5 * t + rng.normal(0, 5, len(idx))
    wd = rng.uniform(0, 360, len(idx))
    return pd.DataFrame({"T (degC)": t, "rh (%)": rh, "wd (deg)": wd}, index=idx)


# --- executor is exact (== pandas) --------------------------------------------------------

def test_extremum_matches_pandas(df):
    f = agent.execute_step(df, {"tool": "extremum", "args": {
        "sensor": "T (degC)", "kind": "max", "start": "2013-06-01", "end": "2013-08-31"}})
    assert f.ok
    assert f.value.value == df.loc["2013-06-01":"2013-08-31", "T (degC)"].max()


def test_average_and_correlation_match_pandas(df):
    a = agent.execute_step(df, {"tool": "average", "args": {"sensor": "T (degC)"}})
    assert a.ok and abs(a.value["mean"] - df["T (degC)"].mean()) < 1e-9
    c = agent.execute_step(df, {"tool": "correlation",
                                "args": {"a": "T (degC)", "b": "rh (%)"}})
    assert c.ok and abs(c.value - df["T (degC)"].corr(df["rh (%)"])) < 1e-9


def test_threshold_matches_pandas(df):
    f = agent.execute_step(df, {"tool": "threshold",
                                "args": {"sensor": "T (degC)", "op": ">", "value": 20}})
    assert f.ok and f.value["count"] == int((df["T (degC)"] > 20).sum())


def test_anomaly_count_runs(df):
    f = agent.execute_step(df, {"tool": "anomaly_count", "args": {"sensor": "T (degC)"}})
    assert f.ok and "windows" in f.computed


def test_circular_extremum_uses_circular_stats(df):
    f = agent.execute_step(df, {"tool": "extremum",
                                "args": {"sensor": "wd (deg)", "kind": "max"}})
    assert f.ok and "compass bearing" in f.computed  # not a meaningless linear max


# --- validation rejects bad steps without crashing ----------------------------------------

def test_execute_step_rejects_bad_input(df):
    def bad(step):
        return agent.execute_step(df, step).ok

    assert not bad({"tool": "nope", "args": {}})                                   # unknown tool
    assert not bad({"tool": "extremum", "args": {"sensor": "XXX", "kind": "max"}})  # unknown sensor
    assert not bad({"tool": "extremum", "args": {"sensor": "T (degC)"}})            # missing kind
    assert not bad({"tool": "extremum", "args": {"sensor": "T (degC)", "kind": "up"}})  # bad kind
    assert not bad({"tool": "threshold",
                    "args": {"sensor": "T (degC)", "op": "!!", "value": 1}})        # bad op
    assert not bad({"tool": "threshold",
                    "args": {"sensor": "T (degC)", "op": ">", "value": "abc"}})     # bad value


# --- orchestration: multi-step, all exact -------------------------------------------------

def test_multi_step_plan_all_exact(df):
    ag = agent.Agent(df, use_llm=False)
    plan = [{"tool": "extremum", "args": {"sensor": "T (degC)", "kind": "max",
                                          "start": f"{y}-06-01", "end": f"{y}-08-31"}}
            for y in (2013, 2014)]
    facts = ag.execute_plan(plan)
    assert all(f.ok for f in facts)
    for y, f in zip((2013, 2014), facts, strict=True):
        assert f.value.value == df.loc[f"{y}-06-01":f"{y}-08-31", "T (degC)"].max()


# --- composition: template offline, faithfulness-gated narrative --------------------------

def test_compose_template_when_llm_off(df):
    ag = agent.Agent(df, use_llm=False)
    facts = ag.execute_plan([{"tool": "average", "args": {"sensor": "T (degC)"}}])
    text, used = ag.compose("summary", facts)
    assert used is False and "🔢" in text and facts[0].computed in text


def test_compose_rejects_hallucinated_synthesis(df, monkeypatch):
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: True)
    monkeypatch.setattr(llm, "explain_synthesis",
                        lambda *a, **k: "The peak was a scorching 999.9 degrees.")  # invented
    ag = agent.Agent(df, use_llm=True)
    facts = ag.execute_plan([{"tool": "extremum",
                              "args": {"sensor": "T (degC)", "kind": "max"}}])
    text, used = ag.compose("hottest?", facts)
    assert used is False and "🔢" in text  # hallucination rejected -> template


def test_compose_accepts_faithful_synthesis(df, monkeypatch):
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: True)
    facts = agent.Agent(df, use_llm=False).execute_plan(
        [{"tool": "average", "args": {"sensor": "T (degC)"}}])
    faithful = facts[0].computed  # keeps every number exactly
    monkeypatch.setattr(llm, "explain_synthesis", lambda *a, **k: faithful)
    text, used = agent.Agent(df, use_llm=True).compose("summary", facts)
    assert used is True and text == faithful


# --- run(): plan -> execute -> compose, and single-step fallback --------------------------

def test_run_multistep_end_to_end(df, monkeypatch):
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: True)
    plan = [{"tool": "extremum", "args": {"sensor": "T (degC)", "kind": "max",
                                          "start": f"{y}-06-01", "end": f"{y}-08-31"}}
            for y in (2013, 2014)]
    monkeypatch.setattr(llm, "plan_steps", lambda *a, **k: plan)
    monkeypatch.setattr(llm, "explain_synthesis",
                        lambda q, facts, **k: " ".join(facts))  # faithful passthrough
    res = agent.Agent(df, use_llm=True).run("compare the hottest summers of 2013 and 2014")
    assert len(res.steps) == 2 and all(f.ok for f in res.steps)
    assert res.used_llm is True


def test_run_falls_back_to_single_step(df):
    # use_llm off -> no plan -> delegates to the deterministic single-step interface
    res = agent.Agent(df, use_llm=False).run("what was the hottest temperature?")
    assert res.steps == [] and isinstance(res.text, str) and res.text
