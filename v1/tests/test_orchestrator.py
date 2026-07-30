"""The Planner is conservative and offline-safe: it only splits when the LLM is on AND the question
is not a protected correlation/compare, and always yields >=1 SubTask covering the whole question."""
import pandas as pd

from src import orchestrator as orch


def _df():
    idx = pd.date_range("2013-01-01", periods=48, freq="h")
    return pd.DataFrame({"T (degC)": range(48), "rh (%)": range(48)}, index=idx)


def test_decompose_offline_is_single_subtask():
    o = orch.Orchestrator(_df(), use_llm=False)
    subs = o.decompose("max temperature in 2015 and min humidity in 2013")
    assert len(subs) == 1
    assert subs[0].question == "max temperature in 2015 and min humidity in 2013"


def test_decompose_protects_correlation(monkeypatch):
    # even with the LLM "on", a correlation question is never split
    monkeypatch.setattr(orch.llm, "is_available", lambda *a, **k: True)
    monkeypatch.setattr(orch.llm, "decompose", lambda *a, **k: [("statistician", "a"),
                                                                ("statistician", "b")])
    o = orch.Orchestrator(_df(), use_llm=True)
    subs = o.decompose("correlation between temperature and humidity in 2013")
    assert len(subs) == 1


def test_decompose_uses_llm_split_and_validates_roles(monkeypatch):
    monkeypatch.setattr(orch.llm, "is_available", lambda *a, **k: True)
    monkeypatch.setattr(orch.llm, "decompose", lambda *a, **k: [
        ("statistician", "max temperature in 2015"),
        ("bogus-role", "min humidity in 2013")])
    o = orch.Orchestrator(_df(), use_llm=True)
    subs = o.decompose("max temp 2015 and min humidity 2013")
    assert [s.role for s in subs] == ["statistician", "statistician"]  # bogus -> statistician
    assert len(subs) == 2


def test_dispatch_facts_equal_pandas_and_preserve_order():
    idx = pd.date_range("2015-01-01", periods=72, freq="h")
    df = pd.DataFrame({"T (degC)": [float(i % 24) for i in range(72)]}, index=idx)
    o = orch.Orchestrator(df, use_llm=False)  # offline: agent falls back to interface per sub-task
    subs = [orch.SubTask("statistician", "max temperature"),
            orch.SubTask("statistician", "min temperature")]
    results = o.dispatch(subs)
    assert [r.question for r in results] == ["max temperature", "min temperature"]  # order kept
    # exact max match pandas on the same frame
    assert f"{df['T (degC)'].max():.2f}" in (results[0].computed + results[0].text)


def test_dispatch_role_restricts_tools(monkeypatch):
    df = pd.DataFrame({"T (degC)": [1.0, 2.0], "rh (%)": [3.0, 4.0]},
                      index=pd.date_range("2015-01-01", periods=2, freq="h"))
    seen = {}

    def fake_run(self, q):
        seen["tools"] = self.tool_names
        from src.agent import AgentAnswer
        return AgentAnswer(q, [], "ok", "ok")
    monkeypatch.setattr(orch.agent.Agent, "run", fake_run)
    o = orch.Orchestrator(df, use_llm=False)
    o.dispatch([orch.SubTask("correlator", "correlation between temperature and humidity")])
    assert seen["tools"] == ("correlation",)
