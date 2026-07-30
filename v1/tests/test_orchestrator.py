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
