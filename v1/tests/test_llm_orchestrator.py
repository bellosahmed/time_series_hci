"""decompose/critique are offline-safe: with Ollama down they return the neutral default."""
from src import llm


def test_decompose_returns_none_when_unavailable(monkeypatch):
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: False)
    assert llm.decompose("max temp 2015 and min humidity 2013",
                         ["statistician", "correlator", "anomaly"],
                         ["T (degC)", "rh (%)"]) is None


def test_critique_returns_empty_when_unavailable(monkeypatch):
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: False)
    assert llm.critique("q", [("statistician", "max T was 30.0")]) == []


def test_decompose_parses_pairs(monkeypatch):
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: True)
    monkeypatch.setattr(llm, "_generate_json", lambda *a, **k: {
        "subtasks": [{"role": "statistician", "question": "max temperature in 2015"},
                     {"role": "statistician", "question": "min humidity in 2013"}]})
    out = llm.decompose("max temp 2015 and min humidity 2013",
                        ["statistician", "correlator", "anomaly"], ["T (degC)", "rh (%)"])
    assert out == [("statistician", "max temperature in 2015"),
                   ("statistician", "min humidity in 2013")]
