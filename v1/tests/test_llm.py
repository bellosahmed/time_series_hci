"""Tests for the Ollama LLM module.

These don't require Ollama to be running: they check graceful behaviour when it's
absent. The actual generation is only exercised if a local server is up.
"""

import pytest

from src import llm


def test_is_available_returns_bool():
    assert isinstance(llm.is_available(), bool)


def test_explain_raises_without_server():
    if llm.is_available():
        pytest.skip("Ollama is running; skip the unavailable-path test.")
    with pytest.raises(RuntimeError):
        llm.explain("max temp?", computed="max T was 37.0 at 2015-07-04.")


def test_plan_parses_and_validates(monkeypatch):
    """plan() should parse JSON, keep allowed fields, and drop invalid ones — no server needed."""
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: True)
    monkeypatch.setattr(llm, "generate", lambda *a, **k: (
        '{"intent":"max","sensors":["T (degC)","bogus"],"start":"2013-06-01",'
        '"end":"2013-08-31","period":"summer 2013","op":">","value":30}'))
    p = llm.plan("hottest above 30 in summer 2013", ["max", "min", "mean"], ["T (degC)", "rh (%)"])
    assert p["intent"] == "max"
    assert p["sensors"] == ["T (degC)"]          # "bogus" dropped (not an allowed column)
    assert p["start"] == "2013-06-01" and p["op"] == ">" and p["value"] == 30.0


def test_faithfulness_flags_changed_numbers():
    facts = "max T (degC) was 37.04 on 4 Jul 2015 16:00."
    assert llm.faithfulness(facts, "The peak was 37.04°C, back in July 2015.")["ok"]   # time reworded, value kept
    assert llm.faithfulness(facts, "The peak was about 37.0°C.")["invented"] == []     # rounding allowed
    bad = llm.faithfulness(facts, "The peak was 38.10°C.")
    assert not bad["ok"] and 38.1 in bad["invented"] and 37.04 in bad["dropped"]


def test_plan_rejects_garbage(monkeypatch):
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: True)
    monkeypatch.setattr(llm, "generate", lambda *a, **k: "sorry, I cannot help")
    assert llm.plan("hello", ["max"], ["T (degC)"]) is None


def test_live_generation_if_available():
    if not llm.is_available():
        pytest.skip("Ollama not running.")
    models = llm.list_models()
    if not models:
        pytest.skip("Ollama running but no models pulled.")
    out = llm.explain("What was the hottest temperature?",
                      computed="max T (degC) was 37.04 at 2015-07-04 16:00.",
                      model=models[0])
    assert isinstance(out, str) and len(out) > 0
