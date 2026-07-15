"""Tests for the Ollama LLM module.

These don't require Ollama to be running: they check graceful behaviour when it's
absent. The actual generation is only exercised if a local server is up.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src import interface, llm


def test_generate_json_parses_a_wrapped_json_block(monkeypatch):
    # Models sometimes wrap JSON in prose despite instructions -- the {...} block must still be found.
    monkeypatch.setattr(llm, "generate", lambda *a, **k: 'Sure, here it is: {"a": 1, "b": [2, 3]} thanks')
    assert llm._generate_json("p", "sys", "model", 5.0) == {"a": 1, "b": [2, 3]}


def test_generate_json_none_on_no_braces(monkeypatch):
    monkeypatch.setattr(llm, "generate", lambda *a, **k: "no json here at all")
    assert llm._generate_json("p", "sys", "model", 5.0) is None


def test_generate_json_none_on_malformed_json(monkeypatch):
    monkeypatch.setattr(llm, "generate", lambda *a, **k: "{not: valid json}")
    assert llm._generate_json("p", "sys", "model", 5.0) is None


def test_generate_json_none_on_network_error(monkeypatch):
    import urllib.error

    def boom(*a, **k):
        raise urllib.error.URLError("connection refused")
    monkeypatch.setattr(llm, "generate", boom)
    assert llm._generate_json("p", "sys", "model", 5.0) is None


def test_generate_sets_seed_and_temperature(monkeypatch):
    captured = {}

    class FakeResp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"response":"ok"}'

    def fake_urlopen(req, timeout=None):
        captured["data"] = req.data
        return FakeResp()

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    out = llm.generate("hi", seed=7, temperature=0.0)
    payload = json.loads(captured["data"])
    assert payload["options"]["seed"] == 7          # reproducible generation
    assert payload["options"]["temperature"] == 0.0
    assert out == "ok"


def test_is_available_returns_bool():
    assert isinstance(llm.is_available(), bool)


def test_faithfulness_flags_invented_integer_count():
    facts = "66 anomalous readings in 24 windows."
    good = "There were 66 unusual readings across 24 windows."
    bad = "There were 66 unusual readings across 87 windows."   # 87 is hallucinated
    assert llm.faithfulness(facts, good)["invented"] == []
    assert 87 in llm.faithfulness(facts, bad)["invented"]


def test_faithfulness_flags_wrong_year():
    facts = "max T (degC) was 37.04 on 4 Jul 2015 16:00."
    assert 2016 in llm.faithfulness(facts, "The record was 37.04 in 2016.")["invented"]


def test_faithfulness_allows_clock_reformatting():
    facts = "max T (degC) was 37.04 on 4 Jul 2015 16:00."
    exp = "The hottest was 37.04 degrees, on 4 July 2015 at 4pm."
    assert llm.faithfulness(facts, exp)["invented"] == []   # 16:00 -> 4pm must not be flagged


def test_faithfulness_ablation_shows_the_gate_actually_does_something(monkeypatch):
    # A pathological model that ALWAYS invents a number, regardless of the real facts: the raw
    # (ungated) rate must come back 100%, but Interface's gate must still show 0% -- this is the
    # comparison the gate's "0 invented numbers" claim depends on; without it, that claim is
    # unfalsifiable (a model that already never hallucinates would show the same "0" trivially).
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: True)
    monkeypatch.setattr(llm, "generate", lambda *a, **k: "The answer is definitely 999.99 degrees.")
    idx = pd.date_range("2024-01-01", periods=48, freq="h")
    df = pd.DataFrame({"T (degC)": np.random.default_rng(0).normal(10, 5, 48)}, index=idx)
    iface_llm = interface.Interface(df, use_llm=True, model="m")
    result = llm.faithfulness_ablation(iface_llm, ["max temperature"], model="m", n_repeats=3)
    assert result["available"] is True
    assert result["raw_invention_rate"] == 1.0     # every raw call invented 999.99
    assert result["gated_invention_rate"] == 0.0    # the gate caught every single one
    assert result["n_total_runs"] == 3
    assert result["per_question"][0]["raw_invented"] == 3
    assert result["per_question"][0]["gated_invented"] == 0


def test_faithfulness_ablation_unavailable_without_ollama(monkeypatch):
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: False)
    idx = pd.date_range("2024-01-01", periods=48, freq="h")
    df = pd.DataFrame({"T (degC)": np.random.default_rng(0).normal(10, 5, 48)}, index=idx)
    iface_llm = interface.Interface(df, use_llm=True, model="m")
    result = llm.faithfulness_ablation(iface_llm, ["max temperature"], model="m")
    assert result == {"available": False}


def test_rewrite_followup_none_without_history(monkeypatch):
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: True)
    assert llm.rewrite_followup("what about 2014?", [], model="m") is None


def test_rewrite_followup_uses_generate(monkeypatch):
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: True)
    monkeypatch.setattr(llm, "generate", lambda *a, **k: "hottest in 2014?\n")
    hist = [{"question": "hottest in 2013?", "computed": "max T (degC) was 34.0 on ..."}]
    assert llm.rewrite_followup("what about 2014?", hist, model="m") == "hottest in 2014?"


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


def test_pick_model_prefers_pulled_then_default(monkeypatch):
    # explicit request always wins
    assert llm.pick_model(requested="my-model") == "my-model"
    # otherwise the first *pulled* preferred model is chosen (larger first)
    monkeypatch.setattr(llm, "list_models", lambda: ["llama3.2:3b", "llama3.1:8b"])
    assert llm.pick_model() == "llama3.1:8b"
    # none of the preferred models pulled -> fall back to DEFAULT_MODEL
    monkeypatch.setattr(llm, "list_models", lambda: ["something-else"])
    assert llm.pick_model() == llm.DEFAULT_MODEL


def test_plan_steps_parses_and_shape_validates(monkeypatch):
    """plan_steps() parses the JSON step list and normalises each step's shape — no server."""
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: True)
    monkeypatch.setattr(llm, "generate", lambda *a, **k: (
        '{"steps":[{"tool":"extremum","args":{"sensor":"T (degC)","kind":"max",'
        '"start":"2013-06-01","end":"2013-08-31"}},{"tool":"average","args":null},'
        '{"bad":"no tool key"}]}'))
    steps = llm.plan_steps("compare summers", [{"name": "extremum", "description": "",
                                                "params": ("sensor",)}], ["T (degC)"])
    assert [s["tool"] for s in steps] == ["extremum", "average"]  # malformed 3rd step dropped
    assert steps[1]["args"] == {}                                 # null args -> {}


def test_plan_steps_none_on_garbage(monkeypatch):
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: True)
    monkeypatch.setattr(llm, "generate", lambda *a, **k: "no json here")
    assert llm.plan_steps("hi", [], ["T (degC)"]) is None


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


def test_default_model_is_actually_pulled():
    """Regression guard: when Ollama is up, DEFAULT_MODEL must resolve (no 404).

    A tag mismatch (e.g. DEFAULT_MODEL='llama3.2' when only 'llama3.2:3b' is pulled) made
    `generate` raise HTTP 404, which the interface silently swallowed and fell back to the
    template — so use_llm=True looked identical to use_llm=False. This test catches that.
    """
    if not llm.is_available():
        pytest.skip("Ollama not running; nothing to check.")
    assert llm.DEFAULT_MODEL in llm.list_models()
    out = llm.generate("Reply with the single word: ok", model=llm.DEFAULT_MODEL)
    assert isinstance(out, str) and out.strip() != ""


def test_llm_explain_path_differs_from_template_when_available():
    """When Ollama is up, the local-LLM explanation must actually run (not fall back).

    Guards the same silent-fallback regression at the interface level: the LLM answer
    should be real prose, not the emoji-structured template string.
    """
    if not llm.is_available():
        pytest.skip("Ollama not running; nothing to check.")
    out = llm.explain(
        "what was the hottest temperature?",
        computed="max T (degC) was 37.04 on 4 Jul 2015 16:00.",
    )
    assert isinstance(out, str) and out.strip() != ""
    assert "🔢" not in out  # not the template format
    fa = llm.faithfulness(facts="max T (degC) was 37.04 on 4 Jul 2015 16:00.", explanation=out)
    assert fa["invented"] == []  # the model must not invent numbers
