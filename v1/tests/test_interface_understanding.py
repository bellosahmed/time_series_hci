"""Understanding/Clarifier agent: llm.understand proposes a routable rewrite of a malformed query
(fixes spelling/paraphrase); the interface accepts it only if the deterministic parser can route it,
and never lets the LLM produce a number. LLM is mocked; no Ollama needed."""

from src import llm


def test_understand_returns_cleaned_rewrite(monkeypatch):
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: True)
    monkeypatch.setattr(llm, "generate", lambda *a, **k: '"humidity in October 2012"\n')
    out = llm.understand("umidty tocber 2012", ["rh (%)"], ["mean", "max"])
    assert out == "humidity in October 2012"


def test_understand_none_reply_is_none(monkeypatch):
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: True)
    monkeypatch.setattr(llm, "generate", lambda *a, **k: "NONE")
    assert llm.understand("what is the meaning of life", ["rh (%)"], ["mean"]) is None


def test_understand_offline_is_none_without_calling_generate(monkeypatch):
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: False)

    def boom(*a, **k):
        raise AssertionError("generate must not be called when offline")

    monkeypatch.setattr(llm, "generate", boom)
    assert llm.understand("anything", ["rh (%)"], ["mean"]) is None


def test_understand_network_error_is_none(monkeypatch):
    import urllib.error
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: True)

    def boom(*a, **k):
        raise urllib.error.URLError("down")

    monkeypatch.setattr(llm, "generate", boom)
    assert llm.understand("anything", ["rh (%)"], ["mean"]) is None


import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import interface  # noqa: E402


def _df():
    idx = pd.date_range("2015-01-01", "2016-01-01", freq="h")
    rng = np.random.default_rng(0)
    return pd.DataFrame({"T (degC)": rng.normal(10, 8, len(idx)),
                         "rh (%)": rng.normal(70, 12, len(idx))}, index=idx)


def test_unroutable_query_uses_llm_rewrite_and_answers(monkeypatch):
    # "flibber the wobbertron" resolves to no intent and no sensor -> the Understanding agent fires.
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: True)
    monkeypatch.setattr(llm, "understand", lambda *a, **k: "maximum temperature in 2015")
    iface = interface.Interface(_df(), use_llm=True)
    ans = iface.ask("flibber the wobbertron")
    assert ans.intent == "max" and ans.value is not None
    assert "I read this as" in ans.text
    assert ans.resolved == "maximum temperature in 2015"


def test_llm_rewrite_that_is_not_routable_is_rejected(monkeypatch):
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: True)
    monkeypatch.setattr(llm, "understand", lambda *a, **k: "still complete nonsense")
    iface = interface.Interface(_df(), use_llm=True)
    ans = iface.ask("flibber the wobbertron")
    assert ans.value is None                            # no guess; fell back to clarify/rephrase


def test_offline_matches_deterministic_and_does_not_call_understand(monkeypatch):
    called = {"n": 0}

    def spy(*a, **k):
        called["n"] += 1
        return "maximum temperature in 2015"

    monkeypatch.setattr(llm, "is_available", lambda *a, **k: False)
    monkeypatch.setattr(llm, "understand", spy)
    iface = interface.Interface(_df(), use_llm=True)
    ans = iface.ask("flibber the wobbertron")
    assert ans.value is None and called["n"] == 0


def test_already_routable_query_does_not_call_understand(monkeypatch):
    called = {"n": 0}

    def spy(*a, **k):
        called["n"] += 1
        return "maximum temperature in 2015"

    monkeypatch.setattr(llm, "is_available", lambda *a, **k: True)
    monkeypatch.setattr(llm, "understand", spy)
    iface = interface.Interface(_df(), use_llm=True)
    ans = iface.ask("what was the max temperature in 2015")
    assert ans.intent == "max" and called["n"] == 0


def test_understand_exception_is_caught_and_falls_back(monkeypatch, caplog):
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: True)

    def boom(*a, **k):
        raise RuntimeError("ollama exploded")

    monkeypatch.setattr(llm, "understand", boom)
    iface = interface.Interface(_df(), use_llm=True)
    with caplog.at_level("WARNING"):
        ans = iface.ask("flibber the wobbertron")
    assert ans.value is None
    assert any("understanding" in r.getMessage().lower() for r in caplog.records)


def test_grayzone_word_recovered_by_understanding(monkeypatch):
    # "avarge" is a gray-zone near-miss the deterministic guardrail would block; the LLM recovers it.
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: True)
    monkeypatch.setattr(llm, "understand", lambda *a, **k: "average temperature")
    iface = interface.Interface(_df(), use_llm=True)
    ans = iface.ask("avarge tempratur")
    assert ans.intent == "mean" and ans.value is not None
    assert "I read this as" in ans.text


def test_grayzone_without_llm_still_asks(monkeypatch):
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: False)
    iface = interface.Interface(_df(), use_llm=True)
    ans = iface.ask("avarge tempratur")
    assert ans.value is None                            # asks to clarify, as today
