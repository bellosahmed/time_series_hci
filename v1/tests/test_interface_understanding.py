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
