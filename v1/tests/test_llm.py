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
