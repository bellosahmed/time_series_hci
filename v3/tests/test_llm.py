"""Tests for the faithfulness gate — the check that the LLM invented no number/IP/temperature.
These are offline (no Ollama needed): they exercise the number-comparison logic directly."""
from src import llm


def test_faithfulness_flags_an_invented_count():
    facts = "There were 544 labelled attack packets."
    bad = "There were 999 labelled attack packets."
    assert llm.faithfulness(facts, bad)["invented"] == [999.0]


def test_faithfulness_allows_rounding():
    facts = "The highest reported temperature was 29.28 °C."
    ok = "The peak reading was about 29.3 °C."
    assert llm.faithfulness(facts, ok)["invented"] == []          # rounding is allowed


def test_faithfulness_ok_when_numbers_preserved():
    facts = "The attack traffic comes from 10.6.56.41."
    ok = "All the malicious packets originate from host 10.6.56.41."
    assert llm.faithfulness(facts, ok)["ok"] is True
