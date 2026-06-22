"""Tests for v2 interface (LLM-independent). Run: cd code/v2 && python3.12 -m pytest tests/"""
import sys
from pathlib import Path

import pytest

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2))
from src import preprocess as pp
from src import interface as ui
from src import query_engine as qe

DATA = V2.parent / "data" / "eu_trade_monthly.csv"


@pytest.fixture(scope="module")
def df():
    return pp.load_trade(DATA)


def test_resolve_country_name_and_code():
    assert "FR" in ui.resolve_countries("how much does france export")
    assert "DE" in ui.resolve_countries("germany vs DE")


def test_resolve_flow():
    assert ui.resolve_flow("france imports") == "IMP"
    assert ui.resolve_flow("france exports") == "EXP"


def test_intent_detection():
    assert ui.detect_intent("what was the highest export") == "max"
    assert ui.detect_intent("are they correlated") == "correlation"
    assert ui.detect_intent("which countries had a shock") == "shock"


def test_max_answer_matches_engine(df):
    a = ui.ask(df, "France's highest export month?")
    e = qe.get_max(df, "FR", "EXP")
    assert str(int(e.value)) in a.computed.replace(",", "") or f"{e.value:,.0f}" in a.computed
    assert e.month in a.computed
    assert a.error is None


def test_synonyms_resolve_same(df):
    a1 = ui.ask(df, "When did Italy export the most?").computed
    a2 = ui.ask(df, "Italy's peak exports?").computed
    assert a1 == a2


def test_correlation_needs_two_countries(df):
    a = ui.ask(df, "is germany correlated")
    assert a.error is not None                      # only one country -> asks for two


def test_shock_routes_to_reasoning(df):
    a = ui.ask(df, "which countries were abnormal in 2020-04?")
    assert a.intent == "shock"
    assert "shared shock" in a.computed
    assert a.inferred is not None                   # labelled interpretation present


def test_out_of_scope_is_handled(df):
    a = ui.ask(df, "tell me a joke")
    assert a.error is not None                       # no fabricated answer


def test_missing_country_asks(df):
    a = ui.ask(df, "what is the highest?")
    assert a.error is not None


def test_summarise_not_confused_with_rise(df):
    # 'summarise' contains 'rise' (a growth keyword) — must still route to summary
    assert ui.detect_intent("summarise the netherlands") == "summary"
    a = ui.ask(df, "summarise the netherlands")
    assert a.intent == "summary"
    assert "surplus" in a.computed
