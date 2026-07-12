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


def test_answer_str_formats_trust_labels():
    computed = ui.Answer("q", "max", "computed fact", inferred="an inference", phrased="a phrasing")
    s = str(computed)
    assert "🔢 computed fact" in s and "🧠 an inference" in s and "💬 a phrasing" in s

    errored = ui.Answer("q", "max", "", error="no country given")
    assert str(errored) == "⚠️ no country given"


def test_mean_answer_matches_engine(df):
    a = ui.ask(df, "average exports for germany")
    m, s = qe.get_mean_std(df, "DE", "EXP")
    assert f"{m:,.0f}" in a.computed and f"{s:,.0f}" in a.computed
    assert a.error is None


def test_mean_missing_country_asks(df):
    a = ui.ask(df, "what is the average?")
    assert a.error is not None


def test_correlation_answer_matches_engine(df):
    a = ui.ask(df, "correlation between germany and france exports")
    c = qe.get_correlation(df, "DE", "FR", "EXP")
    assert f"{c:.2f}" in a.computed
    assert a.error is None


def test_balance_with_country(df):
    a = ui.ask(df, "germany's trade balance")
    b = qe.trade_balance(df, "DE", str(df.index[-1].date()))
    word = "surplus" if b >= 0 else "deficit"
    assert word in a.computed and f"{b:,.0f}" in a.computed


def test_biggest_surplus_without_country(df):
    a = ui.ask(df, "who has the biggest surplus?")
    assert a.intent == "surplus"
    e = qe.biggest_surplus(df, str(df.index[-1].date()))
    assert e.country in a.computed


def test_biggest_deficit_without_country(df):
    a = ui.ask(df, "biggest deficit this month")
    assert a.intent == "deficit"
    e = qe.biggest_deficit(df, str(df.index[-1].date()))
    assert e.country in a.computed


def test_growth_with_two_explicit_months(df):
    a = ui.ask(df, "how much did germany's exports grow from 2010-01 to 2020-01")
    g = qe.growth(df, "DE", "EXP", "2010-01-01", "2020-01-01")
    assert f"{g:+.0f}%" in a.computed


def test_growth_missing_country_asks(df):
    a = ui.ask(df, "how much did exports grow?")
    assert a.error is not None


def test_worst_drop_answer_matches_engine(df):
    a = ui.ask(df, "spain's worst export drop")
    e = qe.worst_drop(df, "ES", "EXP")
    assert f"{e.value:.1f}%" in a.computed and e.month in a.computed


def test_rank_answer_lists_top_countries(df):
    a = ui.ask(df, "top exporters")
    r = qe.rank_countries(df, "EXP", str(df.index[-1].date()), n=5)
    assert all(c in a.computed for c in r.index)


def test_shock_use_llm_calls_explain_and_sets_phrased(df, monkeypatch):
    calls = {}

    def fake_explain(question, computed, inferred=None, model=None, **kw):
        calls["question"] = question
        calls["computed"] = computed
        return "a faithfulness-checked phrasing"

    monkeypatch.setattr(ui.llm, "explain", fake_explain)
    a = ui.ask(df, "which countries were abnormal in 2020-04?", use_llm=True)
    assert a.phrased == "a faithfulness-checked phrasing"
    assert calls["computed"] == a.computed          # the LLM was given the exact computed fact
