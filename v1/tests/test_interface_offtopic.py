"""Off-topic robustness + sensor fidelity on the LLM path.

Two bugs these lock down:
 A. A bare per-sensor question ("what is the specific humidity") must summarise THAT sensor, not be
    reclassified by the planner as the temperature-centric "weather" day-report.
 B. An off-topic message ("what is the name of my cat/dog") must be declined even with the LLM on —
    the LLM must never fabricate a routable weather query from a question with no data signal, whether
    cold or as a follow-up after a valid question.
All run offline by monkeypatching the llm.* calls (no Ollama needed)."""
import pandas as pd
import pytest

from src import interface


@pytest.fixture
def df():
    idx = pd.date_range("2014-01-01", "2015-12-31 23:00", freq="6h")
    n = len(idx)
    return pd.DataFrame(
        {"T (degC)": [float(i % 40) for i in range(n)],
         "rh (%)": [50.0 + (i % 30) for i in range(n)],
         "sh (g/kg)": [5.0 + (i % 10) for i in range(n)]},
        index=idx)


# --- Bug A: named sensor is not downgraded to the "weather" report -------------------------

@pytest.mark.parametrize("bad_intent", ["weather", "missing", "trend", "max"])
def test_named_sensor_not_downgraded_by_planner(monkeypatch, df, bad_intent):
    # A bare per-sensor question resolves the sensor but no statistic; the small planner then guesses
    # a specific intent the user never asked for. That guess must NOT override the honest summary.
    monkeypatch.setattr(interface.llm, "is_available", lambda *a, **k: True)
    monkeypatch.setattr(interface.llm, "plan", lambda *a, **k: {"intent": bad_intent, "sensors": []})
    iface = interface.Interface(df, use_llm=True)
    iface.history = []
    ans = iface.ask("what is the specific humidity")
    assert ans.intent == "summary"
    assert "sh (g/kg)" in ans.sensors


# --- Bug B: off-topic questions are declined even with the LLM on --------------------------

def test_offtopic_question_declined_cold(monkeypatch, df):
    monkeypatch.setattr(interface.llm, "is_available", lambda *a, **k: True)
    # the small model would happily turn nonsense into a routable query; it must not be consulted
    monkeypatch.setattr(interface.llm, "understand", lambda *a, **k: "how much data is missing")
    monkeypatch.setattr(interface.llm, "plan", lambda *a, **k: {"intent": "missing"})
    iface = interface.Interface(df, use_llm=True)
    iface.history = []
    ans = iface.ask("what is the name of my cat")
    assert ans.value is None            # declined, not fabricated
    assert ans.resolved is None         # no interpretation was imposed


def test_offtopic_followup_not_fused_with_prior_context(monkeypatch, df):
    monkeypatch.setattr(interface.llm, "is_available", lambda *a, **k: True)
    # a follow-up rewrite that (wrongly) reuses the previous temperature turn
    monkeypatch.setattr(interface.llm, "rewrite_followup", lambda *a, **k: "max temperature in 2015")
    monkeypatch.setattr(interface.llm, "plan", lambda *a, **k: None)
    monkeypatch.setattr(interface.llm, "understand", lambda *a, **k: None)
    iface = interface.Interface(df, use_llm=True)
    iface.history = [{"question": "max temperature in 2015",
                      "resolved": "max temperature in 2015", "computed": "max T (degC) was 39.00"}]
    ans = iface.ask("what is the name of my dog")
    assert ans.value is None            # not fused into a temperature answer
    assert ans.intent != "max"


def test_offtopic_followup_with_cue_is_declined(monkeypatch, df):
    # "what about my dog" carries a follow-up cue but no data signal -> must NOT reuse prior context.
    monkeypatch.setattr(interface.llm, "is_available", lambda *a, **k: True)
    monkeypatch.setattr(interface.llm, "rewrite_followup", lambda *a, **k: "max temperature in 2015")
    monkeypatch.setattr(interface.llm, "plan", lambda *a, **k: None)
    monkeypatch.setattr(interface.llm, "understand", lambda *a, **k: None)
    iface = interface.Interface(df, use_llm=True)
    iface.history = [{"question": "max temperature in 2015",
                      "resolved": "max temperature in 2015", "computed": "max T (degC) was 39.00"}]
    ans = iface.ask("what about my dog")
    assert ans.value is None


# --- Regression: legitimate follow-ups and sensor questions still work ---------------------

def test_legit_period_followup_still_resolves(monkeypatch, df):
    monkeypatch.setattr(interface.llm, "is_available", lambda *a, **k: True)
    monkeypatch.setattr(interface.llm, "rewrite_followup", lambda *a, **k: "max temperature in 2014")
    iface = interface.Interface(df, use_llm=True)
    iface.history = [{"question": "max temperature in 2015",
                      "resolved": "max temperature in 2015", "computed": "max T (degC) was 39.00"}]
    ans = iface.ask("what about 2014?")   # a real follow-up: has a period + cue
    assert ans.value is not None
    assert "2014" in str(ans.resolved)
