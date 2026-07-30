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


# --- Bug C: the planner must not stamp a FALSE scope on a period-less bare-sensor summary ----

@pytest.mark.parametrize("bad_period", ["winter", "summer", "year"])
def test_named_sensor_summary_gets_no_false_scope(monkeypatch, df, bad_period):
    # The small planner routinely echoes the WHOLE dataset span plus a random season word for a
    # period-less bare-sensor question (start=2014-01-01, end=2015-12-31, period="winter"). The
    # window is the entire record, so the transparency line must NOT claim a scope ("in winter")
    # that contradicts the all-years, all-seasons data actually summarised.
    monkeypatch.setattr(interface.llm, "is_available", lambda *a, **k: True)
    monkeypatch.setattr(interface.llm, "plan", lambda *a, **k: {
        "intent": "anomaly", "sensors": [],
        "start": "2014-01-01", "end": "2015-12-31", "period": bad_period})
    iface = interface.Interface(df, use_llm=True)
    iface.history = []
    ans = iface.ask("what is the specific humidity")
    assert ans.intent == "summary"
    assert "sh (g/kg)" in ans.sensors
    assert f" in {bad_period}" not in ans.computed     # no fabricated season/period scope
    # and the honest "whole record" note is still surfaced instead
    assert any("whole" in n for n in ans.assumptions)


def test_whole_record_guard_survives_lone_jan1_bucket(monkeypatch):
    # An hourly resample leaves a lone Jan-1 00:00 bucket in the year after the data ends. The
    # whole-record guard must still recognise the planner's "...to <Dec 31>" span as whole-record
    # (collapsing to the honest note), not fall through to a redundant date-range scope label.
    idx = pd.date_range("2014-01-01", "2016-01-01 00:00", freq="6h")  # trailing 2016-01-01 00:00
    n = len(idx)
    df2 = pd.DataFrame({"T (degC)": [float(i % 40) for i in range(n)],
                        "sh (g/kg)": [5.0 + (i % 10) for i in range(n)]}, index=idx)
    monkeypatch.setattr(interface.llm, "is_available", lambda *a, **k: True)
    monkeypatch.setattr(interface.llm, "plan", lambda *a, **k: {
        "intent": "anomaly", "sensors": [],
        "start": "2014-01-01", "end": "2015-12-31", "period": "winter"})
    iface = interface.Interface(df2, use_llm=True)
    iface.history = []
    ans = iface.ask("what is the specific humidity")
    assert ans.intent == "summary"
    assert " in 2014-01-01 to 2015-12-31" not in ans.computed  # collapsed, not a redundant scope
    assert " in winter" not in ans.computed
    assert any("whole" in n for n in ans.assumptions)


def test_planner_genuine_subwindow_labelled_from_dates(monkeypatch, df):
    # When the planner proposes a genuine sub-window (not the whole record), the label is derived
    # from the resolved dates, never the free-text period word — so it can't disagree with the data.
    monkeypatch.setattr(interface.llm, "is_available", lambda *a, **k: True)
    monkeypatch.setattr(interface.llm, "plan", lambda *a, **k: {
        "intent": None, "sensors": [],
        "start": "2015-06-01", "end": "2015-08-31", "period": "winter"})  # season word is wrong
    iface = interface.Interface(df, use_llm=True)
    iface.history = []
    ans = iface.ask("what is the specific humidity")
    assert "winter" not in ans.computed                # the wrong season word is not trusted
    assert "2015-06-01 to 2015-08-31" in ans.computed  # label reflects the actual window


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
