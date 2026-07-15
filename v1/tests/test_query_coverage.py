"""Tests for the query-routing coverage benchmark (src/query_coverage.py)."""

import numpy as np
import pandas as pd
import pytest

from src import query_coverage as qc
from src.interface import Interface

ALL_14_COLUMNS = [
    "p (mbar)", "T (degC)", "Tpot (K)", "Tdew (degC)", "rh (%)", "VPmax (mbar)",
    "VPact (mbar)", "VPdef (mbar)", "sh (g/kg)", "H2OC (mmol/mol)", "rho (g/m**3)",
    "wv (m/s)", "max. wv (m/s)", "wd (deg)",
]


@pytest.fixture(scope="module")
def coverage_iface():
    """A synthetic-but-realistic 14-sensor, multi-year Interface for routing checks only --
    values are random (routing tests never assert computed numbers), but the index spans enough
    real calendar time (2012-2016) for every date/season/year-range phrase in GROUND_TRUTH to
    resolve against real rows.
    """
    idx = pd.date_range("2012-01-01", "2016-12-31 23:00", freq="h")
    rng = np.random.default_rng(3)
    data = {col: rng.normal(loc=10 * (i + 1), scale=2, size=len(idx))
            for i, col in enumerate(ALL_14_COLUMNS)}
    df = pd.DataFrame(data, index=idx)
    df["wd (deg)"] = rng.uniform(0, 360, len(idx))  # a plausible compass bearing, not a random normal
    return Interface(df)


def test_score_coverage_returns_valid_percentages(coverage_iface):
    result = qc.score_coverage(coverage_iface)
    assert result["n"] == len(qc.GROUND_TRUTH)
    for key in ("intent_pct", "sensors_pct", "scope_pct", "overall_pct"):
        assert 0.0 <= result[key] <= 100.0


def test_score_coverage_scores_above_90pct_on_intent_routing(coverage_iface):
    # Intent routing (the most load-bearing dimension) should be near-perfect on this labelled
    # set -- every case was written to be unambiguous for a human reader.
    result = qc.score_coverage(coverage_iface)
    assert result["intent_pct"] >= 90.0, result["failures"]


def test_score_coverage_unroutable_cases_stay_unrouted(coverage_iface):
    # The "summary" fallback added this session must not fire for genuinely empty questions.
    unroutable = [c for c in qc.GROUND_TRUTH if c.intent is None]
    assert len(unroutable) >= 2
    for case in unroutable:
        ans = coverage_iface.ask(case.question)
        assert ans.intent is None, f"{case.question!r} should stay unroutable, got {ans.intent!r}"


def test_held_out_generalisation_set_scores_above_90pct(coverage_iface):
    # HELD_OUT_GROUND_TRUTH was authored fresh, after the 4 gaps in GROUND_TRUTH were fixed, to
    # test whether those fixes generalise to new phrasing rather than just closing the specific
    # holes found. This must score well on its own -- it is not the set the fixes were tuned on.
    result = qc.score_coverage(coverage_iface, qc.HELD_OUT_GROUND_TRUTH)
    assert result["n"] == len(qc.HELD_OUT_GROUND_TRUTH)
    assert result["overall_pct"] >= 90.0, result["failures"]


def test_export_and_score_blind_worksheet_round_trips(tmp_path):
    cases = qc.GROUND_TRUTH[:3]
    csv_path = tmp_path / "worksheet.csv"
    qc.export_blind_worksheet(cases, str(csv_path))
    rows = csv_path.read_text().splitlines()
    assert rows[0] == "id,question,your_intent,your_sensors,your_has_scope"
    assert len(rows) == 1 + len(cases)              # header + one row per question
    for case in cases:
        assert case.question in csv_path.read_text()
    # no expected answers leaked into the worksheet a human is meant to fill in blind
    for case in cases:
        if case.intent:
            assert f",{case.intent}," not in csv_path.read_text()

    # a "second rater" who happens to agree with every original label perfectly
    import csv as csv_module
    with open(csv_path, "w", newline="") as fh:
        writer = csv_module.writer(fh)
        writer.writerow(["id", "question", "your_intent", "your_sensors", "your_has_scope"])
        for i, case in enumerate(cases):
            writer.writerow([i, case.question, case.intent or "",
                             ",".join(sorted(case.sensors)), "yes" if case.has_scope else "no"])
    result = qc.score_blind_labels(cases, str(csv_path))
    assert result["n"] == len(cases)
    assert result["intent_agreement_pct"] == 100.0
    assert result["sensors_agreement_pct"] == 100.0
    assert result["scope_agreement_pct"] == 100.0
    assert result["disagreements"] == []


def test_score_blind_labels_detects_a_real_disagreement(tmp_path):
    cases = [qc.Case("what was the max temperature in 2013?", "max", frozenset({"T (degC)"}), True)]
    csv_path = tmp_path / "worksheet.csv"
    import csv as csv_module
    with open(csv_path, "w", newline="") as fh:
        writer = csv_module.writer(fh)
        writer.writerow(["id", "question", "your_intent", "your_sensors", "your_has_scope"])
        writer.writerow([0, cases[0].question, "mean", "T (degC)", "yes"])  # disagrees: mean != max
    result = qc.score_blind_labels(cases, str(csv_path))
    assert result["intent_agreement_pct"] == 0.0
    assert result["sensors_agreement_pct"] == 100.0
    assert len(result["disagreements"]) == 1


def test_score_against_llm_judge_unavailable_without_ollama(monkeypatch):
    from src import llm
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: False)
    result = qc.score_against_llm_judge(qc.GROUND_TRUTH[:2])
    assert result == {"available": False}


def test_independent_intent_judgment_parses_a_real_reply(monkeypatch):
    from src import llm
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: True)
    monkeypatch.setattr(llm, "generate", lambda *a, **k: (
        '{"intent": "max", "sensors": ["T (degC)"], "has_scope": true}'))
    judgment = llm.independent_intent_judgment(
        "what was the max temperature in 2013?", ["max", "min"], ["T (degC)", "rh (%)"])
    assert judgment == {"intent": "max", "sensors": ["T (degC)"], "has_scope": True}


def test_independent_intent_judgment_rejects_out_of_vocabulary_answers(monkeypatch):
    from src import llm
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: True)
    monkeypatch.setattr(llm, "generate", lambda *a, **k: (
        '{"intent": "not_a_real_intent", "sensors": ["not_a_real_sensor"], "has_scope": false}'))
    judgment = llm.independent_intent_judgment(
        "some question", ["max", "min"], ["T (degC)"])
    assert judgment == {"intent": None, "sensors": [], "has_scope": False}
