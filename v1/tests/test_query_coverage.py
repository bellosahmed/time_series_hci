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
