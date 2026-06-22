"""Tests for v2 multi-country reasoning. Run: cd code/v2 && python3.12 -m pytest tests/"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2))
from src import preprocess as pp
from src import reasoning as rz

DATA = V2.parent / "data" / "eu_trade_monthly.csv"


@pytest.fixture(scope="module")
def cz():
    logged = pp.log_transform(pp.load_trade(DATA))
    return rz.cross_section_z(logged, flow="_EXP")


def test_cross_section_shape(cz):
    assert cz.shape[1] == 27                       # 27 countries
    assert "DE" in cz.columns and "FR" in cz.columns


def test_covid_is_shared_shock(cz):
    r = rz.explain_month(cz, "2020-04-01", thresh=2.0)
    assert r.kind == "shared shock"
    assert r.direction == "down"
    assert r.n_down > r.n_up                        # most countries fell


def test_systemic_timeline_peaks_at_crises(cz):
    counts = rz.systemic_timeline(cz, thresh=2.0)
    # COVID April-2020 should be among the most systemic months
    assert counts.loc["2020-04-01"] >= counts.quantile(0.95)


def test_isolated_months_have_single_country(cz):
    iso = rz.find_isolated_months(cz, thresh=2.0)
    counts = rz.systemic_timeline(cz, thresh=2.0)
    assert (counts.loc[iso] == 1).all()


def test_shock_report_sentence(cz):
    r = rz.explain_month(cz, "2020-04-01")
    s = r.sentence()
    assert "shared shock" in s and "2020-04" in s


def test_normal_month_classified_normal():
    # a flat, noiseless cross-section -> no anomalies
    idx = pd.date_range("2000-01-01", periods=40, freq="MS")
    flat = pd.DataFrame({"AA": np.zeros(40), "BB": np.zeros(40)}, index=idx)
    r = rz.shared_shock(flat, idx[20])
    assert r.kind == "normal"
