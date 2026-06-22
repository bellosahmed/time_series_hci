"""Tests for v2 query engine — the truth layer must equal pandas exactly.
Run: cd code/v2 && python3.12 -m pytest tests/"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2))
from src import preprocess as pp
from src import query_engine as qe

DATA = V2.parent / "data" / "eu_trade_monthly.csv"


@pytest.fixture(scope="module")
def df():
    return pp.load_trade(DATA)


def test_get_max_matches_pandas(df):
    e = qe.get_max(df, "FR", "EXP")
    assert e.value == df["FR_EXP"].max()
    assert e.month == str(df["FR_EXP"].idxmax().date())


def test_get_min_matches_pandas(df):
    e = qe.get_min(df, "DE", "IMP")
    assert e.value == df["DE_IMP"].min()


def test_mean_std_match(df):
    mean, std = qe.get_mean_std(df, "IT", "EXP")
    assert mean == df["IT_EXP"].mean()
    assert std == df["IT_EXP"].std()


def test_correlation_matches(df):
    c = qe.get_correlation(df, "DE", "FR", "EXP", growth=True)
    expected = df["DE_EXP"].pct_change().corr(df["FR_EXP"].pct_change())
    assert abs(c - expected) < 1e-12


def test_balance_matches(df):
    b = qe.trade_balance(df, "FR", "2024-03-01")
    assert b == df.loc["2024-03-01", "FR_EXP"] - df.loc["2024-03-01", "FR_IMP"]


def test_growth_matches(df):
    g = qe.growth(df, "PL", "EXP", "2010-01-01", "2024-01-01")
    expected = (df.loc["2024-01-01", "PL_EXP"] / df.loc["2010-01-01", "PL_EXP"] - 1) * 100
    assert abs(g - expected) < 1e-9


def test_worst_drop_is_negative(df):
    e = qe.worst_drop(df, "DE", "EXP")
    assert e.value < 0                                       # it's a fall


def test_biggest_surplus_and_deficit(df):
    m = "2024-03-01"
    bs, bd = qe.biggest_surplus(df, m), qe.biggest_deficit(df, m)
    bals = {c: qe.trade_balance(df, c, m) for c in pp.countries(df)}
    assert bs.value == max(bals.values())
    assert bd.value == min(bals.values())


def test_rank_countries_sorted(df):
    r = qe.rank_countries(df, "EXP", "2024-01-01", n=5)
    assert len(r) == 5
    assert r.is_monotonic_decreasing
    assert r.index[0] == "DE"                                # Germany is the top exporter


def test_country_summary_keys(df):
    s = qe.country_summary(df, "NL")
    assert 0.0 <= s["surplus_share"] <= 1.0
    assert s["months"] == 292
