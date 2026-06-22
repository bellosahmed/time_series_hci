"""Tests for v2 preprocessing. Run: cd code/v2 && python3.12 -m pytest tests/"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2))
from src import preprocess as pp

DATA = V2.parent / "data" / "eu_trade_monthly.csv"


@pytest.fixture(scope="module")
def df():
    return pp.load_trade(DATA)


def test_loads_monthly_sorted(df):
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing
    assert df.shape == (292, 54)


def test_dataset_is_clean(df):
    r = pp.validate(df)
    assert r.is_clean
    assert r.missing_cells == 0 and r.missing_months == 0 and r.nonpositive == 0


def test_countries_and_flows(df):
    assert len(pp.countries(df)) == 27
    assert len(pp.flow_columns(df)) == 54
    assert "DE" in pp.countries(df)


def test_balance_equals_exp_minus_imp(df):
    bal = pp.trade_balance(df)
    expected = df["FR_EXP"] - df["FR_IMP"]
    pd.testing.assert_series_equal(bal["FR_BAL"], expected, check_names=False)


def test_total_equals_exp_plus_imp(df):
    tot = pp.total_trade(df)
    expected = df["DE_EXP"] + df["DE_IMP"]
    pd.testing.assert_series_equal(tot["DE_TOT"], expected, check_names=False)


def test_log_transform_inverts(df):
    logged = pp.log_transform(df)
    # exp(log(x)) == x for the flow columns
    back = np.exp(logged["IT_EXP"])
    pd.testing.assert_series_equal(back, df["IT_EXP"], check_names=False)


def test_add_features_shape(df):
    feat = pp.add_features(df)
    assert feat.shape[1] == 54 + 27 + 27  # flows + balance + total


def test_pipeline_options():
    plain = pp.preprocess(DATA)
    full = pp.preprocess(DATA, log=True, features=True)
    assert plain.shape[1] == 54
    assert full.shape[1] == 108
