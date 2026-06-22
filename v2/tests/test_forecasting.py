"""Tests for v2 forecasting. Run: cd code/v2 && python3.12 -m pytest tests/"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")
V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2))
from src import preprocess as pp
from src import forecasting as fc

DATA = V2.parent / "data" / "eu_trade_monthly.csv"


@pytest.fixture(scope="module")
def fr():
    df = pp.load_trade(DATA)
    return df["FR_EXP"] + df["FR_IMP"]


def test_forecast_shape_and_band(fr):
    f = fc.forecast(fr, steps=12, log=True)
    assert len(f.mean) == 12
    assert (f.upper >= f.mean).all() and (f.mean >= f.lower).all()   # band brackets the mean


def test_forecast_index_is_future(fr):
    f = fc.forecast(fr, steps=6, log=True)
    assert f.mean.index.min() > fr.index.max()                      # forecasts the future


def test_backtest_predicts_normal_period(fr):
    bt = fc.backtest(fr, horizon=24, log=True)
    # on a calm 24-month window the model should generalise: modest error, decent coverage
    assert bt.mape < 15.0
    assert bt.coverage >= 60.0


def test_backtest_does_not_overfit_synthetic():
    # clean trend+season -> a generalising model has near-zero error on unseen months
    idx = pd.date_range("2000-01-01", periods=180, freq="MS")
    s = pd.Series(1000 + np.arange(180) * 5
                  + 100 * np.sin(np.arange(180) * 2 * np.pi / 12), index=idx)
    bt = fc.backtest(s, horizon=12, log=False)
    assert bt.mape < 5.0


def test_backtest_across_covid_detects_breach(fr):
    btc = fc.backtest_from(fr, cutoff="2019-12-01", horizon=12, log=True)
    # the COVID crash must fall below the lower band on at least one month
    assert (btc.actual < btc.lower).sum() >= 1
