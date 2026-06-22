"""Tests for v2 evaluation upgrade. Run: cd code/v2 && python3.12 -m pytest tests/"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2))
from src import preprocess as pp, anomaly as ad, evaluation as ev

DATA = V2.parent / "data" / "eu_trade_monthly.csv"


def _idx(n=12):
    return pd.date_range("2020-01-01", periods=n, freq="MS")


def test_point_adjust_credits_whole_segment():
    idx = _idx(10)
    truth = pd.Series([False]*10, index=idx); truth.iloc[3:7] = True   # 4-month anomaly
    pred = pd.Series([False]*10, index=idx); pred.iloc[5] = True       # one hit inside
    adj = ev.point_adjust(pred, truth)
    assert adj.iloc[3:7].all()                                         # whole segment credited
    assert not adj.iloc[0:3].any()                                     # outside untouched


def test_point_adjust_no_hit_no_credit():
    idx = _idx(10)
    truth = pd.Series([False]*10, index=idx); truth.iloc[3:7] = True
    pred = pd.Series([False]*10, index=idx)                            # no hits
    assert not ev.point_adjust(pred, truth).iloc[3:7].any()


def test_point_adjusted_recall_ge_strict():
    truth = pd.Series([False]*10, index=_idx(10)); truth.iloc[3:7] = True
    pred = pd.Series([False]*10, index=truth.index); pred.iloc[5] = True
    assert ev.score_point_adjusted(pred, truth).recall >= ad.score(pred, truth).recall


def test_ensemble_voting():
    idx = _idx(5)
    a = pd.Series([True, True, False, False, False], index=idx)
    b = pd.Series([True, False, True, False, False], index=idx)
    c = pd.Series([True, False, False, True, False], index=idx)
    maj = ev.ensemble([a, b, c], min_votes=2)
    assert list(maj) == [True, False, False, False, False]            # only month 0 has >=2 votes


def test_pr_curve_shape_and_auc():
    logged = pp.log_transform(pp.load_trade(DATA))
    truth = ad.label_windows(logged.index, [("2008-09-01","2009-04-01"), ("2020-03-01","2020-08-01")])
    scores = ad.stl_residual_z(logged["DE_EXP"]).abs()
    curve = ev.pr_curve(scores, truth, n=20)
    assert set(curve.columns) >= {"precision", "recall", "f1"}
    assert 0.0 <= ev.pr_auc(curve) <= 1.0
