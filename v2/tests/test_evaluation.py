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


def test_bootstrap_ci_brackets_perfect_recall():
    idx = _idx(200)
    truth = pd.Series([False]*200, index=idx); truth.iloc[::10] = True
    ci = ev.bootstrap_ci(truth.copy(), truth, metric="recall", n_boot=200, seed=0)
    assert ci["point"] == 1.0
    assert ci["lower"] <= ci["point"] <= ci["upper"]
    assert 0.0 <= ci["lower"] <= ci["upper"] <= 1.0
    assert ci["n"] == 200          # labelled sample size is exposed to contextualise CI width


def test_bootstrap_ci_has_width_for_imperfect_detector():
    idx = _idx(300)
    truth = pd.Series([False]*300, index=idx); truth.iloc[::10] = True
    pred = truth.copy()
    for p in list(range(0, 300, 10))[::2]:   # drop half the positives -> recall ~0.5
        pred.iloc[p] = False
    ci = ev.bootstrap_ci(pred, truth, metric="recall", n_boot=500, seed=1)
    assert 0.3 < ci["point"] < 0.7
    assert ci["upper"] - ci["lower"] > 0.0


def test_mcnemar_flags_clear_difference():
    idx = _idx(200)
    truth = pd.Series([False]*200, index=idx); truth.iloc[::10] = True
    good = truth.copy()                      # correct everywhere
    bad = pd.Series(False, index=idx)        # flags nothing -> wrong on every positive
    res = ev.mcnemar_test(good, bad, truth)
    assert res["only_a_correct"] == int(truth.sum())
    assert res["only_b_correct"] == 0
    assert 0.0 <= res["p_value"] <= 1.0
    assert res["p_value"] < 0.05


def test_mcnemar_identical_detectors_not_significant():
    idx = _idx(100)
    truth = pd.Series([False]*100, index=idx); truth.iloc[::10] = True
    res = ev.mcnemar_test(truth.copy(), truth.copy(), truth)
    assert res["n_discordant"] == 0
    assert res["p_value"] == 1.0
