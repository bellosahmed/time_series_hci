"""Tests for the labeled-evaluation scorer (`src/evaluation.py`): spike injection, precision/
recall/F1 scoring, point-adjustment, and the seed-aggregated benchmark."""

import numpy as np
import pandas as pd

from src import evaluation


def test_inject_and_score_perfect():
    s = pd.Series(np.zeros(100), index=pd.date_range("2024-01-01", periods=100, freq="h"))
    noisy, labels = evaluation.inject_spikes(s, n=10, magnitude=5.0)
    assert labels.sum() == 10
    # a perfect detector = the labels themselves
    sc = evaluation.score(labels, labels)
    assert sc.precision == 1.0 and sc.recall == 1.0 and sc.f1 == 1.0


def test_score_zero_when_no_overlap():
    idx = pd.date_range("2024-01-01", periods=10, freq="h")
    truth = pd.Series([True] * 5 + [False] * 5, index=idx)
    pred = pd.Series([False] * 5 + [True] * 5, index=idx)
    sc = evaluation.score(pred, truth)
    assert sc.tp == 0 and sc.precision == 0.0 and sc.recall == 0.0


def test_benchmark_aggregates_over_seeds():
    from src import anomaly
    idx = pd.date_range("2024-01-01", periods=500, freq="h")
    clean = pd.Series(np.sin(np.arange(500) / 5.0), index=idx)        # std > 0 so spikes are real
    res = evaluation.benchmark(clean, lambda noisy: anomaly.zscore_anomalies(noisy, threshold=4.0),
                               seeds=range(3), n=10, magnitude=8.0)
    for k in ("precision", "recall", "f1", "pa_recall", "pa_f1"):
        assert 0.0 <= res[k]["mean"] <= 1.0
    assert res["recall"]["mean"] > 0                                  # big spikes get caught


def test_point_adjust_credits_segment():
    idx = pd.date_range("2024-01-01", periods=10, freq="h")
    truth = pd.Series([False, True, True, True, False, False, False, False, False, False], index=idx)
    pred = pd.Series([False, False, True, False, False, False, False, False, False, False], index=idx)
    adj = evaluation.point_adjust(pred, truth)
    assert adj.iloc[1] and adj.iloc[2] and adj.iloc[3]            # whole segment credited
    assert evaluation.score_point_adjusted(pred, truth).recall == 1.0
