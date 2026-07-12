"""Tests for the real-label evaluation metrics."""
import numpy as np

from src import evaluation as ev


def test_average_precision_perfect_ranking():
    scores = np.array([0.1, 0.2, 0.9, 0.8])
    labels = np.array([0, 0, 1, 1])
    assert ev.average_precision(scores, labels) == 1.0


def test_best_f1_on_separable_scores():
    scores = np.array([0.0, 0.1, 0.9, 1.0])
    labels = np.array([0, 0, 1, 1])
    r = ev.best_f1(scores, labels)
    assert r["f1"] == 1.0 and r["recall"] == 1.0 and r["precision"] == 1.0


def test_per_type_recall_splits_by_type():
    scores = np.array([0.9, 0.1, 0.9])
    labels = np.array([1, 1, 1])
    atype = np.array(["modification", "duplication", "modification"])
    r = ev.per_type_recall(scores, labels, atype, threshold=0.5)
    assert r["modification"] == 1.0 and r["duplication"] == 0.0   # value-style detector misses timing


def test_point_adjust_flags_whole_detected_segment():
    labels = np.array([0, 1, 1, 0, 1, 0])
    pred = np.array([0, 0, 1, 0, 0, 0])                 # one hit inside the [1,2] segment
    pa = ev.point_adjust(pred, labels)
    assert list(pa) == [0, 1, 1, 0, 0, 0]              # whole segment credited; the missed [4] not


def test_best_f1_point_adjusted_credits_the_segment():
    scores = np.array([0.0, 0.9, 0.1, 0.0])            # only window 1 scores high
    labels = np.array([0, 1, 1, 0])                    # attack segment = windows 1-2
    r = ev.best_f1_point_adjusted(scores, labels)
    assert r["recall"] == 1.0                          # point-adjust credits the whole segment


def test_bootstrap_ci_returns_ordered_bounds():
    rng = np.random.default_rng(0)
    scores = rng.random(200)
    labels = (rng.random(200) < 0.1).astype(int)
    lo, hi = ev.bootstrap_ci(scores, labels, n=200)
    assert 0.0 <= lo <= hi <= 1.0


def test_mcnemar_flags_clear_difference():
    # 20 discordant points needed for an exact-binomial 20/0 split to clear p<0.05
    # (5/0 only reaches p=0.0625 — too little power, not a bug in the test being wrong).
    labels = np.array([1] * 20 + [0] * 20)
    good = labels.copy()                        # correct everywhere
    bad = np.zeros_like(labels)                 # flags nothing -> wrong on every positive
    res = ev.mcnemar_test(good, bad, labels)
    assert res["only_a_correct"] == int(labels.sum())
    assert res["only_b_correct"] == 0
    assert 0.0 <= res["p_value"] <= 1.0
    assert res["p_value"] < 0.05


def test_mcnemar_identical_predictions_not_significant():
    labels = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    res = ev.mcnemar_test(labels.copy(), labels.copy(), labels)
    assert res["n_discordant"] == 0
    assert res["p_value"] == 1.0
