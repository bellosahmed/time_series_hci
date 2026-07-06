"""Tests for the fusion of the two channels (the headline experiment)."""
import pandas as pd

from src import fusion


def test_late_fusion_max_is_a_union_of_specialists():
    idx = pd.date_range("2019-10-21", periods=3, freq="5min")
    value = pd.Series([0.9, 0.1, 0.1], index=idx)     # value flags window 0
    flow = pd.Series([0.1, 0.1, 0.9], index=idx)      # flow flags window 2
    fused = fusion.late_fusion(value, flow, rule="max")
    assert fused.idxmin() == idx[1]                   # the window neither flags scores lowest
    assert fused[idx[0]] > fused[idx[1]] and fused[idx[2]] > fused[idx[1]]


def test_early_fusion_returns_aligned_series():
    idx = pd.date_range("2019-10-21", periods=10, freq="5min")
    value_feats = pd.DataFrame({131: [0.0] * 10}, index=idx)
    flow = pd.DataFrame({"msg_rate": [10.0] * 10}, index=idx)
    s = fusion.early_fusion(value_feats, flow)
    assert list(s.index) == list(idx) and len(s) == 10


def test_supervised_scores_are_out_of_fold_probabilities():
    import numpy as np
    idx = pd.date_range("2019-10-21", periods=8, freq="5min")
    value = pd.Series([0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9], index=idx)
    flow = pd.Series([0.1, 0.1, 0.9, 0.9, 0.1, 0.1, 0.9, 0.9], index=idx)
    labels = np.array([0, 1, 1, 1, 0, 1, 1, 1])
    groups = np.array(["a", "a", "a", "a", "b", "b", "b", "b"])   # day-blocked folds
    s = fusion.supervised_scores(value, flow, labels, groups)
    assert len(s) == 8 and s.between(0.0, 1.0).all() and list(s.index) == list(idx)
