"""Tests for v2 anomaly detection. Run: cd code/v2 && python3.12 -m pytest tests/"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2))
from src import preprocess as pp
from src import anomaly as ad

DATA = V2.parent / "data" / "eu_trade_monthly.csv"


@pytest.fixture(scope="module")
def logged():
    return pp.log_transform(pp.load_trade(DATA))


def test_zscore_flags_injected_spike():
    # clean seasonal series + one big spike -> must be flagged
    idx = pd.date_range("2000-01-01", periods=120, freq="MS")
    base = 100 + 10 * np.sin(np.arange(120) * 2 * np.pi / 12)
    s = pd.Series(base, index=idx)
    s.iloc[60] += 80                                  # obvious anomaly
    flags = ad.zscore_anomalies(s, threshold=3.0)
    assert flags.iloc[60]


def test_detectors_return_bool_series(logged):
    f1 = ad.zscore_anomalies(logged["DE_EXP"])
    f2 = ad.seasonal_naive_anomalies(logged["DE_EXP"])
    assert f1.dtype == bool and f2.dtype == bool
    assert len(f1) == len(logged)


def test_isoforest_flags_some_months(logged):
    feat = ad.systemic_features(logged)
    flags = ad.isoforest_anomalies(feat, contamination=0.05)
    assert flags.dtype == bool
    assert 0 < flags.sum() < len(flags)               # flags some but not all


def test_label_windows():
    idx = pd.date_range("2020-01-01", periods=12, freq="MS")
    lab = ad.label_windows(idx, [("2020-03-01", "2020-05-01")])
    assert lab.sum() == 3
    assert lab.loc["2020-04-01"]


def test_score_math():
    idx = pd.date_range("2020-01-01", periods=10, freq="MS")
    truth = pd.Series([False]*10, index=idx); truth.iloc[3:6] = True
    pred = pd.Series([False]*10, index=idx); pred.iloc[3:5] = True; pred.iloc[7] = True
    s = ad.score(pred, truth)
    assert (s.tp, s.fp, s.fn, s.tn) == (2, 1, 1, 6)
    assert abs(s.precision - 2/3) < 1e-9
    assert abs(s.recall - 2/3) < 1e-9


def test_detectors_catch_known_crises(logged):
    # recall on the real crises should be clearly > 0 for the residual z-score
    truth = ad.label_windows(logged.index, [("2008-09-01","2009-04-01"), ("2020-03-01","2020-08-01")])
    flags = ad.zscore_anomalies(logged["DE_EXP"], threshold=2.5)
    assert ad.score(flags, truth).recall > 0
