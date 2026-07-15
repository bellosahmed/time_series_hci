"""Tests for the anomaly intent specifically: network-level detector selection, the
multi-sensor reasoning attached to an anomaly answer, and the borderline chart marks."""

import numpy as np
import pandas as pd

from src import interface


def _big_df(n=600, seed=0):
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"T (degC)": rng.normal(10, 3, n)}, index=idx)


def test_anomaly_answer_exposes_borderline_list():
    ans = interface.Interface(_big_df()).ask("any unusual readings?")
    assert ans.intent == "anomaly"
    assert isinstance(ans.borderline, list)  # near-threshold windows are exposed


def test_chart_renders_borderline_marks(df):
    iface = interface.Interface(df)
    ts = df.index[5]
    ans = interface.Answer("q", "anomaly", "t", value=[], sensors=["T (degC)"])
    ans.borderline = [{"start": ts, "end": ts, "n_points": 1}]
    chart = iface._chart_for(ans)
    assert "borderline" in {m["kind"] for m in chart["marks"]}


def test_anomaly_answer_has_reasoning():
    idx = pd.date_range("2014-01-01", periods=24 * 40, freq="h")
    rng = np.random.default_rng(0)
    t = pd.Series(10 + 5 * np.sin(2 * np.pi * np.arange(len(idx)) / 24) + rng.normal(0, 0.5, len(idx)),
                  index=idx)
    h = pd.Series(70 - 2 * (t - t.mean()) + rng.normal(0, 1, len(idx)), index=idx)
    t.iloc[500] += 30  # a clear spike -> at least one anomaly window
    df = pd.DataFrame({"T (degC)": t, "rh (%)": h})
    ans = interface.Interface(df).ask("any unusual temperature readings?")
    assert ans.intent == "anomaly"
    assert ans.inferred is not None and "σ" in ans.inferred   # multi-sensor reasoning attached


def _cross_sensor_df(seed=1):
    idx = pd.date_range("2014-01-01", periods=24 * 40, freq="h")
    rng = np.random.default_rng(seed)
    base = 10 + 5 * np.sin(2 * np.pi * np.arange(len(idx)) / 24)
    t = pd.Series(base + rng.normal(0, 0.5, len(idx)), index=idx)
    h = pd.Series(0.5 * base + rng.normal(0, 0.3, len(idx)), index=idx)   # correlated with t
    h.iloc[500] += 8                                                       # cross-sensor inconsistency
    return pd.DataFrame({"T (degC)": t, "rh (%)": h})


def test_default_detector_is_portfolio_and_answer_is_network_level():
    df = _cross_sensor_df()
    iface = interface.Interface(df)
    assert iface.detector == "portfolio"                  # the recommended default
    ans = iface.ask("any unusual readings?")
    assert ans.intent == "anomaly"
    assert ans.detector == "portfolio"
    assert len(ans.network) > 0                            # network-level windows
    assert "portfolio" in ans.computed.lower()            # detector surfaced to the user
    assert "driven by" in ans.text.lower()                # a culprit sensor is attributed
    # the injected rh fault (iloc[500]) is caught, and attributed to rh
    fault_ts = df.index[500]
    assert any(w["start"] <= fault_ts <= w["end"] for w in ans.network)
    from src import anomaly
    assert anomaly.network_culprit(df, fault_ts) == "rh (%)"


def test_ask_detector_selects_isolation_forest():
    df = _cross_sensor_df()
    ans = interface.Interface(df, detector="isolation_forest").ask("any unusual readings?")
    assert ans.intent == "anomaly"
    assert ans.detector == "isolation_forest"
    assert "isolation_forest" in ans.computed.lower()     # the chosen detector is named
    assert isinstance(ans.network, list)                  # network-level windows list


def test_unknown_detector_is_graceful_not_a_crash():
    df = _cross_sensor_df()
    ans = interface.Interface(df, detector="bogus").ask("any unusual readings?")
    assert "bogus" in ans.text.lower()                    # names the bad choice
    assert "portfolio" in ans.text.lower()                # lists valid detectors to pick from
