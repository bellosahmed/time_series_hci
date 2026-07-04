"""Tests for the NL interface routing and the labeled-evaluation scorer."""

import numpy as np
import pandas as pd
import pytest

from src import interface, evaluation


@pytest.fixture
def df():
    idx = pd.date_range("2024-01-01", periods=48, freq="h")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {"T (degC)": rng.normal(10, 5, 48), "rh (%)": rng.normal(70, 10, 48)},
        index=idx,
    )


def test_detect_intent():
    assert interface.detect_intent("what was the highest temperature?") == "max"
    assert interface.detect_intent("lowest reading") == "min"
    assert interface.detect_intent("average humidity") == "mean"
    assert interface.detect_intent("correlation between temp and humidity") == "correlation"
    assert interface.detect_intent("any unusual readings?") == "anomaly"
    assert interface.detect_intent("hello there") is None
    # word-boundary: short keywords must not match inside larger words
    assert interface.detect_intent("almost done") is None        # not "most" -> max
    assert interface.detect_intent("please stop now") is None    # not "top" -> max


def test_resolve_sensors(df):
    cols = list(df.columns)
    assert interface.resolve_sensors("max temperature", cols) == ["T (degC)"]
    assert "rh (%)" in interface.resolve_sensors("humidity vs temperature", cols)


def test_ask_max_matches_engine(df):
    ans = interface.Interface(df).ask("what was the max temperature?")
    assert ans.intent == "max"
    assert ans.value.value == df["T (degC)"].max()
    assert "Computed" in ans.text


def test_ask_correlation(df):
    ans = interface.Interface(df).ask("correlation between temperature and humidity")
    assert ans.intent == "correlation"
    assert len(ans.sensors) == 2


def test_llm_error_falls_back_to_template_and_logs(df, monkeypatch, caplog):
    # A dead/erroring LLM must not fail silently (the G2 "silently dead LLM" class of bug):
    # the answer falls back to the template AND the failure is logged, not swallowed.
    from src import llm
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: True)
    monkeypatch.setattr(llm, "plan", lambda *a, **k: None)  # isolate the phrasing path

    def boom(*a, **k):
        raise RuntimeError("ollama exploded")

    monkeypatch.setattr(llm, "explain", boom)
    iface = interface.Interface(df, use_llm=True)
    with caplog.at_level("WARNING"):
        ans = iface.ask("max temperature")
    assert "Computed" in ans.text  # template fallback still produced a correct answer
    assert any("ollama exploded" in r.getMessage() for r in caplog.records)


def _wd_df(n=120, seed=0):
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    rng = np.random.default_rng(seed)
    ang = (rng.normal(0, 8, n)) % 360   # tight cluster near 0/360
    return pd.DataFrame(
        {"T (degC)": rng.normal(10, 5, n), "wd (deg)": ang}, index=idx
    )


def test_wd_mean_is_circular_not_linear():
    df = _wd_df()
    ans = interface.Interface(df).ask("average wind direction")
    assert ans.sensors == ["wd (deg)"]
    m = ans.value["mean"]
    # circular mean sits near 0/360, never the linear ~180 of a wrapped cluster
    assert min(m, 360 - m) < 30
    assert "prevailing" in ans.computed.lower() or "circular" in ans.computed.lower()


def test_wd_max_does_not_report_linear_extreme():
    df = _wd_df()
    ans = interface.Interface(df).ask("maximum wind direction")
    # a linear max on a wrapped angle (~359.x) is meaningless — must be redirected
    assert "prevailing" in ans.computed.lower() or "circular" in ans.computed.lower()


def test_wd_correlation_is_circular_linear_unsigned():
    df = _wd_df()
    ans = interface.Interface(df).ask("relationship between wind direction and temperature")
    assert ans.intent == "correlation"
    assert 0.0 <= ans.value <= 1.0


def test_resolve_period():
    assert interface.resolve_period("hottest in 2013")[:2] == ("2013-01-01", "2013-12-31")
    assert interface.resolve_period("average in july 2015")[:2] == ("2015-07-01", "2015-07-31")
    assert interface.resolve_period("between 2012 and 2014")[:2] == ("2012-01-01", "2014-12-31")
    assert interface.resolve_period("hottest ever")[0] is None


def test_resolve_period_seasons():
    # summer = Jun–Aug of the year
    assert interface.resolve_period("hottest in summer 2012")[:2] == ("2012-06-01", "2012-08-31")
    # winter rolls back to the previous December
    start, end, label = interface.resolve_period("coldest in winter 2013")
    assert start == "2012-12-01" and end == "2013-02-28" and "winter" in label


def test_resolve_threshold():
    assert interface.resolve_threshold("how often above 30 degrees?") == (">", 30.0)
    assert interface.resolve_threshold("colder than -5 in 2014") == ("<", -5.0)
    assert interface.resolve_threshold("what is the average?") is None  # no comparison


def test_ask_scopes_to_year():
    # strictly increasing temp across 2013 -> 2014; the all-time max is in 2014.
    idx = pd.date_range("2013-06-01", periods=400, freq="D")
    df = pd.DataFrame({"T (degC)": np.arange(400, dtype=float)}, index=idx)
    ans = interface.Interface(df).ask("what was the hottest temperature in 2013?")
    assert ans.intent == "max"
    assert ans.value.value == df.loc["2013", "T (degC)"].max()   # the 2013 max, not global
    assert ans.value.value < df["T (degC)"].max()               # proves it didn't use the whole record
    assert "2013" in ans.text


def test_resolve_day():
    assert interface.resolve_day("tell me the weather in 26 january")[:3] == (1, 26, None)
    assert interface.resolve_day("weather on 5 july 2015")[:3] == (7, 5, 2015)
    assert interface.resolve_day("conditions on 2015-01-26")[:3] == (1, 26, 2015)
    assert interface.resolve_day("what was the average?") is None


def test_ask_weather_report():
    idx = pd.date_range("2015-01-25", periods=72, freq="h")     # 3 days incl 26 Jan
    df = pd.DataFrame({"T (degC)": np.linspace(-5, 5, 72),
                       "rh (%)": np.linspace(60, 90, 72)}, index=idx)
    ans = interface.Interface(df).ask("tell me the weather on 26 january 2015")
    assert ans.intent == "weather"
    assert "T (degC)" in ans.value and "rh (%)" in ans.value
    assert "26 January 2015" in ans.text


def test_ask_threshold_count():
    idx = pd.date_range("2014-01-01", periods=10, freq="D")
    df = pd.DataFrame({"T (degC)": np.arange(0, 50, 5, dtype=float)}, index=idx)  # 0..45
    ans = interface.Interface(df).ask("how many times was it above 22 degrees?")
    assert ans.intent == "threshold"
    assert ans.value["count"] == 5   # 25,30,35,40,45


def test_resolve_compare():
    assert interface.resolve_compare("compare 2013 and 2014")[:2] == (2013, 2014)
    assert interface.resolve_compare("was 2013 hotter than 2014?")[2] == "max"
    assert interface.resolve_compare("average in 2013") is None


def test_ask_compare():
    idx = pd.date_range("2013-01-01", periods=730, freq="D")     # 2013 + 2014
    t = np.where(idx.year == 2014, 15.0, 5.0)                     # 2014 warmer
    df = pd.DataFrame({"T (degC)": t}, index=idx)
    ans = interface.Interface(df).ask("was 2013 hotter than 2014?")
    assert ans.intent == "compare"
    assert "2014" in ans.text and "2013" in ans.text


def test_ask_trend_rising():
    idx = pd.date_range("2014-01-01", periods=24 * 60, freq="h")
    df = pd.DataFrame({"T (degC)": np.linspace(0, 10, len(idx))}, index=idx)  # steadily up
    ans = interface.Interface(df).ask("is temperature rising in 2014?")
    assert ans.intent == "trend"
    assert ans.value > 0


def test_ask_forecast():
    idx = pd.date_range("2014-01-01", periods=24 * 400, freq="h")
    df = pd.DataFrame({"T (degC)": np.linspace(0, 12, len(idx))}, index=idx)
    ans = interface.Interface(df).ask("what will the temperature be next week?")
    assert ans.intent == "forecast"
    assert ans.value is not None and len(ans.value) == 14


def test_missing_sensor():
    idx = pd.date_range("2014-01-01", periods=10, freq="h")
    df = pd.DataFrame({"T (degC)": range(10)}, index=idx)         # no humidity column
    ans = interface.Interface(df).ask("what is the average humidity?")
    assert "humidity" in ans.text.lower() and "don't track" in ans.text.lower()


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


JENA_COLUMNS = [
    "p (mbar)", "T (degC)", "Tpot (K)", "Tdew (degC)", "rh (%)", "VPmax (mbar)",
    "VPact (mbar)", "VPdef (mbar)", "sh (g/kg)", "H2OC (mmol/mol)", "rho (g/m**3)",
    "wv (m/s)", "max. wv (m/s)", "wd (deg)",
]


def test_all_14_sensors_have_plain_english_aliases():
    cols = JENA_COLUMNS
    phrases = {
        "pressure": "p (mbar)",
        "temperature": "T (degC)",
        "potential temperature": "Tpot (K)",
        "dew point": "Tdew (degC)",
        "humidity": "rh (%)",
        "saturation vapour pressure": "VPmax (mbar)",
        "actual vapour pressure": "VPact (mbar)",
        "vapour pressure deficit": "VPdef (mbar)",
        "specific humidity": "sh (g/kg)",
        "water vapour concentration": "H2OC (mmol/mol)",
        "air density": "rho (g/m**3)",
        "wind speed": "wv (m/s)",
        "wind gust": "max. wv (m/s)",
        "wind direction": "wd (deg)",
    }
    for phrase, col in phrases.items():
        got = interface.resolve_sensors(phrase, cols)
        assert got and got[0] == col, f"{phrase!r} -> {got}, expected {col} first"


def test_specific_alias_beats_generic():
    cols = JENA_COLUMNS
    assert interface.resolve_sensors("wind direction", cols)[0] == "wd (deg)"      # not wind speed
    assert interface.resolve_sensors("vapour pressure", cols)[0] == "VPact (mbar)"  # not pressure
    assert interface.resolve_sensors("specific humidity", cols)[0] == "sh (g/kg)"   # not humidity
