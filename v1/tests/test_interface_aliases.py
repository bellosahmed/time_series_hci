"""Tests for sensor-alias coverage: plain-English phrasings resolving to the right column,
including ones found by the query-coverage/held-out benchmarks (`src/query_coverage.py`)."""

from src import interface

JENA_COLUMNS = [
    "p (mbar)", "T (degC)", "Tpot (K)", "Tdew (degC)", "rh (%)", "VPmax (mbar)",
    "VPact (mbar)", "VPdef (mbar)", "sh (g/kg)", "H2OC (mmol/mol)", "rho (g/m**3)",
    "wv (m/s)", "max. wv (m/s)", "wd (deg)",
]


def test_informal_synonyms_from_the_rag_description_text_are_also_wired_as_aliases():
    # These words already appear in rag_system's own sensor descriptions ("how muggy, sticky,
    # damp..."), but weren't wired into SENSOR_ALIASES -- found by the query-coverage benchmark
    # (src/query_coverage.py), which showed a real routing miss for each of these questions.
    cols = JENA_COLUMNS
    assert interface.resolve_sensors("how muggy does it get", cols) == ["rh (%)"]
    assert interface.resolve_sensors("how breezy is it", cols) == ["wv (m/s)"]
    assert interface.resolve_sensors("how heavy is the air", cols) == ["rho (g/m**3)"]


def test_going_forward_is_a_forecast_synonym():
    # Also found by the coverage benchmark: "next week"/"future"/"outlook" were already forecast
    # keywords, but the equally common "going forward" wasn't.
    assert interface.detect_intent("what will the humidity be going forward?") == "forecast"


def test_variable_is_a_spread_synonym():
    # Found by the held-out generalisation benchmark (query_coverage.HELD_OUT_GROUND_TRUTH):
    # "vary"/"variability" were already spread keywords, but the equally natural adjective
    # "variable" wasn't.
    assert interface.detect_intent("how variable is the dew point") == "spread"


def test_strongest_and_weakest_are_superlative_intent_keywords():
    # "strongest"/"weakest" are natural superlatives for wind/gust/pressure ("the strongest wind
    # gust") -- interface.py's OWN example query for the gust sensor is "the strongest wind gust in
    # 2015". They belong with max/min like the existing "biggest"/"smallest"/"most".
    assert interface.detect_intent("strongest wind gust in 2012") == "max"
    assert interface.detect_intent("weakest wind in 2013") == "min"


def test_strongest_gust_is_not_flagged_for_clarification():
    # Regression: "strongest" fuzzily matched the anomaly keyword "strange" (0.75-0.85 band), so the
    # clarification guardrail asked "did you mean 'strange'?" on a perfectly clear, fully-routable
    # query -- and with the LLM on, the recover-rewrite dropped the gust, answering on wind SPEED.
    assert not any(w == "strongest"
                   for w, _ in interface.clarifications_needed("strongest wind gust in 2012",
                                                               tuple(JENA_COLUMNS)))


def test_strongest_wind_gust_answers_on_the_gust_sensor():
    import numpy as np
    import pandas as pd
    idx = pd.date_range("2012-01-01", periods=240, freq="h")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"T (degC)": rng.normal(5, 3, len(idx)),
                       "wv (m/s)": rng.uniform(0, 8, len(idx)),
                       "max. wv (m/s)": rng.uniform(4, 16, len(idx))}, index=idx)
    ans = interface.Interface(df).ask("strongest wind gust")
    assert ans.sensors[0] == "max. wv (m/s)"   # the gust sensor, not wind speed
    assert ans.intent == "max"


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


def test_short_aliases_need_a_word_boundary_not_just_a_substring():
    # "thin" (-> air density) used to fire on plain substring search inside completely unrelated
    # words -- most importantly "within" ("staying within April"), one of the "recommend"
    # intent's own restriction-phrase keywords, which silently corrupted sensor detection on
    # every such question. "wind" (-> wind speed) has the same risk inside "window".
    cols = JENA_COLUMNS
    assert interface.resolve_sensors("staying within that month", cols) == []
    assert interface.resolve_sensors("show it in a time window", cols) == []
    # the real words must still resolve correctly
    assert interface.resolve_sensors("how thin is the air", cols) == ["rho (g/m**3)"]
    assert interface.resolve_sensors("wind speed today", cols) == ["wv (m/s)"]
