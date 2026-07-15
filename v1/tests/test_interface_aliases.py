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
