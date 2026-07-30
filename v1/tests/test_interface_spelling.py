"""Tests for typo tolerance: `spell_normalize` repairs obvious misspellings of domain words
before routing, so a question like "higest tempreature" still resolves — but conservatively,
never rewriting a legitimate word onto a domain term (which would silently change the meaning).
"""

import numpy as np
import pandas as pd
import pytest

from src import interface
from src.interface_parsing import clarifications_needed, spell_normalize, unrecognized_words

COLS = ("p (mbar)", "T (degC)", "Tpot (K)", "Tdew (degC)", "rh (%)", "VPmax (mbar)",
        "VPact (mbar)", "VPdef (mbar)", "sh (g/kg)", "H2OC (mmol/mol)", "rho (g/m**3)",
        "wv (m/s)", "max. wv (m/s)", "wd (deg)")


@pytest.mark.parametrize("typo, fixed", [
    ("higest", "highest"),
    ("tempreature", "temperature"),
    ("humidty", "humidity"),
    ("avearge", "average"),
    ("maximun", "maximum"),
    ("presure", "pressure"),
    ("minimun", "minimum"),
    ("corelation", "correlation"),
    ("anomoly", "anomaly"),
])
def test_common_typos_are_corrected(typo, fixed):
    out, fixes = spell_normalize(f"{typo} in 2015")
    assert out == f"{fixed} in 2015"
    assert (typo, fixed) in fixes


@pytest.mark.parametrize("word", [
    "summer", "winter", "spring", "autumn", "january", "august", "weather", "warmer",
    "monthly", "yearly", "measure", "present", "between", "record", "during", "season",
    "rainfall", "today", "temperature", "humidity", "pressure",
])
def test_legitimate_words_are_left_untouched(word):
    """A correctly-spelled word (or one simply outside the vocabulary) must never be 'corrected'."""
    out, fixes = spell_normalize(f"{word} 2015")
    assert out == f"{word} 2015"
    assert fixes == []


def test_numbers_and_years_are_never_altered():
    out, fixes = spell_normalize("higest temperature in 2015")
    assert "2015" in out and fixes == [("higest", "highest")]


def _year_df():
    idx = pd.date_range("2015-01-01", "2016-01-01", freq="h")
    rng = np.random.default_rng(0)
    return pd.DataFrame({"T (degC)": rng.normal(10, 5, len(idx)),
                         "rh (%)": rng.normal(70, 10, len(idx))}, index=idx)


def test_end_to_end_typo_question_routes_and_reports_the_fix():
    iface = interface.Interface(_year_df())
    ans = iface.ask("higest tempreature in 2015")
    assert ans.intent == "max"
    assert ans.sensors == ["T (degC)"]
    assert ("higest", "highest") in ans.spelling
    assert ("tempreature", "temperature") in ans.spelling
    assert ans.text.lower().startswith("(reading")   # correction surfaced to the user
    # the user's original wording is preserved for display
    assert ans.question == "higest tempreature in 2015"


def test_clean_question_has_no_spelling_note():
    iface = interface.Interface(_year_df())
    ans = iface.ask("highest temperature in 2015")
    assert ans.spelling == []
    assert not ans.text.lower().startswith("(reading")


@pytest.mark.parametrize("typo, fixed", [
    ("octber", "october"),
    ("septmber", "september"),
    ("wnter", "winter"),
    ("agust", "august"),
])
def test_month_and_season_typos_are_corrected(typo, fixed):
    out, fixes = spell_normalize(f"humidity in {typo} 2012")
    assert out == f"humidity in {fixed} 2012"
    assert (typo, fixed) in fixes


@pytest.mark.parametrize("query", [
    "hottest day in 2013", "average wind speed in winter 2012",
    "correlation between temperature and humidity", "anomalies in 2014",
    "compare 2013 and 2014 temperature", "forecast next week",
    "what kind of day was 5 january 2014", "summary of pressure in july 2015",
    "coldest temperature during july 2015", "recommend hot days in june",
    "how humid was it in august 2011", "dew point in spring 2013",
])
def test_normal_queries_report_no_ignored_words(query):
    assert unrecognized_words(query, COLS) == []


def test_mangled_word_is_reported_as_ignored():
    # "tocber" is too far from "october" (ratio 0.77) to safely auto-correct, so instead of
    # silently dropping it, it must be reported as unrecognized.
    assert unrecognized_words("humidity in tocber 2012", COLS) == ["tocber"]


def test_column_and_unit_mentions_are_not_flagged():
    assert unrecognized_words("vpmax in mbar during 2015", COLS) == []


def test_garbled_query_stops_and_asks_instead_of_guessing():
    """GUARDRAIL: a word too mangled to correct must block the answer and ask the user to fix it,
    rather than silently computing on a guessed scope."""
    iface = interface.Interface(_year_df())
    ans = iface.ask("waht si umidty tocber 2015")
    assert ans.intent is None                        # no computed answer was returned
    assert ans.value is None
    assert ans.ignored == ["tocber"]
    assert ("umidty", "humidity") in ans.spelling    # what it *could* fix is still reported
    assert "tocber" in ans.text
    assert "october" in ans.text.lower()             # offered as a "did you mean?" hint
    assert "again" in ans.text.lower()               # asks the user to retry


def test_guardrail_offers_a_suggestion_when_one_is_plausible():
    iface = interface.Interface(_year_df())
    ans = iface.ask("lowset temperature in 2015")
    assert ans.intent is None
    assert "lowest" in ans.text.lower()              # did-you-mean hint for the sub-threshold typo


def test_hedges_and_politeness_do_not_trip_the_guardrail():
    iface = interface.Interface(_year_df())
    ans = iface.ask("roughly what was the average temperature in 2015 please")
    assert ans.intent == "mean"                      # answered normally, not blocked
    assert ans.ignored == []


@pytest.mark.parametrize("word", [
    "because", "cause", "causes", "change", "second", "degrees", "staying", "within",
    "often", "third", "nonsense", "something", "elephants", "weather",
])
def test_ordinary_words_do_not_trigger_the_guardrail(word):
    """Words that merely appear in a query (causal/ordinal/threshold/off-topic terms) are far
    enough from any domain term that the gray-zone guardrail must leave them alone -- otherwise it
    would block normal questions like 'how many hours above 22 degrees' or 'what causes spikes'."""
    assert clarifications_needed(f"{word} in 2015") == []


@pytest.mark.parametrize("typo, suggestion", [
    ("tocber", "october"),
    ("lowset", "lowest"),
    ("sumemr", "summer"),
])
def test_near_miss_typos_are_flagged_with_a_suggestion(typo, suggestion):
    got = clarifications_needed(f"temperature {typo} 2015")
    assert got == [(typo, suggestion)]
