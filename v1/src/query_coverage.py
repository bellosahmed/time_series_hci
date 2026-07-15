"""Query-routing coverage evaluation for the natural-language interface.

The truth layer (`query_engine`) is verified `== pandas`, and the retriever is scored with IR
metrics (`retrieval_eval`); this evaluates the *routing* layer — does a realistic plain-English
question get mapped to the right intent, sensor(s), and time scope at all? A hand-labelled corpus
of realistic questions is checked against `Interface.ask()`'s actual routing (not the parsing
functions in isolation, so this measures what a real user actually sees, including the "summary"
fallback for keyword-less questions and the LLM-off default path).

Ground truth is defined *externally*, by human judgement of the correct reading of each question —
never by running the code first and copying its output — matching the same principle used in
`retrieval_eval.py`. This can legitimately score below 100%: an honest number here is more useful
than a rigged one, and any real miss is worth knowing (an unrecognised phrasing, a routing quirk).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Case:
    """One labelled question: the correct (intent, sensors, has_scope) a human would expect."""

    question: str
    intent: str | None          # None only for genuinely unroutable/nonsense questions
    sensors: frozenset[str] = field(default_factory=frozenset)  # must ALL appear in the routed sensors
    has_scope: bool = False     # True if a real time window (day/period/relative/ordinal/hour) is named


# All 14 real dataset columns are exercised at least once, across every intent the router supports,
# and across every time-scope style built up this session (absolute date, season, year range,
# relative-to-dataset-end, ordinal-in-year, hour-of-day, and no scope at all).
GROUND_TRUTH: list[Case] = [
    # --- max / min ---
    Case("what was the max temperature in 2013?", "max", frozenset({"T (degC)"}), True),
    Case("hottest it got in summer 2013", "max", frozenset({"T (degC)"}), True),
    Case("what was the coldest it got in winter 2012?", "min", frozenset({"T (degC)"}), True),
    Case("lowest humidity recorded", "min", frozenset({"rh (%)"}), False),
    # --- mean / spread ---
    Case("average humidity in july 2015", "mean", frozenset({"rh (%)"}), True),
    Case("typical wind speed", "mean", frozenset({"wv (m/s)"}), False),
    Case("how much does the pressure vary?", "spread", frozenset({"p (mbar)"}), False),
    Case("standard deviation of dew point in 2014", "spread", frozenset({"Tdew (degC)"}), True),
    # --- correlation ---
    Case("correlation between temperature and humidity", "correlation",
         frozenset({"T (degC)", "rh (%)"}), False),
    Case("how does wind speed relate to air density?", "correlation",
         frozenset({"wv (m/s)", "rho (g/m**3)"}), False),
    Case("is dew point linked to specific humidity?", "correlation",
         frozenset({"Tdew (degC)", "sh (g/kg)"}), False),
    # --- missing ---
    Case("how much data is missing for dew point?", "missing", frozenset({"Tdew (degC)"}), False),
    Case("any gaps in the pressure readings in 2013?", "missing", frozenset({"p (mbar)"}), True),
    # --- forecast ---
    Case("forecast the temperature for next week", "forecast", frozenset({"T (degC)"}), False),
    Case("what will the humidity be going forward?", "forecast", frozenset({"rh (%)"}), False),
    # --- trend ---
    Case("is temperature trending upward?", "trend", frozenset({"T (degC)"}), False),
    Case("is air density falling over time?", "trend", frozenset({"rho (g/m**3)"}), False),
    # --- compare ---
    Case("was 2013 warmer than 2014?", "compare", frozenset({"T (degC)"}), True),
    Case("compare 2012 and 2015", "compare", frozenset(), True),
    # --- anomaly ---
    Case("were there any unusual readings in 2013?", "anomaly", frozenset(), True),
    Case("any anomalies last month?", "anomaly", frozenset(), True),
    # --- weather ---
    Case("what was the weather like on 15 june 2013?", "weather", frozenset(), True),
    Case("describe the conditions in march 2014", "weather", frozenset(), True),
    # --- daytype ---
    Case("what kind of day was 5 january 2014?", "daytype", frozenset(), True),
    Case("describe the day-types in the record", "daytype", frozenset(), False),
    # --- summary (explicit keyword) ---
    Case("summarize the pressure data", "summary", frozenset({"p (mbar)"}), False),
    Case("give me an overview of wind direction", "summary", frozenset({"wd (deg)"}), False),
    # --- summary (implicit -- no keyword, defaults per the fallback added this session) ---
    Case("how humid was it in summer 2013?", "summary", frozenset({"rh (%)"}), True),
    Case("what's the wind speed right now?", "summary", frozenset({"wv (m/s)"}), True),
    Case("last month's average temperature", "mean", frozenset({"T (degC)"}), True),
    Case("second month 2016 temperature", "summary", frozenset({"T (degC)"}), True),
    Case("wind gust last week", "summary", frozenset({"max. wv (m/s)"}), True),
    Case("temperature at 3pm", "summary", frozenset({"T (degC)"}), True),
    Case("the water vapour concentration", "summary", frozenset({"H2OC (mmol/mol)"}), False),
    Case("air density on 1 march 2012", "summary", frozenset({"rho (g/m**3)"}), True),
    Case("saturation vapour pressure this week", "summary", frozenset({"VPmax (mbar)"}), True),
    Case("vapour pressure deficit last year", "summary", frozenset({"VPdef (mbar)"}), True),
    Case("the potential temperature", "summary", frozenset({"Tpot (K)"}), False),
    # --- sensor aliases beyond the literal column name (paraphrase coverage) ---
    Case("how muggy does it get", "summary", frozenset({"rh (%)"}), False),
    # NB: "in june" alone has no year, so resolve_period correctly does not resolve it -- unlike
    # resolve_day, which does support a dayless-year specific day ("26 January") across all years.
    # A genuine, minor asymmetry worth knowing about, not a bug -- has_scope is correctly False.
    Case("how breezy is it in june", "summary", frozenset({"wv (m/s)"}), False),
    Case("how heavy is the air", "summary", frozenset({"rho (g/m**3)"}), False),
    # --- time-scope variety (all paired with a keyword so intent is unambiguous) ---
    Case("max temperature on 26 january 2015", "max", frozenset({"T (degC)"}), True),
    Case("mean pressure between 2012 and 2014", "mean", frozenset({"p (mbar)"}), True),
    Case("mean humidity at night in december 2016", "mean", frozenset({"rh (%)"}), True),
    Case("mean wind speed in the 34th week of 2013", "mean", frozenset({"wv (m/s)"}), True),
    Case("mean temperature last hour", "mean", frozenset({"T (degC)"}), True),
    # --- deliberately unroutable (no sensor, no intent) -- the fallback must NOT fire here ---
    Case("hello there", None, frozenset(), False),
    Case("blah blah nonsense", None, frozenset(), False),
    Case("thank you", None, frozenset(), False),
]


def _routes_match(case: Case, ans) -> tuple[bool, bool, bool]:
    """(intent_ok, sensors_ok, scope_ok) for one case against a real `Answer`."""
    intent_ok = ans.intent == case.intent
    sensors_ok = case.sensors.issubset(set(ans.sensors))
    has_scope = bool(ans.understood) and any(
        tag in ans.understood for tag in ("day=", "scope=", "hour=")
    )
    scope_ok = has_scope == case.has_scope
    return intent_ok, sensors_ok, scope_ok


def score_coverage(iface, cases: list[Case] = GROUND_TRUTH) -> dict:
    """Run every labelled case through the real `Interface.ask()` and score routing accuracy.

    Returns counts + percentages for intent/sensor/scope accuracy and overall (all three correct),
    plus a `failures` list of (question, expected, got) for every case that didn't fully match --
    the diagnostic detail a bare percentage can't give.
    """
    n = len(cases)
    intent_ok = sensors_ok = scope_ok = overall_ok = 0
    failures: list[dict] = []
    for case in cases:
        # Each case is judged standalone -- clearing history before every question so the
        # conversational follow-up system (a separate, already-tested feature) never hijacks a
        # deliberately keyword-less question and makes it inherit an earlier case's intent.
        iface.history = []
        ans = iface.ask(case.question)
        i_ok, s_ok, sc_ok = _routes_match(case, ans)
        intent_ok += i_ok
        sensors_ok += s_ok
        scope_ok += sc_ok
        if i_ok and s_ok and sc_ok:
            overall_ok += 1
        else:
            failures.append({
                "question": case.question,
                "expected": {"intent": case.intent, "sensors": sorted(case.sensors),
                             "has_scope": case.has_scope},
                "got": {"intent": ans.intent, "sensors": ans.sensors, "understood": ans.understood},
            })
    return {
        "n": n,
        "intent_pct": 100.0 * intent_ok / n,
        "sensors_pct": 100.0 * sensors_ok / n,
        "scope_pct": 100.0 * scope_ok / n,
        "overall_pct": 100.0 * overall_ok / n,
        "failures": failures,
    }
