"""Evaluation for the Understanding agent: a corpus of MALFORMED / paraphrased queries and a scorer
that reports how many an interface answers correctly. Run it with the LLM on vs off to quantify the
capability the Understanding agent adds. Ground truth is the obviously-correct human reading, written
before running any code (same principle as ``query_coverage.GROUND_TRUTH``)."""
from __future__ import annotations

from .query_coverage import Case

# Messy queries a human reads instantly but the deterministic parser struggles with: heavy typos
# below the auto-correct threshold, paraphrase, and word-order scrambles. `has_scope` follows the
# same meaning as in query_coverage (a specific time window is named).
MALFORMED_CASES: list[Case] = [
    Case("higist tempreture 2015", "max", frozenset({"T (degC)"}), True),
    Case("umidty tocber 2012", "summary", frozenset({"rh (%)"}), True),
    Case("how muggy did it get in summer 2013", "summary", frozenset({"rh (%)"}), True),
    Case("avrage windspeed 2016", "mean", frozenset({"wv (m/s)"}), True),
    Case("coldist it evar got", "min", frozenset({"T (degC)"}), False),
    Case("presure in jan 2014 lowset", "min", frozenset({"p (mbar)"}), True),
    Case("how breezy in june", "summary", frozenset({"wv (m/s)"}), True),
    Case("dew pont standard deviaton 2014", "spread", frozenset({"Tdew (degC)"}), True),
]


def score(iface, cases: list[Case] = MALFORMED_CASES) -> dict:
    """Run each malformed query through ``iface.ask`` and count outcomes.

    A case is *correct* when the routed intent matches the expected intent and the expected sensors
    are a subset of the routed sensors. A case with no value is *declined* (asked to rephrase) --
    which is the right outcome for a genuinely ambiguous query, not a failure. Returns
    ``{"n", "answered", "correct", "declined"}``.
    """
    n = len(cases)
    answered = correct = declined = 0
    for case in cases:
        iface.history = []
        ans = iface.ask(case.question)
        if ans.value is None:
            declined += 1
            continue
        answered += 1
        if ans.intent == case.intent and case.sensors.issubset(set(ans.sensors)):
            correct += 1
    return {"n": n, "answered": answered, "correct": correct, "declined": declined}
