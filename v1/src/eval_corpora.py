"""Two small evaluation corpora, each paired with its own scorer:

- The multi-agent **orchestrator**'s corpus: genuinely multi-part questions, scored by whether the
  combined answer contains every expected exact fact.
- The **Understanding agent**'s corpus: malformed/paraphrased queries, scored by whether the
  interface still routes them correctly.

Ground truth for both is the obviously-correct set written before running any code (same principle
as ``query_coverage.GROUND_TRUTH``). Kept in one module since both are single-purpose
{corpus + scorer} pairs used only by ``batch_harness.py`` and their own tests.
"""
from __future__ import annotations

from .query_coverage import Case

# --- Orchestrator ---------------------------------------------------------------------------

# (question, tuple of exact substrings that MUST all appear in the combined answer). Substrings are
# scope/sensor markers that a *complete* multi-part answer has to mention -- not recomputed numbers,
# so the check stays independent of the exact values (which the query engine verifies elsewhere).
MULTIPART_CASES: list[tuple[str, tuple[str, ...]]] = [
    ("the highest temperature in 2015 and the lowest humidity in 2013", ("2015", "2013")),
    ("mean pressure in 2014 and how many times temperature was above 30 that year", ("2014",)),
    ("coldest temperature ever and the windiest reading", ("T (degC)", "wv (m/s)")),
]


def score_orchestrator(orch_obj, cases: list[tuple[str, tuple[str, ...]]] = MULTIPART_CASES) -> dict:
    """Run each multi-part question through ``orch_obj.run`` and count *complete* answers (every
    expected substring present in text+computed). Returns ``{"n", "complete"}``."""
    n = len(cases)
    complete = 0
    for question, expected in cases:
        ans = orch_obj.run(question)
        blob = f"{getattr(ans, 'text', '')} {getattr(ans, 'computed', '')}"
        if all(sub in blob for sub in expected):
            complete += 1
    return {"n": n, "complete": complete}


# --- Understanding agent ---------------------------------------------------------------------

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


def score_understanding(iface, cases: list[Case] = MALFORMED_CASES) -> dict:
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
