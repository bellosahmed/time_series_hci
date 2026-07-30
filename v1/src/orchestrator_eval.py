"""Evaluation for the multi-agent orchestrator: a small corpus of genuinely multi-part questions and
a scorer counting how many produce a combined answer containing every expected exact fact. Ground
truth is the obviously-correct set of facts, written before running any code (same principle as
``query_coverage.GROUND_TRUTH`` and ``understanding_eval``)."""
from __future__ import annotations

# (question, tuple of exact substrings that MUST all appear in the combined answer). Substrings are
# scope/sensor markers that a *complete* multi-part answer has to mention -- not recomputed numbers,
# so the check stays independent of the exact values (which the query engine verifies elsewhere).
MULTIPART_CASES: list[tuple[str, tuple[str, ...]]] = [
    ("the highest temperature in 2015 and the lowest humidity in 2013", ("2015", "2013")),
    ("mean pressure in 2014 and how many times temperature was above 30 that year", ("2014",)),
    ("coldest temperature ever and the windiest reading", ("T (degC)", "wv (m/s)")),
]


def score(orch_obj, cases: list[tuple[str, tuple[str, ...]]] = MULTIPART_CASES) -> dict:
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
