"""Retrieval-quality evaluation for the RAG layer — standard IR metrics.

The truth layer (`query_engine`) is already verified `== pandas`; this evaluates the *other*
half of the RAG pipeline — the **retriever** that grounds explanations in sensor metadata —
using the Information-Retrieval measures from the module: **Precision@k, Recall@k, MRR, MAP,
NDCG@k** (binary relevance). Evaluation follows the taught recipe: a small hand-labelled query
set (query → the sensor doc(s) that are genuinely relevant), run the retriever, compare its
ranked output to the labels, and report the metrics in a table.

This lets us quantify — not just assert — retrieval quality, and compare retrievers
(TF-IDF baseline vs. LSA) on the same ground truth. See `docs/archive/RETRIEVAL_EVALUATION.md`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

# --- Hand-labelled ground truth ------------------------------------------------------------
# Each entry: (query, {relevant sensor-doc keys}). Relevance is defined *externally* by us
# (the human assessor), never by the system — the standard IR evaluation principle. Keys are
# the sensor columns used as document ids in `rag_system.SENSOR_DOCS`. Kept small but
# deliberately covering every sensor family; the annotation criterion is "the query names
# (a synonym of) exactly this sensor".
GROUND_TRUTH: list[tuple[str, set[str]]] = [
    ("how hot does it get", {"T (degC)"}),
    ("air temperature in celsius", {"T (degC)"}),
    ("relative humidity percent", {"rh (%)"}),
    ("atmospheric pressure in millibars", {"p (mbar)"}),
    ("wind speed", {"wv (m/s)"}),
    ("wind direction in degrees", {"wd (deg)"}),
    ("dew point temperature", {"Tdew (degC)"}),
    ("air density", {"rho (g/m**3)"}),
    ("maximum wind gust", {"max. wv (m/s)"}),
    ("saturation vapour pressure", {"VPmax (mbar)"}),
]

# Paraphrase / out-of-vocabulary queries: they name a sensor WITHOUT using the knowledge base's
# words, so keyword methods (TF-IDF, BM25, and LSA built on TF-IDF) cannot match them. This set
# isolates the semantic gap that only a dense sentence-encoder retriever can close.
PARAPHRASE_GROUND_TRUTH: list[tuple[str, set[str]]] = [
    ("how hot does it get outside", {"T (degC)"}),
    ("is the air muggy and sticky", {"rh (%)"}),
    ("how breezy is it today", {"wv (m/s)"}),
    ("how heavy or thin is the air", {"rho (g/m**3)"}),
    ("how strong are the sudden blasts of wind", {"max. wv (m/s)"}),
]

# HELD-OUT paraphrase set — an *honest* test that avoids "teaching to the test". Every descriptive
# word here was verified **absent** from the (enriched) knowledge base, so no retriever has seen it.
# Keyword methods therefore cannot match on the word; only a model that generalises from its own
# pretraining (the dense encoder) can. This is the fair generalisation number to report.
HELDOUT_GROUND_TRUTH: list[tuple[str, set[str]]] = [
    ("a sweltering baking afternoon", {"T (degC)"}),
    ("a nippy wintry chilly morning", {"T (degC)"}),
    ("the air feels oppressive and sultry", {"rh (%)"}),
    ("a blustery howling gale", {"wv (m/s)"}),
    ("strong buffeting battering gusts", {"max. wv (m/s)"}),
    ("the air feels weighty and compressed", {"rho (g/m**3)"}),
]


# --- Metrics (operate on a ranked list of ids vs. a set of relevant ids) --------------------

def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Of the top-k retrieved, the fraction that are relevant."""
    if k <= 0:
        return 0.0
    topk = ranked[:k]
    return sum(1 for r in topk if r in relevant) / k


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Of all relevant docs, the fraction found in the top-k."""
    if not relevant:
        return 0.0
    topk = set(ranked[:k])
    return sum(1 for r in relevant if r in topk) / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant result (0 if none retrieved)."""
    for i, r in enumerate(ranked, start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def average_precision(ranked: Sequence[str], relevant: set[str]) -> float:
    """Average of Precision@k taken at each rank where a relevant doc appears (AP)."""
    if not relevant:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for i, r in enumerate(ranked, start=1):
        if r in relevant:
            hits += 1
            precision_sum += hits / i
    return precision_sum / len(relevant)


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Normalised Discounted Cumulative Gain @k (binary relevance)."""
    if not relevant:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 1) for i, r in enumerate(ranked[:k], start=1) if r in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def _ranked_ids(retriever, query: str, k: int) -> list[str]:
    """Run a retriever and return its ranked list of sensor-doc ids (the `.sensor` field)."""
    return [snip.sensor for snip in retriever.retrieve(query, k=k)]


def evaluate_retriever(retriever, ground_truth=GROUND_TRUTH, k: int = 3) -> dict:
    """Evaluate `retriever` over the labelled query set; return the mean IR metrics.

    Returns ``{k, n, p_at_k, r_at_k, mrr, map, ndcg}`` — means across all queries. Any object
    with a ``retrieve(query, k)`` method returning ranked `Snippet`s works (TF-IDF, LSA, or the
    embedding retriever), so retrievers are compared on identical ground truth.
    """
    n = len(ground_truth)
    if n == 0:
        return {"k": k, "n": 0, "p_at_k": 0.0, "r_at_k": 0.0, "mrr": 0.0, "map": 0.0, "ndcg": 0.0}
    p = r = mrr = ap = ndcg = 0.0
    for query, relevant in ground_truth:
        ranked = _ranked_ids(retriever, query, k)
        p += precision_at_k(ranked, relevant, k)
        r += recall_at_k(ranked, relevant, k)
        mrr += reciprocal_rank(ranked, relevant)
        ap += average_precision(ranked, relevant)
        ndcg += ndcg_at_k(ranked, relevant, k)
    return {"k": k, "n": n, "p_at_k": p / n, "r_at_k": r / n,
            "mrr": mrr / n, "map": ap / n, "ndcg": ndcg / n}
