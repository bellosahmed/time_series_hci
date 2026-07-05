"""Tests for the retrieval-quality IR metrics (`src/retrieval_eval.py`).

The metric functions are checked against hand-computed values; `evaluate_retriever` is checked
end-to-end on the real TF-IDF and LSA retrievers over the labelled query set.
"""

import pytest

from src import rag_system
from src import retrieval_eval as re


def test_precision_and_recall_at_k():
    ranked = ["a", "b", "c"]
    assert re.precision_at_k(ranked, {"a", "c"}, 3) == 2 / 3
    assert re.recall_at_k(ranked, {"a", "c"}, 3) == 1.0
    assert re.recall_at_k(ranked, {"a", "z"}, 3) == 0.5      # only 1 of 2 relevant found
    assert re.precision_at_k(ranked, {"a"}, 0) == 0.0        # k=0 guard


def test_reciprocal_rank():
    assert re.reciprocal_rank(["x", "a"], {"a"}) == 0.5      # first relevant at rank 2
    assert re.reciprocal_rank(["a", "x"], {"a"}) == 1.0
    assert re.reciprocal_rank(["x", "y"], {"a"}) == 0.0      # none retrieved


def test_average_precision():
    # relevant at ranks 1 and 3 -> AP = (1/1 + 2/3) / 2
    ap = re.average_precision(["a", "x", "c"], {"a", "c"})
    assert abs(ap - (1.0 + 2 / 3) / 2) < 1e-9


def test_ndcg_at_k_bounds_and_perfect():
    assert re.ndcg_at_k(["a", "b"], {"a"}, 2) == 1.0         # relevant already at rank 1 -> perfect
    assert 0.0 < re.ndcg_at_k(["x", "a"], {"a"}, 2) < 1.0    # discounted for lower rank
    assert re.ndcg_at_k([], {"a"}, 2) == 0.0


def test_evaluate_tfidf_retriever_on_ground_truth():
    m = re.evaluate_retriever(rag_system.Retriever(), k=3)
    assert m["n"] == len(re.GROUND_TRUTH)
    for key in ("p_at_k", "r_at_k", "mrr", "map", "ndcg"):
        assert 0.0 <= m[key] <= 1.0
    assert m["mrr"] >= 0.8                                   # strong baseline on the labelled set


def test_evaluate_lsa_retriever_on_ground_truth():
    m = re.evaluate_retriever(rag_system.LSARetriever(), k=3)
    assert 0.0 <= m["mrr"] <= 1.0 and m["mrr"] >= 0.8        # LSA at least matches the baseline here


def test_enriched_kb_lets_keyword_methods_handle_paraphrases():
    # After enriching the KB with everyday synonyms, even literal TF-IDF matches paraphrase
    # queries well — the earlier vocabulary-mismatch gap is closed by better documents, not a
    # fancier retriever (see docs/RETRIEVAL_EVALUATION.md).
    m = re.evaluate_retriever(rag_system.Retriever(), re.PARAPHRASE_GROUND_TRUTH, k=3)
    assert m["mrr"] >= 0.8


def test_dense_handles_paraphrases_if_available():
    pytest.importorskip("sentence_transformers")
    dense = rag_system.dense_retriever()
    if not getattr(dense, "semantic", False):
        pytest.skip("sentence-transformers model weights not available")
    m = re.evaluate_retriever(dense, re.PARAPHRASE_GROUND_TRUTH, k=3)
    assert m["mrr"] >= 0.6                                   # still strong (though enriched TF-IDF now ties/leads)


# --- held-out generalisation (the "not teaching to the test" check) -----------------------

_HELDOUT_STOP = {"a", "the", "and", "is", "it", "are", "of", "feels", "air", "how", "does",
                 "get", "outside", "today", "morning", "afternoon", "strong"}


def test_heldout_words_are_genuinely_absent_from_kb():
    import re as _re
    kb = set(_re.findall(r"[a-z]+", " ".join(rag_system.SENSOR_DOCS.values()).lower()))
    for q, _ in re.HELDOUT_GROUND_TRUTH:
        for w in _re.findall(r"[a-z]+", q.lower()):
            if w not in _HELDOUT_STOP:
                assert w not in kb, f"held-out word {w!r} leaked into the KB — test is not held out"


def test_keyword_methods_are_weak_on_heldout():
    # honest: on truly unseen vocabulary a keyword method has no word to match -> near-chance
    m = re.evaluate_retriever(rag_system.Retriever(), re.HELDOUT_GROUND_TRUTH, k=3)
    assert m["mrr"] <= 0.5


def test_dense_generalises_on_heldout_if_available():
    pytest.importorskip("sentence_transformers")
    dense = rag_system.dense_retriever()
    if not getattr(dense, "semantic", False):
        pytest.skip("sentence-transformers model weights not available")
    d = re.evaluate_retriever(dense, re.HELDOUT_GROUND_TRUTH, k=3)
    t = re.evaluate_retriever(rag_system.Retriever(), re.HELDOUT_GROUND_TRUTH, k=3)
    assert d["mrr"] > t["mrr"]                               # dense generalises to unseen words; keyword can't
