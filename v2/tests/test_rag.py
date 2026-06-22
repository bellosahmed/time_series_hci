"""Tests for v2 hybrid RAG retriever. Run: cd code/v2 && python3.12 -m pytest tests/"""
import sys
from pathlib import Path

import pytest

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2))
from src import rag_system as rag


@pytest.fixture(scope="module")
def kb():
    return rag.knowledge_base()


@pytest.fixture(scope="module")
def hybrid(kb):
    return rag.HybridRetriever().fit(kb)


def test_bm25_exact_term(kb):
    r = rag.BM25().fit(kb).search("trade surplus", k=3)
    ids = [i for i, _ in r]
    assert "surplus" in ids or "balance" in ids          # lexical hit on the term


def test_semantic_finds_paraphrase(kb):
    # query uses 'pandemic' / 'collapse' — the COVID doc says 'COVID-19' / 'crashed'
    r = rag.SemanticLSA().fit(kb).search("pandemic lockdown trade collapse", k=3)
    ids = [i for i, _ in r]
    assert "event_covid" in ids


def test_hybrid_returns_k(hybrid):
    r = hybrid.search("germany exports surplus", k=3)
    assert len(r) == 3
    assert r[0][1] >= r[-1][1]                            # sorted by score


def test_hybrid_germany_query(hybrid):
    ids = [i for i, _ in hybrid.search("which country is the biggest exporter", k=3)]
    assert "DE" in ids


def test_retrieve_texts_shape(hybrid):
    docs = hybrid.retrieve_texts("2008 financial crisis", k=2)
    assert len(docs) == 2
    assert all(isinstance(t, str) and t for _, t in docs)


def test_explain_has_trust_labels():
    out = rag.explain("DE exports were 100 EUR m",
                      retrieved=[("DE", "Germany is the largest EU exporter.")],
                      inferred="likely a strong export month")
    assert "🔢" in out and "📚" in out and "🧠" in out


def test_rag_provides_no_numbers(kb):
    # the KB must not contain hard statistics that could be mistaken for the truth layer
    joined = " ".join(t for _, t in kb)
    assert "EUR" not in joined                            # no euro amounts in context docs


# --- chunking ---
def test_chunker_splits_long_text():
    text = ("Sentence one is here. Sentence two follows it. Sentence three continues. "
            "Sentence four adds more. Sentence five keeps going. Sentence six ends it.") * 3
    chunks = rag.chunk_text(text, max_words=20, overlap=5)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 30 for c in chunks)      # roughly bounded (max + overlap slack)


def test_chunker_short_text_single_chunk():
    assert rag.chunk_text("Short doc.", max_words=40) == ["Short doc."]


# --- BERT (skip if sentence-transformers unavailable) ---
st = pytest.importorskip("sentence_transformers")


@pytest.fixture(scope="module")
def bert(kb):
    return rag.BertRetriever().fit(kb)


def test_bert_finds_paraphrase(bert):
    ids = [i for i, _ in bert.search("lockdown crash in global commerce", k=3)]
    assert "event_covid" in ids                           # meaning matched, not keywords


def test_hybrid_bert_backend(kb):
    h = rag.HybridRetriever(semantic="bert").fit(kb)
    ids = [i for i, _ in h.search("biggest exporter surplus", k=3)]
    assert "DE" in ids
