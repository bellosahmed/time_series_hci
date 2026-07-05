"""Tests for the RAG retrievers — TF-IDF baseline and the optional embedding retriever.

The embedding tests inject a fake `embed_fn` so they run offline (no Ollama). The point is
that `EmbeddingRetriever` ranks by cosine similarity when a backend is present and degrades
gracefully to the TF-IDF `Retriever` when it is not.
"""

import numpy as np
import pytest

from src import rag_system as rag


def test_bm25_retriever_finds_relevant_doc():
    pytest.importorskip("rank_bm25")
    r = rag.BM25Retriever()
    assert r.retrieve("relative humidity percent", k=1)[0].sensor == "rh (%)"
    assert r.retrieve("air density", k=1)[0].sensor == "rho (g/m**3)"
    assert r.for_sensor("T (degC)").sensor == "T (degC)"
    assert r.for_sensor("nope") is None


def test_dense_retriever_semantic_paraphrase_if_available():
    pytest.importorskip("sentence_transformers")
    r = rag.dense_retriever()
    if not getattr(r, "semantic", False):
        pytest.skip("sentence-transformers model weights not available")
    # the payoff: a paraphrase with NO shared KB words still retrieves the right sensor
    hits = [s.sensor for s in r.retrieve("how hot does it get outside", k=3)]
    assert "T (degC)" in hits


def test_dense_retriever_falls_back_to_tfidf_interface():
    # dense_retriever must always return a usable retriever (same interface), even offline
    r = rag.dense_retriever()
    hits = r.retrieve("relative humidity percent", k=1)
    assert hits and hits[0].sensor == "rh (%)"
    assert r.for_sensor("T (degC)").sensor == "T (degC)"


def test_tfidf_retriever_finds_relevant_doc():
    r = rag.Retriever()
    hits = r.retrieve("relative humidity percent", k=1)  # TF-IDF is literal: use KB wording
    assert hits and hits[0].sensor == "rh (%)"


def test_lsa_retriever_finds_relevant_doc():
    r = rag.LSARetriever()
    assert r.n_components >= 1
    hits = r.retrieve("relative humidity percent", k=1)   # LSA over TF-IDF, in latent space
    assert hits and hits[0].sensor == "rh (%)"
    hits2 = r.retrieve("air density", k=1)
    assert hits2 and hits2[0].sensor == "rho (g/m**3)"


def test_lsa_retriever_for_sensor_direct_lookup():
    r = rag.LSARetriever()
    assert r.for_sensor("T (degC)").sensor == "T (degC)"
    assert r.for_sensor("not a sensor") is None


def test_lsa_retriever_is_deterministic():
    a = [s.sensor for s in rag.LSARetriever().retrieve("wind speed", k=3)]
    b = [s.sensor for s in rag.LSARetriever().retrieve("wind speed", k=3)]
    assert a == b and a[0] == "wv (m/s)"


# A tiny deterministic "embedding": bag-of-words counts over a fixed vocabulary. Cosine over
# these vectors gives a real semantic-style ranking without needing a model.
_VOCAB = ("temperature", "humidity", "pressure", "wind", "density", "dew")


def _fake_embed(text: str):
    t = text.lower()
    return [float(t.count(w)) for w in _VOCAB]


def test_embedding_retriever_ranks_by_cosine():
    r = rag.EmbeddingRetriever(embed_fn=_fake_embed)
    assert r.semantic is True
    hits = r.retrieve("air density", k=1)
    assert hits and hits[0].sensor == "rho (g/m**3)"      # density is a discriminating token
    hits2 = r.retrieve("dew point", k=1)
    assert hits2 and hits2[0].sensor == "Tdew (degC)"


def test_embedding_retriever_falls_back_when_backend_missing():
    def dead_embed(text):
        raise OSError("no ollama")

    r = rag.EmbeddingRetriever(embed_fn=dead_embed)
    assert r.semantic is False                          # backend unavailable
    hits = r.retrieve("relative humidity percent", k=1)  # still works, via TF-IDF fallback
    assert hits and hits[0].sensor == "rh (%)"


def test_embedding_retriever_for_sensor_direct_lookup():
    r = rag.EmbeddingRetriever(embed_fn=_fake_embed)
    snip = r.for_sensor("T (degC)")
    assert snip is not None and snip.sensor == "T (degC)"
    assert r.for_sensor("not a sensor") is None


def test_fake_embed_matrix_is_unit_normalised():
    # guards the internal normalisation (unit rows -> dot product == cosine). A doc with no
    # vocabulary overlap embeds to the zero vector (norm 0); every other row is unit length.
    r = rag.EmbeddingRetriever(embed_fn=_fake_embed)
    norms = np.linalg.norm(r._matrix, axis=1)
    assert np.all((norms < 1e-6) | np.isclose(norms, 1.0))
