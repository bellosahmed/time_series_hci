"""RAG + local LLM: turn computed results into natural-language explanations, offline.

Responsibilities (Phase 6):
- maintain a local vector store (FAISS/Chroma) of sensor descriptions, units, known behaviours
- run a local LLM (Ollama) fully offline
- orchestrate: query_engine computes -> RAG retrieves context -> LLM phrases the answer

GUARDRAIL: the LLM is given the computed numbers; it must never compute them itself.
"""

from __future__ import annotations


def retrieve_context(question: str, k: int = 3) -> list[str]:
    """Retrieve top-k context snippets from the local vector store. TODO."""
    raise NotImplementedError


def explain(question: str, computed_result: dict, context: list[str]) -> str:
    """Ask the local LLM to explain a computed result using retrieved context. TODO."""
    raise NotImplementedError


def answer(question: str) -> str:
    """End-to-end: route question -> query engine -> retrieve context -> explain. TODO."""
    raise NotImplementedError
