"""
v2 local LLM layer (optional) — talks to Ollama over HTTP using only the standard library.

Hard rule: the model may ONLY rephrase numbers it is given. It never computes or invents a value.
If Ollama isn't running, every function degrades gracefully (the interface falls back to templates),
so the system works fully offline with or without the LLM.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"
SYSTEM = (
    "You explain pre-computed EU trade statistics to a user. "
    "You are given exact facts (already computed by code). "
    "Rephrase them in 1-2 short, clear sentences. "
    "NEVER change, round differently, or invent any number, country, or date. "
    "If a cause is given as 'inferred', present it as a possibility, not a fact."
)


def is_available(timeout: float = 1.0) -> bool:
    """True if an Ollama server answers locally."""
    try:
        urllib.request.urlopen(f"{HOST}/api/tags", timeout=timeout)
        return True
    except Exception:
        return False


def list_models(timeout: float = 2.0) -> list[str]:
    try:
        with urllib.request.urlopen(f"{HOST}/api/tags", timeout=timeout) as r:
            data = json.load(r)
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def pick_model(prefer: str = "llama3.2") -> str | None:
    """Choose an installed model, preferring llama3.2* (handles tags like 'llama3.2:3b')."""
    models = list_models()
    if not models:
        return None
    for m in models:
        if m.startswith(prefer):
            return m
    for m in models:
        if m.startswith("llama"):
            return m
    return models[0]


def generate(prompt: str, model: str = DEFAULT_MODEL, system: str = SYSTEM,
             timeout: float = 30.0) -> str:
    """Single-turn generation. Raises on failure (callers handle the fallback)."""
    body = json.dumps({"model": model, "prompt": prompt, "system": system,
                       "stream": False}).encode()
    req = urllib.request.Request(f"{HOST}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["response"].strip()


def explain(question: str, computed: str, inferred: str | None = None,
            model: str = DEFAULT_MODEL, timeout: float = 30.0) -> str | None:
    """Ask the LLM to phrase the computed facts. Returns None if the LLM is unavailable."""
    if not is_available():
        return None
    # if the requested model isn't installed, fall back to any available one
    if model not in list_models():
        model = pick_model() or model
    parts = [f"User question: {question}", f"Computed facts (do not change): {computed}"]
    if inferred:
        parts.append(f"Inferred context (a possibility): {inferred}")
    parts.append("Explain the facts to the user in 1-2 sentences.")
    try:
        return generate("\n".join(parts), model=model, timeout=timeout)
    except Exception:
        return None
