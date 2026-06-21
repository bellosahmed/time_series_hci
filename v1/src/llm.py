"""Local LLM explanation via Ollama (the final 'explanation' step).

This is the *explanation only* layer. It takes facts that have ALREADY been
computed by the query engine and rephrases them in natural language. It must
never produce or alter numbers — the prompt enforces this, and the computed
facts are passed in verbatim.

No third-party dependency: it talks to the local Ollama HTTP server
(http://localhost:11434) via the standard library. If Ollama isn't installed or
running, `is_available()` returns False and callers fall back to the deterministic
template in `rag_system.explain`.

Setup (run once, locally):
    1. Install Ollama:  https://ollama.com/download   (or `curl -fsSL https://ollama.com/install.sh | sh`)
    2. Pull a small model:  `ollama pull llama3.2`
    3. Ensure it's serving:  `ollama serve`  (usually automatic)
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"

SYSTEM_PROMPT = (
    "You are the explanation layer of a time-series analytics system. "
    "You are given facts that were COMPUTED by deterministic code. "
    "Rephrase them into one or two clear, friendly sentences for a non-expert. "
    "RULES: never invent, change, round, or add any number; never contradict the "
    "given facts; if a value is not provided, do not guess it. Explain only what is given."
)


def is_available(timeout: float = 1.0) -> bool:
    """True if a local Ollama server is reachable."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


def list_models() -> list[str]:
    """Names of models currently pulled into the local Ollama."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=2.0) as r:
            data = json.loads(r.read())
        return [m["name"] for m in data.get("models", [])]
    except (urllib.error.URLError, OSError, KeyError, json.JSONDecodeError):
        return []


def generate(prompt: str, model: str = DEFAULT_MODEL, system: str = SYSTEM_PROMPT,
             timeout: float = 60.0) -> str:
    """Send a prompt to the local Ollama model and return its text reply."""
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": 0.2},   # low temp = faithful rephrasing
    }).encode("utf-8")
    req = urllib.request.Request(f"{OLLAMA_URL}/api/generate", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()).get("response", "").strip()


def explain(question: str, computed: str, context: str | None = None,
            inferred: str | None = None, model: str = DEFAULT_MODEL) -> str:
    """Ask the local LLM to phrase a computed answer. Raises if Ollama is unavailable.

    Mirrors `rag_system.explain` but produces natural prose instead of a template.
    The numbers come only from `computed` (and `inferred`); the model just rewords.
    """
    if not is_available():
        raise RuntimeError("Ollama is not running. Start it with `ollama serve` and "
                           "`ollama pull llama3.2`, or use rag_system.explain (template).")
    facts = [f"Question: {question}", f"Computed fact (exact, do not change): {computed}"]
    if context:
        facts.append(f"Background context: {context}")
    if inferred:
        facts.append(f"Reasoning: {inferred}")
    prompt = "\n".join(facts) + "\n\nAnswer the question using only these facts:"
    return generate(prompt, model=model)
