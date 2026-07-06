"""Local LLM explanation via Ollama — the *explanation only* layer.

It rewords facts that were ALREADY computed by the query engine; it must never produce or alter a
number. The `faithfulness` gate enforces this after the fact by comparing the numbers in the facts vs
the generated prose. Talks to a local Ollama server over the stdlib; if it isn't running,
`is_available()` is False and callers fall back to the deterministic template.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = os.environ.get("DAD_MODEL", "llama3.2:3b")

SYSTEM_PROMPT = (
    "You are the explanation layer of an IoT-network security analytics system. You are given facts "
    "COMPUTED by deterministic code about MQTT sensor traffic and labelled attacks in a data centre. "
    "Rephrase them into one or two clear, friendly sentences for a non-expert. RULES: never invent, "
    "change, round, or add any number, count, IP address, or temperature; never contradict the given "
    "facts; if a value is not provided, do not guess it. Explain only what is given."
)

FEWSHOT = """Examples of the style to copy (keep every number, IP and temperature exactly):

Computed fact: There were 104 labelled attack packets on Wednesday.
Answer: On Wednesday the system flagged 104 attack packets.

Computed fact: The attack traffic comes from 10.6.56.41.
Answer: All the malicious traffic came from a single host, 10.6.56.41.
"""


def is_available(timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


def list_models() -> list[str]:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=2.0) as r:
            data = json.loads(r.read())
        return [m["name"] for m in data.get("models", [])]
    except (urllib.error.URLError, OSError, KeyError, json.JSONDecodeError):
        return []


def generate(prompt: str, model: str = DEFAULT_MODEL, system: str = SYSTEM_PROMPT,
             timeout: float = 60.0, max_tokens: int = 400,
             temperature: float = 0.2, seed: int = 0) -> str:
    """Send a prompt to the local Ollama model; low temp + fixed seed → reproducible wording."""
    payload = json.dumps({
        "model": model, "prompt": prompt, "system": system, "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens, "seed": seed},
    }).encode("utf-8")
    req = urllib.request.Request(f"{OLLAMA_URL}/api/generate", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()).get("response", "").strip()


def explain(question: str, computed: str, context: str | None = None,
            inferred: str | None = None, model: str = DEFAULT_MODEL) -> str:
    """Ask the local LLM to phrase a computed answer. Raises if Ollama is unavailable."""
    if not is_available():
        raise RuntimeError("Ollama is not running; use the deterministic template instead.")
    facts = [f"Question: {question}", f"Computed fact (exact, do not change): {computed}"]
    if context:
        facts.append(f"Background context: {context}")
    if inferred:
        facts.append(f"Reasoning: {inferred}")
    prompt = (FEWSHOT + "\nNow write the Answer for this one, in the same style:\n"
              + "\n".join(facts) + "\n\nAnswer:")
    return generate(prompt, model=model)


# ---- faithfulness gate (offline; the core guarantee) ----
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_CLOCK = re.compile(r"\b\d{1,2}:\d{2}\b")


def _guarded(v: float) -> bool:
    """Enforce decimals (measured values) and integers with magnitude > 12 (counts, years, IP octets
    like 56/41). Small integers ≤ 12 are tolerated (they overlap clock hours / small octets), a known
    residual gap in the check."""
    return v != int(v) or abs(v) > 12


def faithfulness(facts: str, explanation: str, tol: float = 0.1) -> dict:
    """Did the LLM alter a guarded number? Rounding is allowed; clock times are stripped first.

    Returns {ok, dropped, invented}: `invented` (a guarded number in the prose matching no source
    value) is the serious failure — a hallucinated figure; `dropped` (a source value omitted) is benign.
    """
    facts = _CLOCK.sub(" ", facts or "")
    explanation = _CLOCK.sub(" ", explanation or "")
    src = [float(x) for x in _NUMBER.findall(facts)]
    exp = [float(x) for x in _NUMBER.findall(explanation)]

    def near(v, pool):
        if v == int(v):
            return any(abs(v - s) < 0.5 for s in pool)
        return any(abs(v - s) <= tol + 0.02 * abs(s) for s in pool)

    invented = sorted({round(v, 3) for v in exp if _guarded(v) and not near(v, src)})
    dropped = sorted({round(v, 3) for v in src if _guarded(v) and not near(v, exp)})
    return {"ok": not invented and not dropped, "dropped": dropped, "invented": invented}
