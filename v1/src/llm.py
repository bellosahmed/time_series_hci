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
import re
import urllib.error
import urllib.request

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:3b"

SYSTEM_PROMPT = (
    "You are the explanation layer of a time-series analytics system. "
    "You are given facts that were COMPUTED by deterministic code. "
    "Rephrase them into one or two clear, friendly sentences for a non-expert. "
    "RULES: never invent, change, round, or add any number; never contradict the "
    "given facts; if a value is not provided, do not guess it. "
    "Do NOT add any qualifier that is not in the facts — no seasons, months, weather "
    "descriptions, compass directions, or cause/effect claims unless they appear verbatim "
    "in the facts (e.g. never call a July date 'winter'). Explain only what is given."
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
             timeout: float = 60.0, max_tokens: int = 512,
             temperature: float = 0.2, seed: int = 0) -> str:
    """Send a prompt to the local Ollama model and return its text reply.

    `max_tokens` (Ollama's `num_predict`) is a generous **safety ceiling**, not a tight
    speed limit: it only stops pathological runaway generation. It's set well above what
    a 1-2 sentence explanation or short plan needs, so it never clips a wanted answer —
    correctness/completeness comes first.

    `temperature` (low) and a fixed `seed` make generation **reproducible**: the same facts
    yield the same phrasing across runs (given the same model build), which is what tames the
    small model's stochasticity. The numbers are already guaranteed exact upstream (the query
    engine) and checked downstream (`faithfulness`); this only stabilises the wording.
    """
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        # low temp + fixed seed => reproducible; length cap stops runaway generation
        "options": {"temperature": temperature, "num_predict": max_tokens, "seed": seed},
    }).encode("utf-8")
    req = urllib.request.Request(f"{OLLAMA_URL}/api/generate", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()).get("response", "").strip()


CLASSIFY_SYSTEM = (
    "You convert a user's question about sensor data into a structured query. "
    "Respond with ONLY compact JSON, no prose, no explanation."
)


def classify(question: str, intents: list[str], sensors: list[str],
             model: str = DEFAULT_MODEL) -> dict | None:
    """Use the LLM to map free text -> {intent, sensors}, choosing from allowed lists.

    The LLM only *classifies*; the query engine still computes the answer. Returns
    None if Ollama is unavailable or the reply can't be parsed/validated.
    """
    if not is_available():
        return None
    prompt = (
        f"Question: {question}\n"
        f"Allowed intents: {intents}\n"
        f"Allowed sensor columns: {sensors}\n"
        "Everyday words map to columns: temperature->T (degC), humidity->rh (%), "
        "pressure->p (mbar), density->rho (g/m**3), wind->wv (m/s), dew->Tdew (degC).\n"
        'Return JSON only: {"intent": <one allowed intent or null>, '
        '"sensors": [<zero or more allowed columns>]}'
    )
    try:
        out = generate(prompt, model=model, system=CLASSIFY_SYSTEM, timeout=30.0)
        match = re.search(r"\{.*\}", out, re.S)
        if not match:
            return None
        data = json.loads(match.group(0))
    except (urllib.error.URLError, OSError, json.JSONDecodeError, ValueError):
        return None
    intent = data.get("intent")
    if intent not in intents:
        intent = None
    valid_sensors = [s for s in data.get("sensors", []) if s in sensors]
    return {"intent": intent, "sensors": valid_sensors}


REWRITE_SYSTEM = (
    "You rewrite a user's short follow-up message into ONE self-contained question about a weather "
    "time-series dataset, using the recent conversation only for context. You DO NOT answer it, and "
    "you must NEVER output any number, statistic, temperature, count, or measurement. If the message "
    "is already self-contained, return it unchanged. Reply with ONLY the rewritten question."
)


def rewrite_followup(question: str, history: list[dict], model: str = DEFAULT_MODEL) -> str | None:
    """Rewrite an elliptical follow-up into a standalone question, using recent turns.

    ``history`` is a list of recent ``{"question", "computed"}`` turns (oldest→newest). The model
    only resolves *language and context* — e.g. after "hottest in 2013?", the follow-up
    "what about 2014?" becomes "hottest in 2014?" — so the deterministic query engine can then
    compute the exact answer. The model never produces a number itself. Returns the rewritten
    question, or None if Ollama is unavailable, there is no history, or the reply is empty.
    """
    if not is_available() or not history:
        return None
    convo = "\n".join(f"User: {h.get('question', '')}\nAnswer: {h.get('computed', '')}"
                      for h in history[-3:])
    prompt = (f"Recent conversation:\n{convo}\n\nUser's new message: {question}\n\n"
              "Rewrite it as one standalone question.")
    try:
        out = generate(prompt, model=model, system=REWRITE_SYSTEM, timeout=30.0)
    except (urllib.error.URLError, OSError):
        return None
    out = (out or "").strip().strip('"').strip()
    return out.splitlines()[0].strip() if out else None


PLAN_SYSTEM = (
    "You translate a user's question about a weather time-series into a STRUCTURED query plan. "
    "You DO NOT answer the question and you must NEVER output any data value, statistic, "
    "temperature, count, or measurement from the data. You only describe WHAT to compute. "
    "Respond with ONLY compact JSON, no prose."
)


def plan(question: str, intents: list[str], sensors: list[str],
         model: str = DEFAULT_MODEL) -> dict | None:
    """Use the LLM to turn free-form text into {intent, sensors, start, end, period, op, value}.

    The model only *interprets language* (including seasons, ranges, and loose phrasing)
    into a plan; the query engine still computes every number. Returns None if Ollama is
    unavailable or the reply can't be parsed. All fields are validated by the caller.
    """
    if not is_available():
        return None
    prompt = (
        f"Question: {question}\n"
        f"Allowed intents: {intents}\n"
        f"Allowed sensor columns: {sensors}\n"
        "Everyday words map to columns: temperature->T (degC), humidity->rh (%), "
        "pressure->p (mbar), density->rho (g/m**3), wind->wv (m/s), dew->Tdew (degC).\n"
        "The data is hourly weather covering 2009-01 to 2016-12.\n"
        "If the question implies a time window (a year, month, season, or range), express it "
        "as start and end calendar dates (YYYY-MM-DD). Seasons: spring=Mar-May, summer=Jun-Aug, "
        "autumn/fall=Sep-Nov, winter=Dec-Feb.\n"
        "If the user compares against a numeric value, set op ('>' for above/hotter/more, "
        "'<' for below/colder/less) and value (the number).\n"
        'Return JSON ONLY: {"intent": <one allowed intent or null>, '
        '"sensors": [<zero or more allowed columns>], '
        '"start": <YYYY-MM-DD or null>, "end": <YYYY-MM-DD or null>, '
        '"period": <short human label like "summer 2012" or null>, '
        '"op": <">" or "<" or null>, "value": <number or null>}'
    )
    try:
        out = generate(prompt, model=model, system=PLAN_SYSTEM, timeout=30.0)
        match = re.search(r"\{.*\}", out, re.S)
        if not match:
            return None
        data = json.loads(match.group(0))
    except (urllib.error.URLError, OSError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    intent = data.get("intent")
    op = data.get("op")
    value = data.get("value")
    return {
        "intent": intent if intent in intents else None,
        "sensors": [s for s in data.get("sensors", []) if s in sensors],
        "start": data.get("start"),
        "end": data.get("end"),
        "period": data.get("period"),
        "op": op if op in (">", "<") else None,
        "value": float(value) if isinstance(value, (int, float)) else None,
    }


# Few-shot examples teach the model the desired style WITHOUT fine-tuning: keep every
# number exact, be clear and friendly, and turn multi-sensor reasoning into a plain story
# (real event vs. faulty sensor). Cheaper, safer, and more controllable than training.
FEWSHOT = """Examples of the style to copy (keep every number and date exactly):

Computed fact: max T (degC) was 37.04 on 4 Jul 2015 16:00.
Answer: The hottest it ever got was 37.04°C, on 4 July 2015 at 4pm.

Computed fact: T (degC) was above 35 18 times (0.03% of readings), first on 19 Aug 2012 15:00.
Answer: It climbed above 35°C only 18 times — about 0.03% of all readings — the first on 19 August 2012 at 3pm.

Computed fact: 66 anomalous T (degC) readings in 2014 in 24 windows.
Reasoning: At 26 Jan 2014 14:00: Tdew (degC) unusually low (-2.9σ); T (degC) unusually low (-2.7σ); rho (g/m**3) unusually high (+2.5σ).
Answer: There were 66 unusual temperature readings in 2014, falling into 24 separate spells. The clearest was on 26 January 2014 at 2pm: temperature and dew point both dropped sharply while air density rose together — a physically consistent cold snap, not a faulty sensor.
"""


_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_CLOCK = re.compile(r"\b\d{1,2}:\d{2}\b")


def _guarded(v: float) -> bool:
    """Which numbers the faithfulness check enforces.

    Decimals (measured values) and integers with magnitude > 12 (counts like "24 windows",
    years like 2015) are enforced. Small integers ≤ 12 are left tolerant: they overlap clock
    hours (after a 12→AM/PM reformat), month numbers and single-digit counts, so enforcing
    them would false-flag legitimate rephrasing ("16:00" → "4pm"). This residual gap (a
    fabricated count ≤ 12) is a known limitation of the check.
    """
    return v != int(v) or abs(v) > 12


def faithfulness(facts: str, explanation: str, tol: float = 0.1) -> dict:
    """Measure whether the LLM altered the *measured values*.

    Compares the numbers in the source facts vs the explanation, with a numeric tolerance so
    **rounding is allowed** ("34.93" → "34.9"). Enforces decimals *and* larger integers (counts,
    years) — so a hallucinated count ("87 windows") or wrong year is caught — while tolerating
    legitimate reformatting: clock times (`16:00` → `4pm`) are stripped first, and small integers
    (≤ 12) are not enforced (see ``_guarded``). Returns ``{ok, dropped, invented}``:
    - **invented** = an enforced number in the explanation that matches *no* source value → the
      serious failure (a hallucinated figure).
    - **dropped** = a source value omitted from the explanation → benign (summarising).
    """
    facts = _CLOCK.sub(" ", facts or "")
    explanation = _CLOCK.sub(" ", explanation or "")
    src = [float(x) for x in _NUMBER.findall(facts)]
    exp = [float(x) for x in _NUMBER.findall(explanation)]

    def near(v, pool):
        # Integers (counts, years) must match near-exactly — the 2% relative slop is for
        # rounding *decimals* ("34.93" → "34.9"), and would wrongly accept 2016 ≈ 2015.
        if v == int(v):
            return any(abs(v - s) < 0.5 for s in pool)
        return any(abs(v - s) <= tol + 0.02 * abs(s) for s in pool)

    invented = sorted({round(v, 3) for v in exp if _guarded(v) and not near(v, src)})
    dropped = sorted({round(v, 3) for v in src if _guarded(v) and not near(v, exp)})
    return {"ok": not invented and not dropped, "dropped": dropped, "invented": invented}


def explain(question: str, computed: str, context: str | None = None,
            inferred: str | None = None, model: str = DEFAULT_MODEL) -> str:
    """Ask the local LLM to phrase a computed answer. Raises if Ollama is unavailable.

    Mirrors `rag_system.explain` but produces natural prose instead of a template.
    The numbers come only from `computed` (and `inferred`); the model just rewords.
    Few-shot examples steer the style (a lighter, safer alternative to fine-tuning).
    """
    if not is_available():
        raise RuntimeError("Ollama is not running. Start it with `ollama serve` and "
                           "`ollama pull llama3.2`, or use rag_system.explain (template).")
    facts = [f"Question: {question}", f"Computed fact (exact, do not change): {computed}"]
    if context:
        facts.append(f"Background context: {context}")
    if inferred:
        facts.append(f"Reasoning: {inferred}")
    prompt = (FEWSHOT + "\nNow write the Answer for this one, in the same style:\n"
              + "\n".join(facts) + "\n\nAnswer:")
    return generate(prompt, model=model)
