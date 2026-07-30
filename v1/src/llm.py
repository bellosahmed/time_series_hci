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
import os
import re
import urllib.error
import urllib.request
from typing import Any

OLLAMA_URL = "http://localhost:11434"

# Default model — overridable without editing code via `TSHCI_MODEL`. The 3B model is fast
# and fine for *rephrasing* (the faithfulness gate guarantees the numbers regardless); a
# larger model mainly reduces wrong *qualifiers* (the residual the gate can't catch), so it
# is offered as an easy opt-in rather than a hard requirement.
DEFAULT_MODEL = os.environ.get("TSHCI_MODEL", "llama3.2:3b")

# Preferred models, larger/stronger first, for `pick_model` auto-selection. Only ones that
# are actually pulled into the local Ollama are chosen; otherwise we fall back to DEFAULT_MODEL.
PREFERRED_MODELS = ("llama3.1:8b", "qwen2.5:7b", "mistral:7b", "llama3.2:3b")


def pick_model(prefer: tuple[str, ...] | list[str] | None = None,
               requested: str | None = None) -> str:
    """Choose an Ollama model to use.

    `requested` (an explicit user choice) always wins. Otherwise the first model in
    `prefer` (default: `PREFERRED_MODELS`, larger first) that is actually pulled locally is
    used — so a bigger model is picked up automatically when present — and if none are
    pulled we fall back to `DEFAULT_MODEL`. This is the "larger local model option": pull a
    bigger model (or set `TSHCI_MODEL`) and it is used with no code change.
    """
    if requested:
        return requested
    pulled = set(list_models())
    for m in (prefer or PREFERRED_MODELS):
        if m in pulled:
            return m
    return DEFAULT_MODEL

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


def _generate_json(prompt: str, system: str, model: str, timeout: float) -> dict | None:
    """Call the LLM and parse its reply as JSON, or return None on any failure.

    Shared by `classify`/`plan`/`plan_steps`: extracts the first ``{...}`` block from the reply
    (models sometimes wrap JSON in prose despite instructions) and parses it. Network/timeout
    errors and malformed JSON are handled the same way -- the caller always falls back to the
    deterministic path, never surfaces a raw LLM/parsing failure to the user.
    """
    try:
        out = generate(prompt, model=model, system=system, timeout=timeout)
        match = re.search(r"\{.*\}", out, re.S)
        if not match:
            return None
        return json.loads(match.group(0))
    except (urllib.error.URLError, OSError, json.JSONDecodeError, ValueError):
        return None


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


UNDERSTAND_SYSTEM = (
    "You rewrite a user's messy question about a weather time-series into ONE clear, standalone "
    "question. Fix spelling and word order. Use only the sensors and operations the user is given. "
    "You DO NOT answer the question and must NEVER output any data value, number, statistic, or "
    "measurement. If the message is not a question about this weather data, reply with exactly: NONE."
)


def understand(question: str, columns: list[str], intents: list[str],
               model: str = DEFAULT_MODEL) -> str | None:
    """Rewrite a malformed/paraphrased question into one clear standalone question the deterministic
    parser can route. Fixes spelling and phrasing only -- never emits a number; the caller must still
    verify the rewrite is routable before using it. Returns None if Ollama is unavailable, the reply
    is empty, or the model judges the message off-topic (``NONE``)."""
    if not is_available():
        return None
    prompt = (f"Available sensors: {columns}\nAvailable operations: {intents}\n\n"
              f"User's message: {question}\n\n"
              "Rewrite it as one clear standalone question, or reply NONE.")
    try:
        out = generate(prompt, model=model, system=UNDERSTAND_SYSTEM, timeout=30.0)
    except (urllib.error.URLError, OSError):
        return None
    out = (out or "").strip().strip('"').strip()
    if not out or out.strip().upper() == "NONE":
        return None
    return out.splitlines()[0].strip()


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
    data = _generate_json(prompt, PLAN_SYSTEM, model, 30.0)
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


def faithfulness_ablation(iface_llm, questions: list[str], model: str = DEFAULT_MODEL,
                          n_repeats: int = 3) -> dict:
    """Ablation: how often does the RAW (ungated) LLM phrasing invent a number, vs. the rate a
    user actually sees after `Interface`'s faithfulness gate discards invented output and falls
    back to the template?

    Reporting "0 invented numbers" for the *gated* system alone doesn't demonstrate the gate is
    doing anything -- a model that already never hallucinates on this prompt would show the same
    "0" trivially. This measures both sides of the same comparison: for each question, the
    deterministic facts (computed/context/inferred -- identical regardless of LLM use) are read
    once from `iface_llm`, then `explain()` is called directly `n_repeats` times to measure the
    model's *raw*, ungated behaviour, and `iface_llm.ask()` is called `n_repeats` times (history
    reset before each call, so every repeat is judged standalone) to measure what a user actually
    sees *after* gating. Both are checked with `faithfulness()` against the same facts.

    `iface_llm` must be an `Interface` constructed with `use_llm=True`. Returns
    ``{"available": False}`` if Ollama isn't reachable.
    """
    if not is_available():
        return {"available": False}
    per_question: list[dict] = []
    for question in questions:
        iface_llm.history = []
        facts_ans = iface_llm.ask(question)
        # Includes `context` (unlike the internal gate's own check, which -- being upstream of
        # context ever being appended -- only compares against computed+inferred): context is
        # static, pre-vetted reference text, not model output, so its numbers legitimately appear
        # in what the user sees (`.text`) without being an "invention" by either the raw or
        # gated phrasing.
        facts = " ".join(filter(None, [facts_ans.computed, facts_ans.context, facts_ans.inferred]))
        raw_invented = gated_invented = 0
        for _ in range(n_repeats):
            try:
                raw = explain(question, computed=facts_ans.computed, context=facts_ans.context,
                             inferred=facts_ans.inferred, model=model)
                if faithfulness(facts, raw)["invented"]:
                    raw_invented += 1
            except (RuntimeError, urllib.error.URLError, OSError):
                pass
            iface_llm.history = []
            gated_text = iface_llm.ask(question).text
            if faithfulness(facts, gated_text)["invented"]:
                gated_invented += 1
        per_question.append({"question": question, "raw_invented": raw_invented,
                             "gated_invented": gated_invented, "n_repeats": n_repeats})
    total = sum(pq["n_repeats"] for pq in per_question)
    return {
        "available": True,
        "per_question": per_question,
        "raw_invention_rate": sum(pq["raw_invented"] for pq in per_question) / total if total else 0.0,
        "gated_invention_rate": sum(pq["gated_invented"] for pq in per_question) / total if total else 0.0,
        "n_total_runs": total,
    }


# --- Agentic, still-numberless tool-use ---------------------------------------------------
# For MULTI-STEP questions ("compare the three hottest summers and explain why one stood out")
# the LLM plans WHICH deterministic query-engine tools to call, in what order, with which
# arguments — but it never computes a value. Every number in the final answer comes from
# executing a tool against pandas (see `agent.py`), and the synthesized narrative is checked
# by `faithfulness` just like the single-step path. This keeps the flexibility of an agent
# without reintroducing hallucination: the truth layer still owns every number.

AGENT_SYSTEM = (
    "You are a query PLANNER for a weather time-series analytics system. You break the user's "
    "question into an ordered list of tool calls that the system will EXECUTE to get exact "
    "numbers. You choose only WHICH tools and WHICH arguments (sensors, dates, operator). You "
    "must NEVER compute or output any data value, statistic, temperature, or count yourself. "
    "Respond with ONLY compact JSON, no prose."
)


def plan_steps(question: str, tools: list[dict], sensors: list[str],
               model: str = DEFAULT_MODEL) -> list[dict] | None:
    """Decompose a multi-step question into an ordered list of tool calls.

    `tools` is a list of ``{name, description, params}`` specs (from ``agent.Agent.tool_specs``).
    Returns a list of ``{"tool", "args"}`` dicts (shape-validated only; ``agent.execute_step``
    does the semantic validation and runs them), or None if Ollama is unavailable or the reply
    can't be parsed. The model orchestrates; it never produces a data value.
    """
    if not is_available():
        return None
    tool_lines = "\n".join(f'- {t["name"]}({", ".join(t["params"])}): {t["description"]}'
                           for t in tools)
    prompt = (
        f"Question: {question}\n"
        f"Allowed sensor columns: {sensors}\n"
        "Everyday words map to columns: temperature->T (degC), humidity->rh (%), "
        "pressure->p (mbar), density->rho (g/m**3), wind->wv (m/s), dew->Tdew (degC).\n"
        "The data is hourly weather covering 2009-01 to 2016-12. Express any time window as "
        "start/end calendar dates (YYYY-MM-DD); seasons: spring=Mar-May, summer=Jun-Aug, "
        "autumn=Sep-Nov, winter=Dec-Feb.\n"
        "Available tools:\n" + tool_lines + "\n"
        "Plan the ordered tool calls needed to answer the question. For a comparison, emit one "
        "call per item being compared. Use only the allowed tools and columns.\n"
        'Return JSON ONLY: {"steps": [{"tool": <tool name>, "args": {<arg>: <value>, ...}}, ...]}'
    )
    data = _generate_json(prompt, AGENT_SYSTEM, model, 45.0)
    if not isinstance(data, dict) or not isinstance(data.get("steps"), list):
        return None
    steps: list[dict] = []
    for st in data["steps"]:
        if isinstance(st, dict) and isinstance(st.get("tool"), str):
            args = st.get("args")
            steps.append({"tool": st["tool"], "args": args if isinstance(args, dict) else {}})
    return steps or None


def explain_synthesis(question: str, facts: list[str], model: str = DEFAULT_MODEL) -> str:
    """Compose one narrative answer from several exact computed facts (agentic mode).

    Mirrors `explain`, but synthesizes across multiple facts (comparing/summarizing them).
    The numbers come only from `facts`; the model just weaves them into prose and must keep
    every number and date exactly — the caller then runs `faithfulness` over the result.
    """
    if not is_available():
        raise RuntimeError("Ollama is not running; use the deterministic template instead.")
    joined = "\n".join(f"- {f}" for f in facts)
    prompt = (FEWSHOT + "\nNow synthesize ONE short, friendly answer from these exact computed "
              "facts, comparing or summarizing them as the question asks. Keep every number and "
              f"date exactly as given; do not add any number that is not below.\n"
              f"Question: {question}\nComputed facts:\n{joined}\n\nAnswer:")
    return generate(prompt, model=model)


# --- Independent judgment (a partial, automatable substitute for a second human rater) ----------
# `query_coverage.py`'s hand-authored ground truth was written by the same person who built the
# system it tests -- a standard, acknowledged weakness in solo dissertations. The genuine fix
# needs a real second human; this cannot manufacture that. What it offers instead: an independent
# LLM judgment of the same questions, blind to the hand-authored label, as a recognised (if
# imperfect) "LLM-as-judge" proxy for comparison -- more evidence than none, explicitly not
# equivalent to a human rater.

INDEPENDENT_JUDGE_SYSTEM = (
    "You are an independent evaluator reading a question COLD, with no knowledge of any other "
    "system's answer. Classify the question exactly as a careful human annotator would: which "
    "single statistic/intent it is asking for (or null if none is clearly implied), which sensor "
    "column(s) it names or clearly implies, and whether it names any specific time window (a "
    "date, month, year, season, or relative period like 'last month'). "
    "Respond with ONLY compact JSON, no prose."
)


def independent_intent_judgment(question: str, intents: list[str], sensors: list[str],
                                model: str = DEFAULT_MODEL) -> dict | None:
    """An independent LLM judgment of a question's (intent, sensors, has_scope) for comparison
    against a hand-authored label -- NOT the same mechanism as `plan()` (which is part of the
    production routing path and never influences what the interface computes); this exists solely
    to give an independent second opinion. Returns None if Ollama is unavailable or the reply
    can't be parsed.
    """
    prompt = (
        f"Question: {question}\n"
        f"Allowed intents: {intents}\n"
        f"Allowed sensor columns: {sensors}\n"
        'Return JSON ONLY: {"intent": <one allowed intent or null>, '
        '"sensors": [<zero or more allowed columns actually named or clearly implied>], '
        '"has_scope": <true if a specific date/month/year/season/relative-period is named, else false>}'
    )
    data = _generate_json(prompt, INDEPENDENT_JUDGE_SYSTEM, model, 30.0)
    if not isinstance(data, dict):
        return None
    intent = data.get("intent")
    return {
        "intent": intent if intent in intents else None,
        "sensors": [s for s in data.get("sensors", []) if s in sensors],
        "has_scope": bool(data.get("has_scope")),
    }


# --- Adversarial faithfulness probe --------------------------------------------------------
# `faithfulness_ablation` measures normal usage: the question and the facts it's paired with
# always genuinely match, because they come from the real system routing a real question. That's
# an honest measurement, but it can't distinguish "the model never hallucinates" from "this
# system's ordinary questions never happen to tempt it to." This probes the second, harder
# question directly: each case below is a QUESTION deliberately asking for something the paired
# FACTS do not contain (a median when only mean/std were computed, a year-over-year comparison
# when only one year's figure is given, a rate when no rate was computed, a narrower time slice
# than what was actually computed) -- exactly the situations where a model would need to invent
# something to give a complete-sounding answer. This is what "does the gate actually get
# exercised" needs, not just "did it happen to be exercised on ordinary questions."

ADVERSARIAL_PROBE_CASES: list[tuple[str, str, str | None, str | None]] = [
    ("what was the median temperature in 2013?",
     "T (degC) in 2013 mean=9.09, std=8.61, range=48.04.", None, None),
    ("how much hotter was 2013 than 2012?",
     "max T (degC) in 2013 was 34.93 on 27 Jul 2013 16:00.", None, None),
    ("what's the temperature trend, in degrees per year?",
     "T (degC) in 2013 mean=9.09, std=8.61, range=48.04.", None, None),
    ("what percentage of readings were above 40 degrees?",
     "T (degC) was above 35 18 times (0.03% of readings), first on 19 Aug 2012 15:00.", None, None),
    ("how many of the anomalies happened specifically in July?",
     "66 anomalous T (degC) readings in 2014 in 24 windows.", None, None),
]


def adversarial_faithfulness_probe(cases: list[tuple[str, str, str | None, str | None]] = ADVERSARIAL_PROBE_CASES,
                                   model: str = DEFAULT_MODEL, n_repeats: int = 3) -> dict:
    """Stress-test `explain()`'s faithfulness with deliberately mismatched (question, facts)
    pairs -- see the module comment above. Returns the invention rate under adversarial pressure,
    plus every invented number found, for real evidence rather than an assumption either way.
    Returns ``{"available": False}`` if Ollama isn't reachable.
    """
    if not is_available():
        return {"available": False}
    per_case: list[dict[str, Any]] = []
    for question, computed, context, inferred in cases:
        facts = " ".join(filter(None, [computed, context, inferred]))
        invented_runs = 0
        invented_numbers: list[float] = []
        for _ in range(n_repeats):
            try:
                out = explain(question, computed=computed, context=context, inferred=inferred,
                             model=model)
            except (RuntimeError, urllib.error.URLError, OSError):
                continue
            result = faithfulness(facts, out)
            if result["invented"]:
                invented_runs += 1
                invented_numbers.extend(result["invented"])
        per_case.append({"question": question, "invented_runs": invented_runs,
                         "n_repeats": n_repeats, "invented_numbers": invented_numbers})
    total = sum(c["n_repeats"] for c in per_case)
    total_invented = sum(c["invented_runs"] for c in per_case)
    return {
        "available": True,
        "per_case": per_case,
        "invention_rate": total_invented / total if total else 0.0,
        "n_total_runs": total,
    }
