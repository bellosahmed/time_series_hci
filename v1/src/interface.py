"""Natural-language interface — the HCI layer.

Routes a plain-English question to the exact query engine, attaches retrieved
context, and returns a trust-aware answer. Routing is deterministic (keyword
intent matching), so the *right numbers* never depend on an LLM.

Phrasing can be done two ways:
- **template** (default) — `rag_system.explain`, fully offline, no model.
- **local LLM** — set `use_llm=True`; if a local Ollama server is running it
  rephrases the *already-computed* facts into natural prose. It never changes
  the numbers. Falls back to the template if Ollama is unavailable.

Flow:  question -> interpret (intent + sensor) -> query engine (truth)
                -> RAG context -> phrase (template or LLM) -> trust-aware answer
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import analysis, anomaly, llm, query_engine as qe, rag_system

# Map everyday words to dataset columns.
SENSOR_ALIASES = {
    "temperature": "T (degC)", "temp": "T (degC)", "hot": "T (degC)", "cold": "T (degC)",
    "humidity": "rh (%)", "humid": "rh (%)",
    "pressure": "p (mbar)",
    "density": "rho (g/m**3)",
    "dew": "Tdew (degC)", "dewpoint": "Tdew (degC)",
    "wind": "wv (m/s)", "gust": "max. wv (m/s)",
    "potential": "Tpot (K)",
}

INTENTS = {
    "max": ["max", "maximum", "highest", "hottest", "peak", "warmest", "top", "biggest", "most"],
    "min": ["min", "minimum", "lowest", "coldest", "smallest", "least"],
    "mean": ["mean", "average", "typical", "normally"],
    "spread": ["std", "deviation", "variability", "spread", "variance", "vary", "fluctuate"],
    "correlation": ["correlation", "correlate", "related", "relationship", "relate", "versus",
                    " vs ", "depend", "affect", "influence", "connection"],
    "missing": ["missing", "gap", "gaps", "absent", "incomplete"],
    "anomaly": ["anomaly", "anomalies", "unusual", "abnormal", "weird", "strange", "outlier",
                "odd", "spike", "wrong", "problem"],
    "summary": ["summary", "summarise", "summarize", "describe", "overview", "tell me about"],
}


@dataclass
class Answer:
    question: str
    intent: str | None
    text: str
    value: object = None
    sensors: list = field(default_factory=list)
    computed: str = ""
    context: str | None = None
    inferred: str | None = None


def resolve_sensors(question: str, columns) -> list[str]:
    """Find which dataset columns a question refers to (via aliases + exact names)."""
    q = question.lower()
    found: list[str] = []
    for word, col in SENSOR_ALIASES.items():
        if word in q and col in columns and col not in found:
            found.append(col)
    for col in columns:  # also catch exact column names mentioned
        if col.lower() in q and col not in found:
            found.append(col)
    return found


def detect_intent(question: str) -> str | None:
    q = f" {question.lower()} "
    for intent, keywords in INTENTS.items():
        if any(kw in q for kw in keywords):
            return intent
    return None


class Interface:
    """Answer natural-language questions about a prepared DataFrame."""

    def __init__(self, df: pd.DataFrame, use_llm: bool = False, model: str = "llama3.2"):
        self.df = df
        self.retriever = rag_system.Retriever()
        self.use_llm = use_llm
        self.model = model

    def _context_for(self, sensor: str) -> str | None:
        snip = self.retriever.for_sensor(sensor)
        return snip.text if snip else None

    def _phrase(self, question, computed, context=None, inferred=None) -> str:
        """Produce the answer text: local LLM if enabled+available, else template."""
        if self.use_llm and llm.is_available():
            try:
                return llm.explain(question, computed=computed, context=context,
                                   inferred=inferred, model=self.model)
            except Exception:
                pass  # fall back to template on any LLM error
        return rag_system.explain(f"Q: {question}", computed=computed,
                                  context=context, inferred=inferred)

    def _reply(self, question, intent, value, sensors, computed, context=None, inferred=None):
        text = self._phrase(question, computed, context, inferred)
        return Answer(question, intent, text, value, sensors, computed, context, inferred)

    def _interpret(self, question: str):
        """Return (intent, sensors). Keyword match first; LLM fallback for free phrasing."""
        intent = detect_intent(question)
        sensors = resolve_sensors(question, list(self.df.columns))
        # If keywords couldn't classify and a local LLM is enabled, let it interpret.
        if intent is None and self.use_llm and llm.is_available():
            parsed = llm.classify(question, list(INTENTS.keys()), list(self.df.columns),
                                  model=self.model)
            if parsed:
                intent = parsed["intent"] or intent
                for s in parsed["sensors"]:
                    if s not in sensors:
                        sensors.append(s)
        return intent, sensors

    def ask(self, question: str) -> Answer:
        intent, sensors = self._interpret(question)
        sensor = sensors[0] if sensors else "T (degC)"  # default to temperature
        ctx = self._context_for(sensor)

        if intent == "max":
            r = qe.get_max(self.df, sensor)
            return self._reply(question, intent, r, [sensor],
                               computed=f"max {sensor} was {r.value:.2f} at {r.when}.", context=ctx)

        if intent == "min":
            r = qe.get_min(self.df, sensor)
            return self._reply(question, intent, r, [sensor],
                               computed=f"min {sensor} was {r.value:.2f} at {r.when}.", context=ctx)

        if intent in ("mean", "spread"):
            r = qe.get_mean_std(self.df, sensor)
            return self._reply(question, intent, r, [sensor],
                               computed=f"{sensor} mean={r['mean']:.2f}, std={r['std']:.2f}, range={r['range']:.2f}.",
                               context=ctx)

        if intent == "correlation":
            if len(sensors) < 2:
                sensors = (sensors + ["T (degC)", "rho (g/m**3)"])[:2]
            a, b = sensors[0], sensors[1]
            c = qe.get_correlation(self.df, a, b)
            rel = "move together" if c > 0 else "move in opposite directions"
            return self._reply(question, intent, c, [a, b],
                               computed=f"correlation({a}, {b}) = {c:.3f}.", inferred=f"They {rel}.")

        if intent == "missing":
            n = qe.count_missing(self.df, sensor)
            return self._reply(question, intent, n, [sensor],
                               computed=f"{sensor} has {n} missing values.", context=ctx)

        if intent == "anomaly":
            decomp = analysis.decompose(self.df[sensor], period=24)
            flags = anomaly.zscore_anomalies(decomp["resid"].dropna(), threshold=4.0)
            windows = qe.get_anomaly_windows(flags)
            first = (f"First window: {windows[0]['start']} -> {windows[0]['end']}." if windows else None)
            return self._reply(question, intent, windows, [sensor],
                               computed=f"{int(flags.sum())} anomalous {sensor} readings in {len(windows)} windows.",
                               context=first or ctx)

        if intent == "summary":
            s = qe.sensor_summary(self.df, sensor)
            return self._reply(question, intent, s, [sensor],
                               computed=(f"{sensor}: min={s['min']:.2f}, max={s['max']:.2f}, "
                                         f"mean={s['mean']:.2f}, std={s['std']:.2f}, missing={s['missing']}."),
                               context=ctx)

        return Answer(question, None,
                      "I couldn't map that to a known query. Try: max/min/mean/"
                      "correlation/missing/anomaly/summary of a sensor.", None, sensors)


def repl(df: pd.DataFrame, use_llm: bool = False) -> None:  # pragma: no cover
    """Tiny interactive loop (the CLI HCI front-end)."""
    iface = Interface(df, use_llm=use_llm)
    mode = "LLM" if (use_llm and llm.is_available()) else "template"
    print(f"Ask about the data ({mode} phrasing). e.g. 'what was the max temperature?'. Ctrl-D to quit.")
    while True:
        try:
            q = input("\n> ").strip()
        except EOFError:
            break
        if q:
            print(iface.ask(q).text)
