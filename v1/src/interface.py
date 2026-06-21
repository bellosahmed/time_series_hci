"""Natural-language interface — the HCI layer.

Routes a plain-English question to the exact query engine, attaches retrieved
context, and returns a trust-aware answer. The routing is deterministic (keyword
intent matching), so no LLM is needed to get the *right numbers* — an LLM can
later improve the phrasing only.

Flow:  question -> interpret (intent + sensor) -> query engine (truth)
                -> RAG context -> trust-aware answer
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import analysis, anomaly, query_engine as qe, rag_system

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
    "max": ["max", "maximum", "highest", "hottest", "peak", "warmest"],
    "min": ["min", "minimum", "lowest", "coldest"],
    "mean": ["mean", "average", "typical"],
    "spread": ["std", "deviation", "variability", "spread", "variance"],
    "correlation": ["correlation", "correlate", "related", "relationship", "relate", "versus", " vs "],
    "missing": ["missing", "gap", "gaps", "absent"],
    "anomaly": ["anomaly", "anomalies", "unusual", "abnormal", "weird", "strange", "outlier"],
    "summary": ["summary", "summarise", "summarize", "describe", "overview"],
}


@dataclass
class Answer:
    question: str
    intent: str | None
    text: str
    value: object = None
    sensors: list = field(default_factory=list)


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

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.retriever = rag_system.Retriever()

    def _context_for(self, sensor: str) -> str | None:
        snip = self.retriever.for_sensor(sensor)
        return snip.text if snip else None

    def ask(self, question: str) -> Answer:
        intent = detect_intent(question)
        sensors = resolve_sensors(question, list(self.df.columns))
        sensor = sensors[0] if sensors else "T (degC)"  # default to temperature
        ctx = self._context_for(sensor)

        if intent == "max":
            r = qe.get_max(self.df, sensor)
            text = rag_system.explain(
                f"Q: {question}",
                computed=f"max {sensor} was {r.value:.2f} at {r.when}.", context=ctx)
            return Answer(question, intent, text, r, [sensor])

        if intent == "min":
            r = qe.get_min(self.df, sensor)
            text = rag_system.explain(
                f"Q: {question}",
                computed=f"min {sensor} was {r.value:.2f} at {r.when}.", context=ctx)
            return Answer(question, intent, text, r, [sensor])

        if intent in ("mean", "spread"):
            r = qe.get_mean_std(self.df, sensor)
            computed = f"{sensor} mean={r['mean']:.2f}, std={r['std']:.2f}, range={r['range']:.2f}."
            return Answer(question, intent,
                          rag_system.explain(f"Q: {question}", computed=computed, context=ctx),
                          r, [sensor])

        if intent == "correlation":
            if len(sensors) < 2:
                sensors = (sensors + ["T (degC)", "rho (g/m**3)"])[:2]
            a, b = sensors[0], sensors[1]
            c = qe.get_correlation(self.df, a, b)
            rel = "move together" if c > 0 else "move in opposite directions"
            return Answer(question, intent,
                          rag_system.explain(
                              f"Q: {question}",
                              computed=f"correlation({a}, {b}) = {c:.3f}.",
                              inferred=f"They {rel}."),
                          c, [a, b])

        if intent == "missing":
            n = qe.count_missing(self.df, sensor)
            return Answer(question, intent,
                          rag_system.explain(f"Q: {question}",
                                             computed=f"{sensor} has {n} missing values.", context=ctx),
                          n, [sensor])

        if intent == "anomaly":
            decomp = analysis.decompose(self.df[sensor], period=24)
            flags = anomaly.zscore_anomalies(decomp["resid"].dropna(), threshold=4.0)
            windows = qe.get_anomaly_windows(flags)
            computed = f"{int(flags.sum())} anomalous {sensor} readings in {len(windows)} windows."
            first = ("First window: "
                     f"{windows[0]['start']} -> {windows[0]['end']}." if windows else "")
            return Answer(question, intent,
                          rag_system.explain(f"Q: {question}", computed=computed, context=first or ctx),
                          windows, [sensor])

        if intent == "summary":
            s = qe.sensor_summary(self.df, sensor)
            computed = (f"{sensor}: min={s['min']:.2f}, max={s['max']:.2f}, "
                        f"mean={s['mean']:.2f}, std={s['std']:.2f}, missing={s['missing']}.")
            return Answer(question, intent,
                          rag_system.explain(f"Q: {question}", computed=computed, context=ctx),
                          s, [sensor])

        # Fallback
        return Answer(question, None,
                      "I couldn't map that to a known query. Try: max/min/mean/"
                      "correlation/missing/anomaly/summary of a sensor.", None, sensors)


def repl(df: pd.DataFrame) -> None:  # pragma: no cover
    """Tiny interactive loop (the CLI HCI front-end)."""
    iface = Interface(df)
    print("Ask about the data (e.g. 'what was the max temperature?'). Ctrl-D to quit.")
    while True:
        try:
            q = input("\n> ").strip()
        except EOFError:
            break
        if q:
            print(iface.ask(q).text)
