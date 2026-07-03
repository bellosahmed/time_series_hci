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

import re
from dataclasses import dataclass, field

import pandas as pd

from . import analysis, anomaly, forecasting, llm, query_engine as qe, rag_system, reasoning

from .interface_parsing import (  # noqa: F401  re-exported for callers and tests
    INTENTS,
    SENSOR_ALIASES,
    _coerce_date,
    detect_intent,
    resolve_compare,
    resolve_day,
    resolve_period,
    resolve_sensors,
    resolve_threshold,
)


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
    chart: object = None


class Interface:
    """Answer natural-language questions about a prepared DataFrame."""

    def __init__(self, df: pd.DataFrame, use_llm: bool = False, model: str = llm.DEFAULT_MODEL):
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

    def _understand(self, question: str):
        """Work out (intent, sensors, start, end, label, threshold) for a question.

        Deterministic parsing first (keywords + period/threshold regex, fully offline).
        If a local LLM is enabled it *refines* the plan — choosing the intent/sensor and
        resolving free-form or seasonal dates from any phrasing — but it only ever returns
        a structured plan; every number is still computed by the query engine, never the LLM.
        """
        intent = detect_intent(question)
        sensors = resolve_sensors(question, list(self.df.columns))
        start, end, label = resolve_period(question)
        thr = resolve_threshold(question)
        day = resolve_day(question)

        # The LLM only *fills gaps* the reliable deterministic parser left open — it
        # never overrides a confident keyword/explicit-date match (a small model is
        # error-prone, e.g. flipping "typical"->mean into max).
        if self.use_llm and llm.is_available():
            plan = llm.plan(question, list(INTENTS.keys()), list(self.df.columns),
                            model=self.model)
            if plan:
                if intent is None:
                    intent = plan.get("intent")
                for s in plan.get("sensors", []):
                    if s not in sensors:
                        sensors.append(s)
                if start is None and end is None:           # no explicit period found
                    s2, e2 = _coerce_date(plan.get("start")), _coerce_date(plan.get("end"))
                    if s2 or e2:
                        start, end = s2 or e2, e2 or s2
                        label = plan.get("period") or f"{start} to {end}"
                if thr is None and plan.get("op") in (">", "<") and plan.get("value") is not None:
                    thr = (plan["op"], float(plan["value"]))
        return intent, sensors, start, end, label, thr, day

    def _slice(self, start, end, label):
        """Return (data, where_label) for a date window; (None, label) if it has no rows."""
        if start is None and end is None:
            return self.df, ""
        sub = self.df.loc[start:end]
        if len(sub):
            return sub, f" in {label}"
        return None, label

    def _missing_concept(self, question: str):
        """If the user named a sensor concept this dataset doesn't have, return the word."""
        q = question.lower()
        for word, col in SENSOR_ALIASES.items():
            if re.search(rf"\b{re.escape(word)}\b", q) and col not in self.df.columns:
                return word
        return None

    def _year_stat(self, year: int, sensor: str, stat: str):
        sub = self.df[self.df.index.year == year]
        if not len(sub):
            return None
        col = sub[sensor]
        return float(col.max() if stat == "max" else col.min() if stat == "min" else col.mean())

    def _compare(self, question, sensor, ctx, yA, yB, stat):
        a, b = self._year_stat(yA, sensor, stat), self._year_stat(yB, sensor, stat)
        if a is None or b is None:
            return Answer(question, "compare",
                          f"I don't have {sensor} data for {yA if a is None else yB}.",
                          None, [sensor])
        word = {"max": "highest", "min": "lowest", "mean": "average"}[stat]
        if stat == "min":
            extreme, direction = (yA if a <= b else yB), "colder"
        else:
            extreme, direction = (yA if a >= b else yB), "warmer"
        return self._reply(question, "compare", {"yA": yA, "a": a, "yB": yB, "b": b}, [sensor],
                           computed=f"{word} {sensor}: {yA} = {a:.2f}, {yB} = {b:.2f} "
                                    f"(difference {abs(a - b):.2f}).",
                           context=ctx,
                           inferred=f"{extreme} was {direction} on this measure.")

    def ask(self, question: str) -> Answer:
        """Answer a question and attach a small chart payload for the web UI."""
        ans = self._answer(question)
        try:
            ans.chart = self._chart_for(ans)
        except Exception:
            ans.chart = None
        return ans

    def _scoped(self, question: str, sensor: str):
        """Re-derive the (deterministic) scoped series for charting."""
        day = resolve_day(question)
        if day is not None:
            m, d, y, _ = day
            idx = self.df.index
            return self.df[(idx.month == m) & (idx.day == d) & ((idx.year == y) if y else True)]
        start, end, _ = resolve_period(question)
        return self.df.loc[start:end] if (start or end) else self.df

    def _chart_for(self, ans: "Answer"):
        """Build a small downsampled chart (line + optional markers / forecast band)."""
        if not ans.sensors or ans.intent in ("correlation", "compare", "missing"):
            return None
        sensor = ans.sensors[0]
        if sensor not in self.df.columns:
            return None
        data = self._scoped(ans.question, sensor)
        if not len(data):
            return None
        if ans.intent == "forecast" and ans.value is not None:
            return self._forecast_chart(sensor, data, ans.value)
        s = data[sensor].dropna()
        step = max(1, len(s) // 220)
        ds = s.iloc[::step]
        marks = []
        if ans.intent in ("max", "min") and hasattr(ans.value, "when"):
            marks = [{"x": ans.value.when.isoformat(), "y": round(float(ans.value.value), 2),
                      "kind": ans.intent}]
        elif ans.intent == "anomaly" and isinstance(ans.value, list):
            for w in ans.value[:80]:
                ts = w["start"]
                if ts in s.index:
                    marks.append({"x": ts.isoformat(), "y": round(float(s.loc[ts]), 2),
                                  "kind": "anomaly"})
        return {"label": sensor,
                "x": [t.isoformat() for t in ds.index],
                "y": [round(float(v), 2) for v in ds.values],
                "marks": marks}

    def _forecast_chart(self, sensor, data, fc):
        hist = data[sensor].resample("1D").mean().dropna().iloc[-60:]
        return {"label": sensor,
                "x": [t.isoformat() for t in hist.index],
                "y": [round(float(v), 2) for v in hist.values],
                "marks": [],
                "forecast": {"x": [t.isoformat() for t in fc.index],
                             "mean": [round(float(v), 2) for v in fc["mean"]],
                             "lower": [round(float(v), 2) for v in fc["lower"]],
                             "upper": [round(float(v), 2) for v in fc["upper"]]}}

    def _answer(self, question: str) -> Answer:
        intent, sensors, start, end, label, thr, day = self._understand(question)

        # Did the user ask about a sensor this dataset doesn't have?
        if not sensors:
            miss = self._missing_concept(question)
            if miss:
                return Answer(question, intent,
                              f"I don't track {miss} in this dataset. Available sensors: "
                              f"{', '.join(self.df.columns)}.", None, [])
        sensor = sensors[0] if sensors else "T (degC)"  # default to temperature
        ctx = self._context_for(sensor)

        # Compare two periods ("was 2013 hotter than 2014?").
        cmp = resolve_compare(question)
        if cmp and intent in (None, "compare", "max", "min", "mean", "trend", "weather"):
            return self._compare(question, sensor, ctx, *cmp)

        # Scope: a specific day, else a year/month/season/range window, else whole record.
        if day is not None:
            m, d, y, day_label = day
            idx = self.df.index
            mask = (idx.month == m) & (idx.day == d) & ((idx.year == y) if y else True)
            data = self.df[mask]
            if y:
                scope = day_label
            else:
                present = sorted({int(yr) for yr in data.index.year})
                scope = f"{day_label} (across {present[0]}–{present[-1]})" if present else day_label
            where = f" on {scope}"
        else:
            data, where = self._slice(start, end, label)
            scope = label

        if data is None or not len(data):
            return Answer(question, intent,
                          f"I have no data for {scope or 'that period'}; the record runs "
                          f"{self.df.index.min():%Y-%m} to {self.df.index.max():%Y-%m}.",
                          None, [sensor])

        # Weather report: a multi-sensor snapshot for the chosen day/period.
        if intent == "weather":
            wanted = [c for c in ("T (degC)", "rh (%)", "Tdew (degC)", "p (mbar)", "wv (m/s)")
                      if c in data.columns]
            rep = qe.weather_report(data, wanted)
            bits = []
            if "T (degC)" in rep:
                t = rep["T (degC)"]
                bits.append(f"temperature averaged {t['mean']:.1f}°C (low {t['min']:.1f}, high {t['max']:.1f})")
            if "rh (%)" in rep:
                bits.append(f"humidity ~{rep['rh (%)']['mean']:.0f}%")
            if "Tdew (degC)" in rep:
                bits.append(f"dew point ~{rep['Tdew (degC)']['mean']:.1f}°C")
            if "p (mbar)" in rep:
                bits.append(f"pressure ~{rep['p (mbar)']['mean']:.0f} mbar")
            if "wv (m/s)" in rep:
                bits.append(f"wind ~{rep['wv (m/s)']['mean']:.1f} m/s")
            place = scope or "the whole record"
            computed = (f"Weather for {place}: " + "; ".join(bits)
                        + f". ({len(data)} hourly readings)")
            return self._reply(question, "weather", rep, wanted, computed=computed, context=ctx)

        # Threshold questions ("how often above 30?", "colder than -5 in 2013").
        if thr is not None:
            op, value = thr
            r = qe.threshold_events(data, sensor, op, value)
            sign = "above" if op == ">" else "below"
            when = f", first on {qe.fmt_when(r['first'])}" if r["first"] is not None else ""
            return self._reply(question, "threshold", r, [sensor],
                               computed=(f"{sensor} was {sign} {value:g}{where} "
                                         f"{r['count']} times ({r['pct']:.2f}% of readings){when}."),
                               context=ctx)

        if intent == "forecast":
            try:
                fc = forecasting.forecast(data[sensor], steps=14)
            except Exception:
                return self._reply(question, intent, None, [sensor],
                                   computed=f"I couldn't fit a forecast for {sensor}{where}.",
                                   context=ctx)
            first, last = fc["mean"].iloc[0], fc["mean"].iloc[-1]
            end = qe.fmt_when(fc.index[-1], with_time=False)
            return self._reply(question, intent, fc, [sensor],
                               computed=(f"14-day forecast for {sensor}: about {first:.1f} next day, "
                                         f"around {last:.1f} by {end} "
                                         f"(80% range {fc['lower'].min():.1f} to {fc['upper'].max():.1f})."),
                               context=ctx,
                               inferred="This projects the usual pattern and recent trend; sudden "
                                        "shocks (storms, heatwaves) can't be predicted, only detected later.")

        if intent == "trend":
            slope = analysis.trend_slope(data[sensor])         # units per hour
            per_year = slope * 24 * 365.25
            direction = ("rising" if per_year > 0.05 else
                         "falling" if per_year < -0.05 else "roughly flat")
            return self._reply(question, intent, per_year, [sensor],
                               computed=f"{sensor}{where} is {direction} at "
                                        f"{per_year:+.2f} per year (linear trend).",
                               context=ctx,
                               inferred="This is the straight-line trend only; weather also has "
                                        "strong seasonal cycles, so short windows can mislead.")

        if intent == "max":
            r = qe.get_max(data, sensor)
            return self._reply(question, intent, r, [sensor],
                               computed=f"max {sensor}{where} was {r.value:.2f} on {qe.fmt_when(r.when)}.", context=ctx)

        if intent == "min":
            r = qe.get_min(data, sensor)
            return self._reply(question, intent, r, [sensor],
                               computed=f"min {sensor}{where} was {r.value:.2f} on {qe.fmt_when(r.when)}.", context=ctx)

        if intent in ("mean", "spread"):
            r = qe.get_mean_std(data, sensor)
            return self._reply(question, intent, r, [sensor],
                               computed=f"{sensor}{where} mean={r['mean']:.2f}, std={r['std']:.2f}, range={r['range']:.2f}.",
                               context=ctx)

        if intent == "correlation":
            if len(sensors) < 2:
                sensors = (sensors + ["T (degC)", "rho (g/m**3)"])[:2]
            a, b = sensors[0], sensors[1]
            c = qe.get_correlation(data, a, b)
            rel = "move together" if c > 0 else "move in opposite directions"
            return self._reply(question, intent, c, [a, b],
                               computed=f"correlation({a}, {b}){where} = {c:.3f}.", inferred=f"They {rel}.")

        if intent == "missing":
            n = qe.count_missing(data, sensor)
            return self._reply(question, intent, n, [sensor],
                               computed=f"{sensor}{where} has {n} missing values.", context=ctx)

        if intent == "anomaly":
            decomp = analysis.decompose(data[sensor], period=24)
            flags = anomaly.zscore_anomalies(decomp["resid"].dropna(), threshold=4.0)
            windows = qe.get_anomaly_windows(flags)
            # Multi-sensor reasoning: explain *why* the first
            # flagged moment looked unusual — what every sensor did at that timestamp.
            inferred = None
            if windows:
                try:
                    inferred = reasoning.describe_moment(data, windows[0]["start"])
                except Exception:
                    inferred = None
            first = f"First anomaly on {qe.fmt_when(windows[0]['start'])}." if windows else None
            return self._reply(question, intent, windows, [sensor],
                               computed=f"{int(flags.sum())} anomalous {sensor} readings{where} in {len(windows)} windows.",
                               context=first or ctx, inferred=inferred)

        if intent == "summary":
            s = qe.sensor_summary(data, sensor)
            return self._reply(question, intent, s, [sensor],
                               computed=(f"{sensor}{where}: min={s['min']:.2f}, max={s['max']:.2f}, "
                                         f"mean={s['mean']:.2f}, std={s['std']:.2f}, missing={s['missing']}."),
                               context=ctx)

        return Answer(question, None,
                      "I couldn't map that to a known query. Try max / min / mean / trend / "
                      "compare / correlation / missing / anomaly / weather / summary — optionally "
                      "scoped to a year, month, season or day (e.g. 'hottest in summer 2013'), or a "
                      "threshold ('how often above 30?').", None, sensors)


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
