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

import calendar
import logging
import re
from dataclasses import dataclass, field

import pandas as pd

from . import analysis, anomaly, clustering, forecasting, llm, rag_system, reasoning
from . import query_engine as qe
from .interface_parsing import (  # noqa: F401  re-exported for callers and tests
    DIRECTION_LABELS,
    DIRECTION_SUPERLATIVES,
    INTENTS,
    LEGEND_EXPLANATIONS,
    SENSOR_ALIASES,
    _coerce_date,
    clarifications_needed,
    correlation_strength,
    detect_intent,
    hour_mask,
    is_causal_query,
    is_correlation_breakdown_query,
    is_followup_cue,
    is_forward_looking_query,
    resolve_any_period,
    resolve_compare,
    resolve_day,
    resolve_direction,
    resolve_explicit_scope_restriction,
    resolve_followup,
    resolve_generic_direction,
    resolve_hour,
    resolve_legend_query,
    resolve_ordinal_unit_in_year,
    resolve_period,
    resolve_recurring_month,
    resolve_relative_period,
    resolve_secondary_period,
    resolve_sensors,
    resolve_threshold,
    spell_normalize,
    split_target_clause,
)

logger = logging.getLogger(__name__)


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
    borderline: list = field(default_factory=list)  # near-threshold "almost anomalies"
    network: list = field(default_factory=list)  # multi-sensor portfolio (Mahalanobis+flat-line) windows
    detector: str | None = None  # which network detector produced the anomaly answer (transparency)
    resolved: str | None = None  # standalone rewrite of a follow-up (transparency), if any
    understood: str | None = None  # trace of what the deterministic parser found (transparency)
    spelling: list = field(default_factory=list)  # typo fixes applied before routing: (orig, fixed)
    ignored: list = field(default_factory=list)  # meaningful words the router did not understand
    assumptions: list = field(default_factory=list)  # defaults the router had to assume (transparency)


# Selectable **network-level** anomaly detectors (all read the whole in-scope sensor frame and return a
# boolean Series; all are deterministic code — the LLM never detects, only phrases). "portfolio" is the
# notebook-05 recommendation (Mahalanobis ∪ flat-line) and the default.
NETWORK_DETECTORS: dict = {
    "portfolio": anomaly.sensor_network_anomalies,
    "mahalanobis": anomaly.mahalanobis_anomalies,
    "isolation_forest": anomaly.isolation_forest_anomalies,
    "lof": anomaly.lof_anomalies,
    "ocsvm": anomaly.ocsvm_anomalies,
}
DEFAULT_DETECTOR = "portfolio"


# --- Semantic fallback router --------------------------------------------------------------
# When a question matches no intent *and* no sensor keyword, it reaches the fallback below.
# Rather than show a raw "I couldn't map that" error, we measure the question's *meaning*
# against the sensor knowledge base with a dense encoder (Sentence-BERT; Reimers & Gurevych,
# 2019) and either give a targeted on-topic suggestion or a polite off-topic redirect.
#
# DOMAIN_THRESHOLD (cosine) separates the two. It was set from measured scores, not guessed:
# off-topic queries score <= ~0.12 ("what is my dog's name" 0.07, "hello there" 0.12) while
# on-topic ones score >= ~0.18 ("how sunny is it" 0.42, "what is the air like" 0.41). 0.15 sits
# in that gap. It is deliberately conservative: wrongly redirecting a borderline weather question
# is a smaller harm than confidently answering an off-topic one.
DOMAIN_THRESHOLD = 0.15

# Friendly name + one example query per sensor, so a semantic match becomes a concrete, helpful
# suggestion ("you seem to be asking about humidity -- try 'how humid was summer 2013'") instead
# of a generic error.
SENSOR_HINTS: dict[str, tuple[str, str]] = {
    "T (degC)": ("temperature", "hottest day in 2015"),
    "Tpot (K)": ("potential temperature", "average potential temperature in 2013"),
    "Tdew (degC)": ("the dew point", "highest dew point in summer 2014"),
    "rh (%)": ("humidity", "how humid was summer 2013"),
    "VPmax (mbar)": ("saturation vapour pressure", "max saturation vapour pressure in 2012"),
    "VPact (mbar)": ("actual vapour pressure", "average vapour pressure in 2014"),
    "VPdef (mbar)": ("the vapour-pressure deficit", "highest vapour-pressure deficit in 2015"),
    "sh (g/kg)": ("specific humidity", "average specific humidity in 2013"),
    "H2OC (mmol/mol)": ("water-vapour concentration", "max water-vapour concentration in 2012"),
    "rho (g/m**3)": ("air density", "average air density in winter 2013"),
    "p (mbar)": ("air pressure", "lowest pressure in 2014"),
    "wv (m/s)": ("wind speed", "average wind speed in winter 2012"),
    "max. wv (m/s)": ("wind gusts", "the strongest wind gust in 2015"),
    "wd (deg)": ("wind direction", "the prevailing wind direction in 2013"),
}


class Interface:
    """Answer natural-language questions about a prepared DataFrame."""

    def __init__(self, df: pd.DataFrame, use_llm: bool = False, model: str = llm.DEFAULT_MODEL,
                 detector: str = DEFAULT_DETECTOR, domain_retriever=None,
                 raw: pd.DataFrame | None = None):
        self.df = df
        # `df` is the hourly-mean grid the analysis/anomaly/forecast layers need (regularly spaced).
        # `raw` is the native-resolution frame (Jena = 10-min) used ONLY for point statistics
        # (max/min/mean/std): hourly averaging smooths peaks, so the max of the hourly means is
        # systematically below the true reading (e.g. July-2015 peak 37.04 hourly vs 37.28 raw).
        # Extrema/summary queries therefore run on `raw` so they reconcile with the source data.
        # Defaults to `df` when no raw frame is supplied, so callers that only have the hourly
        # frame (tests, verify_results) keep their previous behaviour unchanged.
        self.raw = raw if raw is not None else df
        self._anchor = df.index.max()  # the dataset's own "now" -- a static record has no real one
        self.retriever = rag_system.Retriever()
        self.use_llm = use_llm
        self.model = model
        self.detector = detector          # which network-level anomaly detector the anomaly intent uses
        self.history: list[dict] = []  # recent turns, for follow-up resolution
        self._daytypes: tuple | None = None  # lazily-built K-means day-types (labels, profiles)
        # Semantic fallback router (below): a dense retriever used only to judge domain relevance
        # of an otherwise-unroutable question. Built lazily on first use (loads a model), and
        # injectable here so tests can exercise both branches offline without the encoder.
        self._domain_retriever_injected = domain_retriever
        self._domain_clf = None

    def _day_types(self):
        """Lazily compute K-means day-types once: (labels per day, per-cluster mean features)."""
        if self._daytypes is None:
            feats = clustering.daily_features(self.df)
            labels = clustering.kmeans_cluster(feats, k=4)
            self._daytypes = (labels, clustering.cluster_profiles(feats, labels))
        return self._daytypes

    def _routable(self, question: str) -> bool:
        """True if the deterministic parser can route `question` (an intent/sensor/period/day).

        Used to reject a garbled LLM rewrite: only a rewrite that maps to a real query is
        allowed to replace the user's words.
        """
        cols = list(self.df.columns)
        start, end, _ = resolve_any_period(question, self._anchor)
        return bool(detect_intent(question) or resolve_sensors(question, cols)
                    or start or end or resolve_day(question) is not None
                    or resolve_legend_query(question) or resolve_hour(question)[0] is not None)

    def _needs_context(self, question: str) -> bool:
        """True if `question` looks elliptical — i.e. not already a self-contained query.

        A self-contained question has an intent *and* a sensor or a period (e.g. "hottest in
        2013"); such questions are answered directly, without invoking the LLM rewriter. Only
        elliptical follow-ups ("what about 2014?", "and the peak?") are rewritten.
        """
        cols = list(self.df.columns)
        start, end, _ = resolve_any_period(question, self._anchor)
        has_scope = bool(resolve_sensors(question, cols) or start or end
                         or resolve_day(question) is not None
                         or resolve_hour(question)[0] is not None)
        return not (detect_intent(question) is not None and has_scope)

    def _context_for(self, sensor: str) -> str | None:
        snip = self.retriever.for_sensor(sensor)
        return snip.text if snip else None

    def _domain_classifier(self):
        """Lazily build the dense semantic retriever used to judge domain relevance (cached).

        Uses a Sentence-BERT encoder when available and transparently falls back to TF-IDF
        otherwise (``rag_system.dense_retriever``), so it is always usable. Injectable via the
        constructor's ``domain_retriever`` so tests can drive both branches without the encoder.
        """
        if self._domain_clf is None:
            self._domain_clf = (self._domain_retriever_injected
                                or rag_system.dense_retriever())
        return self._domain_clf

    def _semantic_fallback(self, question: str) -> Answer:
        """Answer an unroutable question without ever surfacing a raw error.

        Runs a *semantic* search of the question against the sensor knowledge base. A confident
        match (score >= ``DOMAIN_THRESHOLD``) means the question is about the weather data, so we
        return a targeted suggestion naming the closest sensor and an example query. No confident
        match means it is off-topic, so we return a polite redirect to the dataset's scope. This
        path produces no number, so the trust guarantee is untouched.
        """
        hits = self._domain_classifier().retrieve(question, k=1)
        if hits and hits[0].score >= DOMAIN_THRESHOLD:
            friendly, example = SENSOR_HINTS.get(hits[0].sensor, ("the weather data",
                                                                  "hottest day in 2015"))
            text = (f"I think you're asking about {friendly}, but I'm not sure exactly what to "
                    f"compute — could you rephrase to confirm? For example '{example}', or ask for "
                    f"the max, min, average, trend or anomalies over a year, month or season.")
            return Answer(question, None, text, None, [])
        text = ("I couldn't understand that as a question about this weather dataset — temperature, "
                "humidity, wind, pressure, dew point and related sensors, recorded hourly from 2009 "
                "to 2016. Could you rephrase and try again? For example 'coldest day in 2013' or "
                "'average wind speed in winter 2012'.")
        return Answer(question, None, text, None, [])

    def _phrase(self, question, computed, context=None, inferred=None) -> str:
        """Produce the answer text: local LLM if enabled+available, else template."""
        if self.use_llm and llm.is_available():
            try:
                out = llm.explain(question, computed=computed, context=context,
                                  inferred=inferred, model=self.model)
                # Trust gate: if the model fabricated a figure absent from the computed
                # facts, discard its phrasing and use the exact template instead.
                facts = " ".join(filter(None, [computed, inferred]))
                invented = llm.faithfulness(facts, out)["invented"]
                if invented:
                    logger.warning("LLM invented %s for %r; using template.", invented, question)
                else:
                    return out
            except Exception as exc:
                # Log, don't swallow: a silently-dead LLM once went unnoticed for weeks.
                logger.warning("LLM phrasing failed (%s); using template.", exc)
        return rag_system.explain(f"Q: {question}", computed=computed,
                                  context=context, inferred=inferred)

    def _reply(self, question, intent, value, sensors, computed, context=None, inferred=None):
        text = self._phrase(question, computed, context, inferred)
        return Answer(question, intent, text, value, sensors, computed, context, inferred)

    def _clarify(self, question: str, spell_fixes: list, clarify_words: list) -> Answer:
        """A no-answer response asking the user to fix words that look misspelled but couldn't be
        confidently corrected. `clarify_words` is a list of (word, suggestion) pairs.

        `intent` is None (this is not a computed answer) and `value` is None, so callers/UI treat it
        as a prompt, not a result. Any typos that *were* confidently corrected are noted first.
        """
        parts = [f"'{w}' (did you mean '{sug}'?)" for w, sug in clarify_words]
        prefix = ""
        if spell_fixes:
            fixed = ", ".join(f"'{o}' as '{f}'" for o, f in spell_fixes)
            prefix = f"I read {fixed}, but "
        msg = (f"{prefix}I couldn't understand {', '.join(parts)}. "
               f"Could you check the spelling and ask again?")
        if prefix:
            msg = msg[0].upper() + msg[1:]
        return Answer(question, None, msg, None, [],
                      spelling=spell_fixes, ignored=[w for w, _ in clarify_words])

    # Intents that legitimately consume an "A and B" phrasing as a SINGLE question (a correlation
    # is between two sensors; a comparison is between two periods) — never split these on "and".
    _COMPOUND_KEEP_INTENTS = {"correlation", "compare"}
    # Splits a compound question into its parts. Only clear separators: semicolons, "&", explicit
    # additive cues, and bare " and ". Each candidate part is re-validated before we commit to a
    # split, so this regex being permissive is safe.
    _COMPOUND_SPLIT_RE = re.compile(
        r"\s*;\s*|\s+and\s+also\s+|\s+as\s+well\s+as\s+|\s+&\s+|\s+plus\s+|\s+and\s+", re.IGNORECASE)

    def _split_questions(self, question: str) -> list[str] | None:
        """Return ≥2 sub-questions if `question` is genuinely compound, else None.

        Conservative on purpose: a question is only treated as compound when (a) it is NOT itself a
        correlation/compare question (those legitimately use "and"), and (b) at least two of the
        split parts each independently look like their own question — i.e. each carries its own
        intent keyword. This keeps "correlation between temperature and humidity", "compare 2012 and
        2015" and "mean pressure between 2012 and 2014" as single questions, while
        "max temperature in 2015 and min humidity in 2013" becomes two.
        """
        if detect_intent(question) in self._COMPOUND_KEEP_INTENTS:
            return None
        if resolve_compare(question) is not None:      # "warmer than", "difference between", …
            return None
        parts = [p.strip(" ?.") for p in self._COMPOUND_SPLIT_RE.split(question) if p.strip(" ?.")]
        if len(parts) < 2:
            return None
        # each *kept* part must independently resolve an intent; require at least two such parts so a
        # trailing fragment ("…between 2012 and 2014" -> "2014") never triggers a bogus split.
        answerable = [p for p in parts if detect_intent(p) is not None]
        if len(answerable) < 2:
            return None
        return answerable

    def _answer_compound(self, clauses: list[str]) -> Answer:
        """Answer each sub-question independently through the deterministic engine and combine them.

        Each clause is self-contained, so follow-up/history resolution is skipped and the shared
        history is not consulted (or polluted) mid-list — every number is still computed by the
        query engine, one clause at a time.
        """
        saved = self.history
        self.history = []
        try:
            subs = [self._answer(c) for c in clauses]
        finally:
            self.history = saved
        text = "\n\n".join(f"{i}. {s.text}" for i, s in enumerate(subs, 1))
        sensors = sorted({sensor for s in subs for sensor in (s.sensors or [])})
        computed = " | ".join(s.computed for s in subs if s.computed)
        return Answer("", "multi", text, [s.value for s in subs], sensors, computed=computed)

    # Intents that answer about ONE named sensor over a period — the ones where a missing
    # sensor / statistic / period means the router had to assume a default worth flagging.
    _ASSUMPTION_INTENTS = {"max", "min", "mean", "spread", "summary", "trend", "threshold"}
    # Words that imply the temperature sensor even though they are not in SENSOR_ALIASES ("hot"/
    # "cold" are aliases, but "warmest"/"coolest" are not). When one is present, defaulting to
    # temperature is what the user meant, not a blind guess, so the sensor note is suppressed.
    _TEMP_IMPLIED = {"hot", "hottest", "hotter", "cold", "coldest", "colder", "warm", "warmest",
                     "warmer", "cool", "coolest", "cooler", "freezing", "scorching", "chilly",
                     "mild", "icy", "sweltering", "balmy"}

    def _assumption_notes(self, resolved: str, ans: Answer) -> list[str]:
        """Short notes for any default the router assumed, so an under-specified question is
        answered with a best guess AND an invitation to correct it, rather than silently. Empty
        when the question named its sensor, statistic and period explicitly.
        """
        if ans.intent not in self._ASSUMPTION_INTENTS:
            return []
        cols = list(self.df.columns)
        notes: list[str] = []
        # (1) sensor defaulted — no sensor named in the text, but the answer used one. Skip when a
        # temperature-implying word ("warmest") makes the temperature default what the user meant.
        words = set(re.findall(r"[a-z]+", resolved.lower()))
        if (ans.sensors and not resolve_sensors(resolved, cols)
                and not (ans.sensors[0] == "T (degC)" and words & self._TEMP_IMPLIED)):
            friendly = SENSOR_HINTS.get(ans.sensors[0], (ans.sensors[0],))[0]
            notes.append(f"assuming you mean {friendly} — name another sensor "
                         f"(humidity, pressure, wind, …) if not")
        # (2) statistic defaulted — fell back to a full overview because no max/min/mean/… keyword
        if ans.intent == "summary" and detect_intent(resolved) is None:
            notes.append("no specific statistic named, so this is a full overview — "
                         "ask for the max, min or average for just one")
        # (3) period defaulted — no year/month/season/day, so the whole record was used
        start, end, _ = resolve_any_period(resolved, self._anchor)
        if not start and not end and resolve_day(resolved) is None:
            lo = self.df.index.min().year
            hi_ts = self.df.index.max()   # a resample can leave a lone Jan-1 00:00 bucket in the next
            hi = hi_ts.year - 1 if (hi_ts.month == 1 and hi_ts.day == 1) else hi_ts.year
            notes.append(f"covering the whole {lo}–{hi} record — name a year, month or season "
                         f"to narrow it")
        # (4) another analytical intent also matches the wording (an ambiguous request)
        others = [name for name, kws in INTENTS.items()
                  if name != ans.intent
                  and name in (self._ASSUMPTION_INTENTS | {"compare", "correlation"})
                  and any(re.search(rf"\b{re.escape(k)}\b", resolved.lower()) for k in kws)]
        if others:
            notes.append(f"read as a '{ans.intent}' question — say '{others[0]}' if you meant that")
        return notes

    def _understand(self, question: str):
        """Work out (intent, sensors, start, end, label, threshold, day, hour) for a question.

        Deterministic parsing first (keywords + period/threshold regex, fully offline).
        If a local LLM is enabled it *refines* the plan — choosing the intent/sensor and
        resolving free-form or seasonal dates from any phrasing — but it only ever returns
        a structured plan; every number is still computed by the query engine, never the LLM.
        """
        intent = detect_intent(question)
        sensors = resolve_sensors(question, list(self.df.columns))
        start, end, label = resolve_any_period(question, self._anchor)
        thr = resolve_threshold(question)
        day = resolve_day(question)
        hour = resolve_hour(question)

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
                        # A one-sided bound (only start or only end) must stay open-ended --
                        # collapsing both to the one supplied date would silently turn "from
                        # June onward" into a single-day window instead of an open range.
                        start, end = s2, e2
                        label = plan.get("period") or (
                            f"{start} to {end}" if start and end
                            else f"from {start}" if start else f"up to {end}")
                if thr is None and plan.get("op") in (">", "<") and plan.get("value") is not None:
                    thr = (plan["op"], float(plan["value"]))
        return intent, sensors, start, end, label, thr, day, hour

    def _describe_understanding(self, question: str) -> str:
        """A short, human-readable trace of what the deterministic parser found in `question` --
        sensor(s), time scope, and hour -- surfaced for transparency on every answer, not just
        follow-up rewrites. Deliberately uses only the pure parsing functions (never `_understand`,
        which may call the LLM planner) so building this trace can never trigger a second LLM call
        and always reflects the verifiable, non-LLM parse -- the part of the pipeline a user can
        actually audit.
        """
        sensors = resolve_sensors(question, list(self.df.columns))
        day = resolve_day(question)
        start, end, label = resolve_any_period(question, self._anchor)
        hour = resolve_hour(question)
        bits = []
        if sensors:
            bits.append("sensor=" + ", ".join(sensors))
        if day is not None:
            bits.append(f"day={day[3]}")
        elif label:
            bits.append(f"scope={label}")
        if hour[0] is not None:
            bits.append(f"hour={hour[2]}")
        return "; ".join(bits) if bits else "no specific sensor or time scope named (using defaults)"

    def _slice(self, start, end, label):
        """Return (data, where_label) for a date window; (None, label) if it has no rows."""
        return qe.slice_window(self.df, start, end, label=label)

    def _scope_raw(self, day, start, end, label, hour):
        """The SAME window as the hourly `data`, but sliced from the native-resolution `self.raw`
        frame so extremes/statistics reflect the true readings, not hourly means. Mirrors the
        scope logic in `_answer` exactly. Returns an empty frame (not None) when the window has no
        raw rows, so callers can fall back to the hourly slice rather than crash."""
        if day is not None:
            m, d, y, _ = day
            idx = self.raw.index
            rd = self.raw[(idx.month == m) & (idx.day == d) & ((idx.year == y) if y else True)]
        else:
            rd, _ = qe.slice_window(self.raw, start, end, label=label)
            if rd is None:
                rd = self.raw.iloc[0:0]
        if hour[0] is not None and len(rd):
            rd = rd[hour_mask(rd.index, hour[0], hour[1])]
        return rd

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
        """Answer a question and attach a small chart payload for the web UI.

        Supports conversational follow-ups: with the LLM on and prior turns in ``self.history``,
        an elliptical follow-up ("what about 2014?") is rewritten into a standalone question
        before routing. The rewrite only resolves language/context — every number is still
        computed by the deterministic query engine. Legend questions and the LLM-off case are
        answered directly (single-shot), the latter being the documented no-model limitation.
        """
        original = question
        # Repair obvious typos of domain words BEFORE any routing, so a misspelled sensor/intent
        # ("higest tempreature") still resolves. Conservative + transparent: only high-confidence
        # fixes to long domain words, and every fix is surfaced on the answer (see spell_normalize).
        question, spell_fixes = spell_normalize(question)
        resolved = question
        # Compound question ("max temp in 2015 AND min humidity in 2013"): answer each part
        # separately rather than silently answering only the first (and merging their scopes). Done
        # before follow-up resolution because a compound question is self-contained.
        clauses = self._split_questions(question)
        if clauses:
            ans = self._answer_compound(clauses)
            ans.question = original
            if spell_fixes:
                ans.spelling = spell_fixes
                fixes = ", ".join(f"'{o}' as '{f}'" for o, f in spell_fixes)
                ans.text = f"(Reading {fixes}.) {ans.text}"
            ans.understood = f"compound: {len(clauses)} sub-questions"
            self.history.append({"question": question, "resolved": question, "computed": ans.computed})
            self.history = self.history[-6:]
            return ans
        # Attempt follow-up resolution when the question is elliptical, or when a continuation cue
        # ("and…", "what about…") means it should inherit the prior scope even if it restates an
        # intent/sensor (e.g. "and the coldest?" -> the coldest *in the year we were discussing*).
        if (resolve_legend_query(question) is None and self.history
                and (self._needs_context(question) or is_followup_cue(question))):
            used = False
            # Richer path first: with the LLM on, let it rewrite free-form ellipsis.
            if self.use_llm and llm.is_available():
                try:
                    rw = llm.rewrite_followup(question, self.history, model=self.model)
                    if rw and self._routable(rw) and rw.strip().lower() != question.strip().lower():
                        resolved = rw.strip()   # accept only a rewrite the parser can actually route
                        used = True
                    elif rw:
                        logger.info("follow-up rewrite %r not routable; trying deterministic memory.", rw)
                except Exception as exc:
                    logger.warning("follow-up rewrite failed (%s); trying deterministic memory.", exc)
            # Deterministic follow-up memory: works with the LLM off, and backs up an unusable rewrite.
            if not used:
                det = resolve_followup(question, self.history, list(self.df.columns))
                if det and self._routable(det) and det.strip().lower() != question.strip().lower():
                    resolved = det.strip()

        # GUARDRAIL: if a word looks like a misspelled sensor/period/intent term but is too
        # uncertain to auto-correct (a mangled month "tocber", "lowset"->lowest), do NOT answer on a
        # scope we had to guess at -- stop and ask the user to fix it, offering the "did you mean?"
        # hint. Answering with a silent-wrong scope would break the verifiable-answers premise.
        # Ordinary words far from any term ("because", "degrees") are excluded, so normal questions
        # are never blocked.
        clarify_words = clarifications_needed(resolved, tuple(self.df.columns))
        if clarify_words:
            clar = self._clarify(original, spell_fixes, clarify_words)
            self.history.append({"question": original, "resolved": resolved, "computed": ""})
            self.history = self.history[-6:]
            return clar

        ans = self._answer(resolved)
        ans.question = original                 # keep the user's own wording for display
        if spell_fixes:
            ans.spelling = spell_fixes          # transparency: which typos were corrected
            fixes = ", ".join(f"'{o}' as '{f}'" for o, f in spell_fixes)
            ans.text = f"(Reading {fixes}.) {ans.text}"
        # Best-guess-with-a-note: if the router had to assume a sensor / statistic / period (or the
        # wording matches two intents), answer anyway but flag the assumption so the user can refine.
        notes = self._assumption_notes(resolved, ans)
        if notes:
            ans.assumptions = notes
            joined = "; ".join(notes)
            ans.text = f"{ans.text}\n\nℹ️ {joined[0].upper()}{joined[1:]}."
        if resolved != original:
            ans.resolved = resolved             # transparency: what we understood it as
        ans.understood = self._describe_understanding(resolved)
        try:
            ans.chart = self._chart_for(ans)
        except Exception as exc:
            logger.warning("chart build failed for %r (%s).", question, exc)
            ans.chart = None

        self.history.append({"question": question, "resolved": resolved, "computed": ans.computed})
        self.history = self.history[-6:]        # keep a short rolling window
        return ans

    def _scoped(self, question: str, sensor: str):
        """Re-derive the (deterministic) scoped series for charting."""
        day = resolve_day(question)
        if day is not None:
            m, d, y, _ = day
            idx = self.df.index
            data = self.df[(idx.month == m) & (idx.day == d) & ((idx.year == y) if y else True)]
        else:
            start, end, _ = resolve_any_period(question, self._anchor)
            data = self.df.loc[start:end] if (start or end) else self.df
        hour = resolve_hour(question)
        if hour[0] is not None:
            data = data[hour_mask(data.index, hour[0], hour[1])]
        return data

    @staticmethod
    def _series_payload(s):
        """{x, y} for a pandas Series, timestamps ISO-formatted and values rounded."""
        return {"x": [t.isoformat() for t in s.index],
                "y": [round(float(v), 2) for v in s.values]}

    def _chart_for(self, ans: Answer):
        """Build a small chart payload (line + optional markers / forecast / comparison).

        The payload carries day- and month-resampled ``views`` so the web UI can offer a
        day-by-day / month-by-month toggle; the top-level ``x``/``y`` mirror the default view
        (kept for back-compatibility with older readers and the JS unit tests).
        """
        if ans.intent == "compare" and isinstance(ans.value, dict) and "yA" in ans.value:
            return self._compare_chart(ans.sensors[0], ans.value) if ans.sensors else None
        if not ans.sensors or ans.intent in ("correlation", "missing"):
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
        if not len(s):
            return None
        marks = []
        if ans.intent in ("max", "min") and isinstance(ans.value, dict) and "when" in ans.value:
            marks = [{"x": ans.value["when"].isoformat(), "y": round(float(ans.value["value"]), 2),
                      "kind": ans.intent}]
        elif ans.intent == "anomaly" and isinstance(ans.value, list):
            for w in ans.value[:80]:
                ts = w["start"]
                if ts in s.index:
                    marks.append({"x": ts.isoformat(), "y": round(float(s.loc[ts]), 2),
                                  "kind": "anomaly"})
            for w in ans.borderline[:80]:            # near-threshold "almost anomalies"
                ts = w["start"]
                if ts in s.index:
                    marks.append({"x": ts.isoformat(), "y": round(float(s.loc[ts]), 2),
                                  "kind": "borderline"})

        # Aggregate to whole days / months so the line is aligned and readable at any span.
        # A view is only offered when it has enough points to be meaningful (>= 2).
        span_days = (s.index[-1] - s.index[0]).total_seconds() / 86400 if len(s) > 1 else 0.0
        views: dict = {}
        if span_days >= 1.5:
            day = s.resample("1D").mean().dropna()
            step = max(1, len(day) // 400)           # cap a multi-year daily line
            views["day"] = self._series_payload(day.iloc[::step])
        if span_days >= 55:
            month = s.resample("1MS").mean().dropna()
            if len(month) >= 2:
                views["month"] = self._series_payload(month)

        if views:
            default = "month" if (span_days > 365 and "month" in views) else "day"
            base = views[default]
        else:                                         # sub-day scope: keep raw hourly readings
            step = max(1, len(s) // 220)
            base = self._series_payload(s.iloc[::step])
            default = None
        return {"label": sensor, "marks": marks, "views": views, "default": default, **base}

    def _compare_chart(self, sensor, val):
        """Overlay two years' month-of-year profiles so a comparison shows a real chart.

        Both years are mapped onto a shared synthetic year (2000) so their monthly means line
        up on a Jan-Dec axis; the UI labels months only (``monthaxis``). Returns None if either
        year lacks enough monthly coverage to draw a line.
        """
        if sensor not in self.df.columns:
            return None
        s = self.df[sensor].dropna()
        yA, yB = val["yA"], val["yB"]

        def profile(year):
            sub = s[s.index.year == year]
            if not len(sub):
                return None
            m = sub.resample("1MS").mean().dropna()
            if len(m) < 2:
                return None
            return {"x": [pd.Timestamp(2000, t.month, 1).isoformat() for t in m.index],
                    "y": [round(float(v), 2) for v in m.values]}

        pA, pB = profile(yA), profile(yB)
        if pA is None or pB is None:
            return None
        return {"label": f"{sensor} — {yA}", "monthaxis": True, "marks": [],
                "x": pA["x"], "y": pA["y"],
                "series2": {"label": f"{sensor} — {yB}", "x": pB["x"], "y": pB["y"]}}

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

    def _analog_scope(self, day, start, end, label):
        """A parsed day/period -> (freq, target_start, target_end, scope_label) for the
        "recommend" intent's analog mode, or None if the question named no day/month/year to
        find similar dates for. Only day/month/year granularities are supported (not seasons —
        their Dec-Feb wrap doesn't tile cleanly into `resample` buckets).
        """
        if day is not None and day[2] is not None:              # a specific, dated day
            m, d, y, day_label = day
            ts = pd.Timestamp(year=y, month=m, day=d)
            return "D", ts, ts, day_label
        if start is not None and end is not None:
            s, e = pd.Timestamp(start), pd.Timestamp(end)
            if s.month == 1 and s.day == 1 and e.month == 12 and e.day == 31 and s.year == e.year:
                return "YS", s, e, label or str(s.year)
            next_day = e + pd.Timedelta(days=1)
            if s.day == 1 and next_day.day == 1 and s.year == e.year and s.month == e.month:
                return "MS", s, e, label or f"{s:%B %Y}"
        return None

    def _answer(self, question: str) -> Answer:
        # Legend/marker questions ("what is the red dot?") are explained from a fixed
        # knowledge base — the meaning is a known fact, so no computation is involved.
        marker = resolve_legend_query(question)
        if marker is not None:
            return self._reply(question, "legend", marker, [],
                               computed=LEGEND_EXPLANATIONS[marker])

        intent, sensors, start, end, label, thr, day, hour = self._understand(question)

        # Did the user ask about a sensor this dataset doesn't have?
        if not sensors:
            miss = self._missing_concept(question)
            if miss:
                return Answer(question, intent,
                              f"I don't track {miss} in this dataset. Available sensors: "
                              f"{', '.join(self.df.columns)}.", None, [])
        sensor = sensors[0] if sensors else "T (degC)"  # default to temperature
        ctx = self._context_for(sensor)

        # A question that names a real sensor but no statistic keyword ("what was the temperature
        # at 3pm?") used to fall through to the generic "I couldn't map that" fallback even though
        # the sensor and scope resolved perfectly -- defaulting to "summary" (min/max/mean/std/
        # missing) answers it honestly without guessing which single statistic was meant. Only
        # applies once a sensor was actually named; a truly unroutable question still falls
        # through to the fallback below.
        if intent is None and sensors:
            intent = "summary"

        # Day-type classification ("what kind of day was 5 Jan 2014?"): surface the K-means
        # clustering through the interface (it was previously reachable only via main.py).
        if intent == "daytype":
            labels, profiles = self._day_types()
            names = clustering.day_type_labels(profiles)
            tcol = "mean_T (degC)" if "mean_T (degC)" in profiles.columns else profiles.columns[0]
            if day is not None and day[2] is not None:          # a specific, dated day
                m, d, y, day_label = day
                key = pd.Timestamp(year=y, month=m, day=d)
                if key not in labels.index:
                    return Answer(question, "daytype",
                                  f"I have no daily data for {day_label} to classify.", None, [])
                cl = clustering.cluster_membership(labels, key)
                return self._reply(question, "daytype", {"cluster": cl, "label": names[cl]},
                                   ["T (degC)"],
                                   computed=(f"{day_label} was a '{names[cl]}' day (K-means day-type "
                                             f"{cl} of {len(names)}; that type averages "
                                             f"{profiles.loc[cl, tcol]:.1f}°C)."),
                                   context=ctx,
                                   inferred="Day-types are K-means clusters over each day's "
                                            "mean/std/min/max across all sensors, coldest→warmest.")
            bits = "; ".join(f"'{names[c]}' (~{profiles.loc[c, tcol]:.1f}°C)"
                             for c in sorted(names, key=lambda c: profiles.loc[c, tcol]))
            return self._reply(question, "daytype", names, [],
                               computed=f"The record groups into {len(names)} K-means day-types: {bits}.",
                               context=ctx,
                               inferred="Ask e.g. 'what kind of day was 5 January 2014?' to "
                                        "classify a specific day.")

        # Recommend ("suggest similar dates", "recommend hot days in June"): two modes.
        # Directional mode ranks days by a named quality (hot/cold/humid/dry/windy/calm) within
        # a scope; analog mode finds other periods with the most similar overall weather to a
        # named day/month/year.
        if intent == "recommend":
            # Known adjectives (hot/cold/humid/dry/windy/calm) first, for their natural
            # phrasing; a generic "highest/lowest <sensor>" covers the other 11 sensors that
            # have no colloquial adjective (pressure, dew point, wind direction, ...).
            direction = resolve_direction(question) or resolve_generic_direction(
                question, list(self.df.columns))
            if direction is not None:
                dsensor, ascending = direction
                month_num = resolve_recurring_month(question)
                if month_num is not None:                       # "hot days in June" (no year)
                    scoped = self.df[self.df.index.month == month_num]
                    years = scoped.index.year
                    rec_scope = (f"{calendar.month_name[month_num]} (every year, "
                                  f"{years.min()}–{years.max()})") if len(scoped) else \
                                 calendar.month_name[month_num]
                elif start is not None and end is not None:     # "hot days in June 2015"
                    scoped, _ = self._slice(start, end, label)
                    rec_scope = label or "that period"
                elif day is None:                                # no scope named at all
                    scoped = self.df
                    rec_scope = "the whole record"
                else:                                             # a specific day + adjective
                    scoped = None
                    rec_scope = None
                if scoped is None or not len(scoped) or dsensor not in scoped.columns:
                    return Answer(question, intent,
                                  f"I have no data for {rec_scope or 'that period'} to rank.",
                                  None, [dsensor])
                result = qe.rank_calendar_days(scoped, self.df, dsensor, ascending, top_n=5)
                if not result["candidates"]:
                    return Answer(question, intent,
                                  f"I have no daily data for {rec_scope} to rank.", None, [dsensor])
                if (dsensor, ascending) in DIRECTION_LABELS:
                    word = DIRECTION_LABELS[(dsensor, ascending)]
                    superlative = DIRECTION_SUPERLATIVES[word]
                else:                                    # generic sensor, no natural adjective
                    word = f"low {dsensor}" if ascending else f"high {dsensor}"
                    superlative = f"lowest {dsensor}" if ascending else f"highest {dsensor}"
                lines = "; ".join(f"{qe.fmt_when(c['date'], with_time=False)} ({c['value']:.1f})"
                                   for c in result["candidates"])
                best = result["candidates"][0]
                best_when = qe.fmt_when(best["date"], with_time=False)
                if is_forward_looking_query(question):
                    # "When next will it be hot?" has no real future to search -- this is a
                    # fixed historical record ending at `self._anchor`, not a live feed, so
                    # there's no forward window to restrict to (unlike analog mode's "when
                    # next", where the user-named target date has real years of data after it).
                    # Answer honestly with the RECURRING pattern instead of pretending to point
                    # at a real future date beyond the data.
                    months = sorted({c["date"].month for c in result["candidates"]})
                    month_names = ", ".join(calendar.month_name[m] for m in months)
                    computed = (f"This is a historical record (ends {self._anchor:%d %b %Y}), so "
                                f"there's no real 'next' to point to — but going by the pattern, "
                                f"{word} days in {rec_scope} happen most around {month_names}. "
                                f"The {superlative} on record: {lines}.")
                    if not result["meets_threshold"]:
                        computed += (f" Note: {rec_scope} never gets genuinely {word} in this "
                                    f"record (record-wide {word} threshold {result['threshold']:.1f}) "
                                    f"— {best_when} ({best['value']:.1f}) is the closest it gets.")
                    return self._reply(question, "recommend", result, [dsensor], computed=computed,
                                       context=ctx)
                if result["meets_threshold"]:
                    computed = (f"The {superlative} days in {rec_scope}: {lines}. Best pick: "
                                f"{best_when} ({best['value']:.1f}) — go with that one.")
                else:
                    computed = (f"{rec_scope} isn't genuinely {word} in this record (the record-wide "
                                f"{word} threshold is {result['threshold']:.1f}) — these are just the "
                                f"closest it gets, not a real {word} recommendation: {lines}. If you "
                                f"must pick one, {best_when} ({best['value']:.1f}) is the closest it gets, "
                                f"but expect nothing like real {word} weather.")
                return self._reply(question, "recommend", result, [dsensor], computed=computed,
                                   context=ctx)

            # The TARGET (the weather being matched) is normally whatever `_understand` already
            # parsed. But when a "similar to X" clause is present AND neither named period has a
            # day number (so `resolve_day` can't disambiguate by day-adjacency), the marker
            # clause overrides -- "a day to go out in January 2010 similar to June 2010" would
            # otherwise resolve to January (dict order: jan sorts before jun), even though "similar
            # to June" unambiguously means June is the target, not January.
            clause = split_target_clause(question)
            target_day, target_start, target_end, target_label = day, start, end, label
            if clause is not None:
                clause_day = resolve_day(clause)
                if clause_day is not None:
                    target_day, target_start, target_end, target_label = clause_day, None, None, ""
                else:
                    c_start, c_end, c_label = resolve_period(clause)
                    if c_start or c_end:
                        target_day, target_start, target_end, target_label = None, c_start, c_end, c_label

            scope_info = self._analog_scope(target_day, target_start, target_end, target_label)
            if scope_info is None:
                return Answer(question, intent,
                              "I need a specific day, month, or year to find similar dates for — "
                              "try e.g. 'recommend dates similar to 13 February 2013'.", None, sensors)
            freq, t_start, t_end, rec_scope = scope_info

            # A second, explicit period named anywhere in the question -- same period restated
            # ("...6 April 2013 in the month of April 2013"), a genuinely different one ("...5
            # September 2012 in November 2012", even a different year: "...in June 2016"), or the
            # OTHER half of a "similar to" split ("January 2010 similar to June 2010" -> June is
            # the target above, January is the restriction found here) -- means search only
            # inside THAT period, not the whole record. A phrase like "staying within that month"
            # with no literal second date falls back to the target's own month/year. A bare date
            # (nothing else named) searches everywhere.
            search_start = search_end = restrict_label = None
            if target_day is not None and target_day[2] is not None:
                t_month, t_year = target_day[0], target_day[2]
            elif t_start is not None:
                # a month-level target has a real month to strip; a year-level one (freq="YS")
                # has none -- month=0 never matches a real MONTHS entry (1-12), so nothing is
                # spuriously stripped from the search text for a bare-year target.
                t_month, t_year = (pd.Timestamp(t_start).month if freq == "MS" else 0), pd.Timestamp(t_start).year
            else:
                t_month, t_year = 0, None
            if t_year is not None:
                secondary = resolve_secondary_period(question, t_month, t_year)
                if secondary is not None:
                    search_start, search_end, restrict_label = secondary
                elif resolve_explicit_scope_restriction(question):
                    # Anaphoric restriction ("staying within that month") names no second date
                    # to parse -- fall back to the MONTH CONTAINING the target's own date (not
                    # `target_start/target_end`, which are None whenever the target came from a
                    # specific day rather than a bare month/year).
                    month_start = pd.Timestamp(t_start).replace(day=1)
                    month_end = month_start + pd.offsets.MonthEnd(0)
                    search_start, search_end = month_start, month_end
                    restrict_label = f"{month_start:%B %Y}"

            # "When next do we get conditions like this?" is asking for SOON, not "sometime in
            # the next 8 years, possibly a different season" -- restrict to a near-term forward
            # window by default (no restriction named), or make a named restriction forward-only.
            forward_looking = is_forward_looking_query(question) and target_day is not None \
                and target_day[2] is not None
            if forward_looking:
                target_ts = pd.Timestamp(year=target_day[2], month=target_day[0], day=target_day[1])
                if search_start is None:
                    search_start = target_ts + pd.Timedelta(days=1)
                    search_end = target_ts + pd.DateOffset(months=3)
                    restrict_label = f"the 3 months after {rec_scope}"
                else:
                    search_start = max(pd.Timestamp(search_start), target_ts + pd.Timedelta(days=1))

            # A restriction means "find THE DAY within it" -- day-level candidates regardless of
            # whether the target itself was named at month/year granularity.
            search_freq = "D" if search_start is not None else freq
            wanted = [c for c in (sensors or
                                   ["T (degC)", "rh (%)", "Tdew (degC)", "p (mbar)", "wv (m/s)"])
                      if c in self.df.columns]
            result = qe.find_analog_periods(self.df, wanted, t_start, t_end, search_freq, top_n=5,
                                            search_start=search_start, search_end=search_end)
            if not result["candidates"]:
                if forward_looking:
                    return Answer(question, intent,
                                  f"Nothing close to {rec_scope} shows up in {restrict_label} — "
                                  f"the record may not run that far forward, or that stretch is "
                                  f"missing data.", None, wanted)
                where = f"within {restrict_label} " if restrict_label else ""
                return Answer(question, intent,
                              f"I don't have enough other periods {where}to compare {rec_scope} against.",
                              None, wanted)
            def _label(ts):
                if search_freq == "YS":
                    return str(ts.year)
                if search_freq == "MS":
                    return f"{ts:%B %Y}"
                return qe.fmt_when(ts, with_time=False)

            lines = "; ".join(_label(c["when"]) for c in result["candidates"])
            noun = {"D": "Dates", "MS": "Months", "YS": "Years"}[search_freq]
            best = result["candidates"][0]
            noun_singular = {"D": "day", "MS": "month", "YS": "year"}[search_freq]
            if forward_looking:
                computed = (f"The next time conditions look like {rec_scope} is likely around "
                            f"{_label(best['when'])} (closest match in {restrict_label}). Other "
                            f"close days coming up: {lines}.")
            else:
                within = f" (searched within {restrict_label} only)" if restrict_label else ""
                computed = (f"{noun} with the most similar weather to {rec_scope}{within}: {lines}. "
                            f"Closest match: {_label(best['when'])} — if you want conditions like "
                            f"{rec_scope}, that's the {noun_singular} to go with.")

            # Honesty check, restricted searches only: is the closest match actually close on
            # EVERY requested sensor, or just the least-bad option that's badly off on at least
            # one (e.g. "January similar to June" restricted to January -- January never gets
            # June-hot)? Checking only the first requested sensor would miss a restriction that's
            # fine on temperature but badly wrong on humidity -- so every sensor in `wanted` is
            # compared against the whole record's own natural spread for THAT sensor, and the
            # worst offender(s) are named. The bar is always data-derived, never invented.
            if restrict_label and result["target"]:
                mismatches = []
                for s in wanted:
                    if s not in result["target"] or s not in best:
                        continue
                    target_value, best_value = result["target"][s], best[s]
                    gap = abs(target_value - best_value)
                    full_std = float(self.df[s].std())
                    if full_std and gap > full_std:
                        mismatches.append((s, target_value, best_value, gap, full_std))
                if mismatches:
                    mismatches.sort(key=lambda m: m[3] / m[4], reverse=True)  # worst first
                    bits = "; ".join(
                        f"{s} — {rec_scope} averaged {tv:.1f}, best {restrict_label} offers "
                        f"{bv:.1f} (a {g:.1f} gap, more than the record's typical {sd:.1f} spread)"
                        for s, tv, bv, g, sd in mismatches)
                    computed += (f" Honest caveat: don't count on {restrict_label} for genuinely "
                                f"{rec_scope}-like conditions — {bits}.")
            return self._reply(question, "recommend", result, wanted, computed=computed, context=ctx)

        # Causal questions ("does X cause Y?", "what causes…?"): we measure correlation, not
        # causation — say so explicitly rather than pass a correlation off as an answer.
        causal = is_causal_query(question)
        if causal and len(sensors) < 2:
            return self._reply(question, "correlation", None, sensors,
                               computed=("This system measures correlation (how sensors move together), "
                                         "not causation — it can't say what *causes* a change; that needs "
                                         "controlled experiments or causal inference, which are out of "
                                         "scope. Ask 'how does X relate to Y?' for their correlation, or "
                                         "ask about anomalies to see which sensors moved together."),
                               context=ctx)
        if causal:
            intent = "correlation"   # answer with the correlation, plus the causation caveat below

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

        if hour[0] is not None:
            data = data[hour_mask(data.index, hour[0], hour[1])]
            where = f"{where} at {hour[2]}" if where else f" at {hour[2]}"
            if not len(data):
                return Answer(question, intent,
                              f"I have no data for {hour[2]}{(' ' + scope) if scope else ''}.",
                              None, [sensor])

        # Native-resolution slice for the exact point statistics (max/min/mean/std/summary/weather).
        # Same window as `data`; falls back to the hourly slice if the raw frame has no rows there
        # (only possible when `self.raw` is the hourly frame anyway, i.e. no raw was supplied).
        raw_data = self._scope_raw(day, start, end, label, hour)
        if not len(raw_data):
            raw_data = data

        # Weather report: a multi-sensor snapshot for the chosen day/period.
        if intent == "weather":
            wanted = [c for c in ("T (degC)", "rh (%)", "Tdew (degC)", "p (mbar)", "wv (m/s)")
                      if c in data.columns]
            rep = qe.weather_report(raw_data, wanted)
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
            # A count is exact, but on a compass bearing it is a literal wrap-unaware comparison.
            caveat = (" (note: wind direction is a compass bearing that wraps at 360°, so this is a "
                      "literal angular comparison, not a directional 'more/less')"
                      if sensor in analysis.CIRCULAR_SENSORS else "")
            return self._reply(question, "threshold", r, [sensor],
                               computed=(f"{sensor} was {sign} {value:g}{where} "
                                         f"{r['count']} times ({r['pct']:.2f}% of readings){when}.{caveat}"),
                               context=ctx)

        if intent == "forecast":
            if sensor in analysis.CIRCULAR_SENSORS:
                return self._reply(question, intent, None, [sensor],
                                   computed=(f"{sensor} is a circular compass bearing (0–360°); it "
                                             "can't be forecast with a linear time-series model here — "
                                             "the wrap at 360° breaks the model."), context=ctx)
            try:
                fc = forecasting.forecast(data[sensor], steps=14)
            except Exception as exc:
                logger.warning("forecast failed for %s (%s).", sensor, exc)
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
            if sensor in analysis.CIRCULAR_SENSORS:
                return self._reply(question, intent, None, [sensor],
                                   computed=(f"{sensor} is a circular compass bearing (0–360°); a "
                                             "linear trend is not meaningful on it — the 0°/360° wrap "
                                             "makes the slope arbitrary, so direction has no straight-"
                                             "line 'rising/falling'."), context=ctx)
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

        # Circular sensors (wind direction) wrap at 360°, so linear max/min/mean/std are
        # invalid — route them through circular statistics instead (a linear "max" just
        # reports ~359° regardless of the actual prevailing wind).
        if sensor in analysis.CIRCULAR_SENSORS and intent in ("max", "min", "mean", "spread"):
            cs = analysis.circular_stats_deg(data[sensor])
            prevailing = analysis.compass_point(cs["mean"])
            if intent in ("max", "min"):
                computed = (f"{sensor} is a circular compass bearing (0–360°), so a plain "
                            f"{'maximum' if intent == 'max' else 'minimum'} is meaningless "
                            f"(359° and 1° are 2° apart, not 358°). Prevailing direction{where} "
                            f"is {cs['mean']:.1f}° ({prevailing}), circular spread {cs['std']:.1f}°.")
            else:
                computed = (f"{sensor}{where}: prevailing (circular mean) direction "
                            f"{cs['mean']:.1f}° ({prevailing}), circular std {cs['std']:.1f}°.")
            return self._reply(question, intent, cs, [sensor], computed=computed, context=ctx)

        if intent == "max":
            r = qe.get_max(raw_data, sensor)
            return self._reply(question, intent, r, [sensor],
                               computed=f"max {sensor}{where} was {r['value']:.2f} on {qe.fmt_when(r['when'])}.", context=ctx)

        if intent == "min":
            r = qe.get_min(raw_data, sensor)
            return self._reply(question, intent, r, [sensor],
                               computed=f"min {sensor}{where} was {r['value']:.2f} on {qe.fmt_when(r['when'])}.", context=ctx)

        if intent in ("mean", "spread"):
            r = qe.get_mean_std(raw_data, sensor)
            return self._reply(question, intent, r, [sensor],
                               computed=f"{sensor}{where} mean={r['mean']:.2f}, std={r['std']:.2f}, range={r['range']:.2f}.",
                               context=ctx)

        if intent == "correlation":
            if len(sensors) < 2:
                sensors = (sensors + ["T (degC)", "rho (g/m**3)"])[:2]
            a, b = sensors[0], sensors[1]
            cause_note = (" Note: this is correlation, not causation — the system measures "
                          "association, not cause." if causal else "")
            circ = [s for s in (a, b) if s in analysis.CIRCULAR_SENSORS]
            if len(circ) == 1:  # one angle + one linear sensor -> circular–linear correlation
                deg = circ[0]
                lin = b if deg == a else a
                c = analysis.circular_linear_corr(data[deg], data[lin])
                strength = correlation_strength(c)
                return self._reply(question, intent, c, [a, b],
                                   computed=f"circular–linear correlation({deg}, {lin}){where} = {c:.3f} "
                                            f"({strength}; 0–1, unsigned; {deg} is a compass angle)."
                                            f"{cause_note}",
                                   inferred="Wind direction wraps at 360°, so this uses a circular "
                                            "measure of association, not linear Pearson.")
            c = qe.get_correlation(data, a, b)
            rel = "move together" if c > 0 else "move in opposite directions"
            if c == 0:
                strength_sign = "no correlation"
            else:
                strength_sign = f"{correlation_strength(c)} {'positive' if c > 0 else 'negative'} correlation"
            computed = (f"correlation({a}, {b}){where} = {c:.3f} ({strength_sign}).{cause_note}")
            if is_correlation_breakdown_query(question):
                # "why is that only moderate?" -- explain the NUMBER's own magnitude with real
                # computed evidence, not a canned line: how much variance it linearly explains,
                # whether a nonlinear pattern is being missed, and whether the relationship
                # actually holds steady across the year or is really a few-season effect that
                # whole-record pooling averages into one middling figure.
                bd = qe.correlation_breakdown(data, a, b)
                bits = [f"R²={bd['r_squared']:.2f} ({a} 'explains' about {bd['r_squared'] * 100:.0f}% "
                        f"of {b}'s variance in a straight line — the rest is other factors)"]
                if abs(bd["spearman"] - bd["pearson"]) < 0.1:
                    bits.append(f"rank correlation={bd['spearman']:.3f}, close to the Pearson value, "
                                "so this is genuinely close to linear, not a stronger pattern being missed")
                else:
                    bits.append(f"rank correlation={bd['spearman']:.3f} — noticeably different from "
                                "Pearson, hinting at a real but non-linear relationship")
                if len(bd["by_season"]) >= 2:
                    season_bits = "; ".join(f"{s} {v:+.2f}" for s, v in bd["by_season"].items())
                    spread = max(bd["by_season"].values()) - min(bd["by_season"].values())
                    if spread > 0.2:
                        bits.append(f"by season: {season_bits} — the strength varies a lot across "
                                    "the year, so the whole-record number is an average of quite "
                                    "different regimes, not one steady relationship")
                    else:
                        bits.append(f"by season: {season_bits} — fairly steady across the year, so "
                                    "this isn't a seasonal-averaging effect")
                computed += " Breakdown: " + "; ".join(bits) + "."
            return self._reply(question, intent, c, [a, b], computed=computed, inferred=f"They {rel}.")

        if intent == "missing":
            n = qe.count_missing(data, sensor)
            return self._reply(question, intent, n, [sensor],
                               computed=f"{sensor}{where} has {n} missing values.", context=ctx)

        if intent == "anomaly":
            # Network-level detection: the chosen detector reads the whole in-scope sensor frame and a
            # window is anomalous if the sensor *network* looks wrong (a channel inconsistent with its
            # neighbours, or a frozen sensor) — not just one sensor in isolation. The detector is
            # selectable (portfolio/mahalanobis/isolation_forest/lof/ocsvm); all are deterministic code.
            det = self.detector
            fn = NETWORK_DETECTORS.get(det)
            if fn is None:
                valid = ", ".join(NETWORK_DETECTORS)
                return Answer(question, "anomaly",
                              f"Unknown detector {det!r}. Choose one of: {valid}.", None, [sensor],
                              detector=det)
            flags = fn(data)
            windows = qe.get_anomaly_windows(flags)
            # Multi-sensor reasoning + attribution: explain *why* the first flagged moment looked
            # unusual (what every sensor did) and name the sensor most responsible.
            inferred = culprit = None
            if windows:
                try:
                    flat_sensor = anomaly.flatline_culprit(data, windows[0]["start"])
                    inferred = reasoning.describe_network_moment(
                        data, windows[0]["start"], flatline_sensor=flat_sensor)
                except Exception as exc:
                    logger.warning("reasoning failed at %s (%s).", windows[0]["start"], exc)
                    inferred = None
                culprit = anomaly.network_culprit(data, windows[0]["start"])
            n_sensors = data.shape[1]
            first = (f"First cross-sensor anomaly on {qe.fmt_when(windows[0]['start'])}"
                     f"{f' (driven by {culprit})' if culprit else ''}." if windows else None)
            ans = self._reply(question, intent, windows, [sensor],
                              computed=f"{int(flags.sum())} cross-sensor anomaly windows{where} "
                                       f"across {n_sensors} sensors (detector: {det}).",
                              context=first or ctx, inferred=inferred)
            ans.network = windows
            ans.detector = det
            return ans

        if intent == "summary":
            s = qe.sensor_summary(raw_data, sensor)
            return self._reply(question, intent, s, [sensor],
                               computed=(f"{sensor}{where}: min={s['min']:.2f}, max={s['max']:.2f}, "
                                         f"mean={s['mean']:.2f}, std={s['std']:.2f}, missing={s['missing']}."),
                               context=ctx)

        # Nothing routed: neither an intent nor a sensor was found. Instead of a raw error, use a
        # semantic search to tell an on-topic-but-unclear question (-> targeted suggestion) from an
        # off-topic one (-> polite redirect). See DOMAIN_THRESHOLD / SENSOR_HINTS above.
        return self._semantic_fallback(question)


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
