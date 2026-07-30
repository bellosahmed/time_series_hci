"""Natural-language parsing for the interface — the pure, offline text layer.

Everything here is deterministic and side-effect-free: it turns a plain-English question
into structured pieces (intent, sensors, period, day, comparison, threshold) using keyword
and regex matching only. No model, no data, no I/O — which makes it independently testable
and keeps `interface.py` focused on orchestration.

Split out of `interface.py` (which had grown to ~585 lines) so parsing and routing are
separate concerns. `interface` re-exports these names, so `interface.resolve_sensors` etc.
still work.
"""

from __future__ import annotations

import difflib
import re

import pandas as pd

# Map everyday words/phrases to dataset columns — covers ALL 14 sensors so each is nameable
# in plain English. Matching is longest-phrase-first with span consumption (see
# `resolve_sensors`), so a specific multi-word alias ("vapour pressure deficit", "wind
# direction", "specific humidity") wins over a shorter generic word nested inside it
# ("pressure", "wind", "humidity"). Both British/American spellings of vapo(u)r are accepted.
SENSOR_ALIASES = {
    # temperature family
    "temperature": "T (degC)", "temp": "T (degC)", "hot": "T (degC)", "cold": "T (degC)",
    "potential temperature": "Tpot (K)", "potential": "Tpot (K)",
    "dew point": "Tdew (degC)", "dewpoint": "Tdew (degC)", "dew": "Tdew (degC)",
    # humidity / water vapour -- "muggy"/"clammy"/"sticky"/"damp" match the colloquial vocabulary
    # already used in rag_system's own rh (%) description (found missing by query_coverage.py)
    "relative humidity": "rh (%)", "humidity": "rh (%)", "humid": "rh (%)",
    "muggy": "rh (%)", "clammy": "rh (%)", "sticky": "rh (%)", "damp": "rh (%)",
    "specific humidity": "sh (g/kg)",
    "water vapour concentration": "H2OC (mmol/mol)", "water vapor concentration": "H2OC (mmol/mol)",
    "water vapour": "H2OC (mmol/mol)", "water vapor": "H2OC (mmol/mol)",
    # vapour pressures (specific phrases before the generic "vapour pressure")
    "saturation vapour pressure": "VPmax (mbar)", "saturation vapor pressure": "VPmax (mbar)",
    "vapour pressure deficit": "VPdef (mbar)", "vapor pressure deficit": "VPdef (mbar)",
    "actual vapour pressure": "VPact (mbar)", "actual vapor pressure": "VPact (mbar)",
    "vapour pressure": "VPact (mbar)", "vapor pressure": "VPact (mbar)",
    # pressure / density -- "heavy"/"thick"/"dense"/"thin" match rag_system's rho description
    "pressure": "p (mbar)",
    "air density": "rho (g/m**3)", "density": "rho (g/m**3)",
    "heavy": "rho (g/m**3)", "thick": "rho (g/m**3)", "dense": "rho (g/m**3)", "thin": "rho (g/m**3)",
    # wind (direction/gust phrases before the generic "wind") -- "breezy"/"blowy"/"gusty" match
    # rag_system's wv (m/s) description
    "wind direction": "wd (deg)", "direction": "wd (deg)",
    "wind gust": "max. wv (m/s)", "gust": "max. wv (m/s)",
    "wind speed": "wv (m/s)", "wind": "wv (m/s)",
    "breezy": "wv (m/s)", "blowy": "wv (m/s)", "gusty": "wv (m/s)",
}

# --- Typo tolerance -------------------------------------------------------------------------
# The router matches intent/sensor keywords EXACTLY, so a misspelling ("tempreature") finds
# nothing. `spell_normalize` fixes obvious typos of domain words BEFORE routing, but stays
# deliberately conservative so it never silently changes the meaning of a question:
#   * only words of length >= 5 are corrected, and only ONTO vocabulary words of length >= 5.
#     This keeps the short, easily-confused intent keywords (max/min/hot/cold/top/vs) exact-only
#     -- fuzzy-mapping a typo to "max" vs "min" could flip the answer.
#   * a high similarity cutoff (0.82) means only near-misses are touched; unknown real words
#     (e.g. "during", "rainfall") are left alone.
#   * every correction is returned to the caller, which surfaces it ("read 'tempreature' as
#     'temperature'") -- the guess is shown, never hidden, preserving the auditable-routing goal.
_SPELL_MIN_LEN = 5
# 0.85, not lower: single-character typos of the domain words all score >= 0.857
# ("maximun"->"maximum", "avearge"->"average"), while the nearest false positive
# ("monthly"->"month", a real word that leaks in from the "next month" intent phrase) sits at
# 0.833 and is correctly rejected. Erring toward missing a correction beats silently rewriting a
# legitimate word.
_SPELL_CUTOFF = 0.85
_SPELL_VOCAB: set[str] | None = None


def _spell_vocab() -> set[str]:
    """The set of domain words a typo may be corrected onto (length >= _SPELL_MIN_LEN), cached.

    Built from the sensor aliases and intent keywords defined below, so it always tracks the
    real routing vocabulary -- add a sensor alias and its typos become correctable automatically.
    """
    global _SPELL_VOCAB
    if _SPELL_VOCAB is None:
        vocab: set[str] = set()
        for alias in SENSOR_ALIASES:
            vocab.update(w for w in alias.split() if len(w) >= _SPELL_MIN_LEN)
        for keywords in INTENTS.values():
            for kw in keywords:
                vocab.update(w for w in kw.split() if len(w) >= _SPELL_MIN_LEN and w.isalpha())
        # Month + season names so a misspelled *period* ("octber", "septmber", "wnter") is repaired
        # and the scope still resolves. Only names >= _SPELL_MIN_LEN qualify, so the short months
        # (may/jun/jul and the 3-letter abbreviations) stay exact-only.
        vocab.update(w for w in MONTHS if len(w) >= _SPELL_MIN_LEN)
        vocab.update(w for w in SEASONS if len(w) >= _SPELL_MIN_LEN)
        _SPELL_VOCAB = vocab
    return _SPELL_VOCAB


def spell_normalize(question: str) -> tuple[str, list[tuple[str, str]]]:
    """Return (corrected_question, corrections) where corrections is a list of (original, fixed).

    Conservatively repairs typos of domain words; leaves numbers, months, and any word it isn't
    confident about untouched (see the module note above for the safety rules).
    """
    vocab = _spell_vocab()
    corrections: list[tuple[str, str]] = []

    def fix(m: re.Match[str]) -> str:
        tok = m.group(0)
        low = tok.lower()
        if len(low) < _SPELL_MIN_LEN or not low.isalpha() or low in vocab:
            return tok
        near = difflib.get_close_matches(low, vocab, n=1, cutoff=_SPELL_CUTOFF)
        if near and near[0] != low:
            corrections.append((tok, near[0]))
            return near[0]
        return tok

    return re.sub(r"[A-Za-z]+", fix, question), corrections


# A word must be at least this similar to a known term to be offered as a "did you mean?" hint.
# Lower than the auto-correct cutoff (0.85): a moderate match is worth *suggesting* to the user even
# when it is too weak to apply silently -- "tocber" -> "october" (0.77) is a good suggestion, a bad
# silent correction.
_SPELL_SUGGEST_CUTOFF = 0.6


def spell_suggest(word: str) -> str | None:
    """The closest known domain word to `word` above _SPELL_SUGGEST_CUTOFF, else None.

    Used to turn "I didn't understand X" into "did you mean Y?" without ever auto-applying Y.
    """
    near = difflib.get_close_matches(word.lower(), _spell_vocab(), n=1, cutoff=_SPELL_SUGGEST_CUTOFF)
    return near[0] if near and near[0] != word.lower() else None


# Common non-domain words that carry no scope/intent/sensor meaning — question words, articles,
# prepositions, and generic time/measure nouns. A leftover word NOT in here (and not a
# sensor/intent/month/season term) is one the router silently dropped, so it's worth flagging.
_FILLER_WORDS = {
    "what", "whats", "which", "when", "where", "there", "here", "this", "that", "these", "those",
    "how", "why", "was", "were", "is", "are", "the", "for", "and", "with", "from", "about",
    "tell", "show", "give", "please", "kindly", "could", "would", "can", "get", "got", "you",
    "your", "does", "did", "over", "been", "have", "has", "had", "during", "between", "than",
    "versus", "any", "all", "some", "most", "least", "value", "values", "reading", "readings",
    "data", "record", "records", "level", "levels", "amount", "point", "much", "many", "into",
    "day", "days", "daily", "week", "weeks", "weekly", "month", "months", "monthly", "year",
    "years", "yearly", "time", "times", "date", "dates", "season", "seasons", "hour", "hours",
    "morning", "afternoon", "evening", "night", "noon", "midnight", "today", "tomorrow",
    "yesterday", "period", "range", "whole", "entire", "across", "throughout",
    # hedges / politeness / connectors — benign filler that must not trip the clarification guardrail
    "around", "roughly", "approximately", "exactly", "actually", "really", "maybe",
    "perhaps", "quite", "rather", "thanks", "thank", "want", "need", "like", "looking",
    "find", "output", "result", "results", "figure", "number", "stats", "statistics", "info",
    "information", "overall", "total", "recorded", "measured", "observed",
}
_KNOWN_WORDS: set[str] | None = None


def _known_words() -> set[str]:
    """All words the router understands: sensor aliases, intent keywords, months, seasons,
    comparison words, plus harmless filler. Cached. Anything else (>= 5 chars) in a question was
    not consumed by routing — a candidate 'ignored word' to surface."""
    global _KNOWN_WORDS
    if _KNOWN_WORDS is None:
        known = set(_FILLER_WORDS)
        for alias in SENSOR_ALIASES:
            known.update(alias.split())
        for keywords in INTENTS.values():
            for kw in keywords:
                known.update(w for w in kw.split() if w.isalpha())
        known.update(MONTHS)
        known.update(SEASONS)
        for phrases in COMPARE.values():
            for ph in phrases:
                known.update(ph.split())
        _KNOWN_WORDS = known
    return _KNOWN_WORDS


# The clarification guardrail fires only in this gray zone: a word similar enough to a domain term
# to be a probable typo, but below the 0.85 auto-correct line. Measured separation: genuine typos
# score >= 0.77 to their nearest term ("tocber"->october 0.77, "lowset"->lowest 0.83), while
# ordinary English words that merely appear in a query ("because" 0.67, "degrees" 0.67, "staying"
# 0.71, "nonsense" 0.62) all sit <= 0.71 -- so 0.75 cleanly splits "misspelled domain word, ask the
# user" from "just an unrelated word, let normal routing/fallback handle it".
_SPELL_GUARD_FLOOR = 0.75


def clarifications_needed(question: str, columns: tuple[str, ...] = ()) -> list[tuple[str, str]]:
    """Words that look like a misspelled domain term (period/sensor/intent) but are too uncertain to
    auto-correct — each with its best-guess match. Returns ``[(word, suggestion), ...]``.

    These are the only words worth *stopping to ask about*: close to a known term (>= 0.75) yet
    below the auto-correct cutoff (0.85). Words far from any domain term are ordinary language, not
    typos, and are deliberately excluded so the guardrail never blocks a normal question.
    """
    vocab = _spell_vocab()
    known = set(_known_words())
    for col in columns:
        known.update(re.findall(r"[a-z]+", col.lower()))
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for tok in re.findall(r"[A-Za-z]+", question.lower()):
        if len(tok) < _SPELL_MIN_LEN or tok in known or tok in seen:
            continue
        seen.add(tok)
        match = difflib.get_close_matches(tok, vocab, n=1, cutoff=_SPELL_GUARD_FLOOR)
        if match and match[0] != tok:
            ratio = difflib.SequenceMatcher(None, tok, match[0]).ratio()
            if _SPELL_GUARD_FLOOR <= ratio < _SPELL_CUTOFF:
                out.append((tok, match[0]))
    return out


def unrecognized_words(question: str, columns: tuple[str, ...] = ()) -> list[str]:
    """Meaningful words (>= 5 letters) the router does not recognize — i.e. that it silently
    ignored. Returns them in order, de-duplicated, so the caller can warn the user rather than
    quietly answer a broader/different question than they asked. Short filler-length typos
    (\"waht\", \"si\") are below the length cutoff and not reported.

    `columns` (the dataset's raw column names) are folded into the known set so a user who names a
    column or unit directly ("vpmax", "mbar") is not told it was ignored.
    """
    known = set(_known_words())
    for col in columns:
        known.update(re.findall(r"[a-z]+", col.lower()))
    out: list[str] = []
    for tok in re.findall(r"[A-Za-z]+", question.lower()):
        if len(tok) >= 5 and tok not in known and tok not in out:
            out.append(tok)
    return out


INTENTS = {
    # recommend is BEFORE max/min/trend so "recommend the hottest days" routes here, not to
    # max (they share vocabulary like "hottest"/"warmer" since recommending IS a kind of ranking)
    "recommend": ["recommend", "recommendation", "recommends", "suggest", "suggestion",
                  "suggests", "similar to", "similar day", "similar days", "similar date",
                  "similar dates", "similar month", "similar months", "similar year",
                  "similar years", "similar conditions", "similar weather", "analogous",
                  "other dates", "other days like", "like this day", "like this month",
                  "like this year", "closest to", "when next", "when again", "next time",
                  "when will we", "when do we get", "how soon"],
    "max": ["max", "maximum", "highest", "hottest", "peak", "warmest", "top", "biggest", "most",
            "hot", "record high"],
    "min": ["min", "minimum", "lowest", "coldest", "smallest", "least", "cold", "record low"],
    "mean": ["mean", "average", "typical", "normally", "usual"],
    "spread": ["std", "deviation", "variability", "variable", "spread", "variance", "vary",
               "fluctuate"],
    "correlation": ["correlation", "correlate", "related", "relationship", "relate", "versus",
                    " vs ", "depend", "affect", "influence", "connection", "connected", "link",
                    "linked", "associated", "associate"],
    "missing": ["missing", "gap", "gaps", "absent", "incomplete"],
    "forecast": ["forecast", "predict", "prediction", "projection", "next week", "next month",
                 "next few days", "coming days", "future", "will it", "going to be", "going forward",
                 "outlook"],
    "trend": ["trend", "trending", "rising", "falling", "increasing", "decreasing", "increase",
              "decrease", "warming", "cooling", "going up", "going down", "over time",
              "warmer", "colder", "hotter", "cooler"],
    "compare": ["compare", "comparison", "versus", "vs", "difference between"],
    "anomaly": ["anomaly", "anomalies", "unusual", "abnormal", "weird", "strange", "outlier",
                "outliers", "odd", "spike", "spikes", "anomalous", "wrong", "problem"],
    "weather": ["weather", "conditions", "climate"],
    # daytype is BEFORE summary so "describe the day-types" routes here, not to summary
    "daytype": ["day type", "day types", "day-type", "day-types", "type of day", "types of day",
                "kind of day", "kinds of day", "sort of day", "day category", "categorise the day",
                "categorize the day", "classify the day"],
    "summary": ["summary", "summarise", "summarize", "describe", "overview", "tell me about"],
}


def resolve_sensors(question: str, columns) -> list[str]:
    """Find which dataset columns a question refers to (via aliases + exact names).

    Aliases are matched **longest phrase first**, and the character span each match covers is
    then consumed, so a specific multi-word alias wins over a shorter generic word nested
    inside it — e.g. "wind direction" -> ``wd (deg)`` (not wind speed), "vapour pressure" ->
    ``VPact (mbar)`` (not pressure), "specific humidity" -> ``sh (g/kg)`` (not humidity). This
    lets every one of the 14 sensors be named in plain English without a generic word stealing
    the match. Returns the matched columns in order of first appearance in the question.
    """
    q = question.lower()
    consumed = [False] * len(q)
    hits: list[tuple[int, str]] = []  # (position, column)
    for word in sorted(SENSOR_ALIASES, key=len, reverse=True):
        col = SENSOR_ALIASES[word]
        if col not in columns:
            continue
        # \b word boundaries -- plain substring search let short aliases fire inside unrelated
        # words, e.g. "thin" (-> air density) matching inside "within" ("staying WITHIN April"),
        # corrupting sensor detection on any "recommend" restriction phrase that used the word.
        for m in re.finditer(rf"\b{re.escape(word)}\b", q):
            start, end = m.start(), m.end()
            if not any(consumed[start:end]):
                for i in range(start, end):
                    consumed[i] = True
                hits.append((start, col))
                break
    found: list[str] = []
    for _, col in sorted(hits):  # order by appearance in the question
        if col not in found:
            found.append(col)
    for col in columns:  # also catch exact column names mentioned (e.g. "rh (%)")
        if col.lower() in q and col not in found:
            found.append(col)
    return found


# Directional adjectives for the "recommend" intent's ranking mode ("recommend hot days in
# June") -> (sensor, ascending). ascending=False means rank for the HIGHEST values (hot,
# humid, windy); ascending=True means rank for the LOWEST (cold, dry, calm).
DIRECTION_WORDS = {
    "hot": ("T (degC)", False), "hottest": ("T (degC)", False), "warm": ("T (degC)", False),
    "warmest": ("T (degC)", False), "warmer": ("T (degC)", False),
    "cold": ("T (degC)", True), "coldest": ("T (degC)", True), "cool": ("T (degC)", True),
    "coolest": ("T (degC)", True), "cooler": ("T (degC)", True), "chilly": ("T (degC)", True),
    "humid": ("rh (%)", False), "muggy": ("rh (%)", False), "clammy": ("rh (%)", False),
    "sticky": ("rh (%)", False), "damp": ("rh (%)", False),
    "dry": ("rh (%)", True), "driest": ("rh (%)", True),
    "windy": ("wv (m/s)", False), "gusty": ("wv (m/s)", False), "breezy": ("wv (m/s)", False),
    "blowy": ("wv (m/s)", False),
    "calm": ("wv (m/s)", True), "calmest": ("wv (m/s)", True), "still": ("wv (m/s)", True),
}

# The canonical word to describe each (sensor, ascending) pair back to the user, regardless of
# which specific synonym ("warmest"/"hot"/"warmer") they actually typed.
DIRECTION_LABELS = {
    ("T (degC)", False): "hot", ("T (degC)", True): "cold",
    ("rh (%)", False): "humid", ("rh (%)", True): "dry",
    ("wv (m/s)", False): "windy", ("wv (m/s)", True): "calm",
}

# Superlative phrasing for each label — English irregulars ("hot"->"hottest", not "hotest")
# mean this can't be built by string concatenation.
DIRECTION_SUPERLATIVES = {
    "hot": "hottest", "cold": "coldest", "humid": "most humid", "dry": "driest",
    "windy": "windiest", "calm": "calmest",
}


def resolve_direction(question: str) -> tuple[str, bool] | None:
    """A directional adjective ("hot", "driest", "calm"...) -> (sensor, ascending), else None."""
    q = question.lower()
    for word, pair in DIRECTION_WORDS.items():
        if re.search(rf"\b{word}\b", q):
            return pair
    return None


# Generic direction words that work with ANY named sensor, not just the T/rh/wv adjectives
# above -- covers the other 11 sensors (pressure, dew point, wind direction, specific humidity,
# vapour pressures, air density, potential temperature) which have no natural colloquial
# adjective ("recommend the highest-pressure days", "days with the lowest dew point").
GENERIC_DIRECTION_WORDS = {
    "highest": False, "greatest": False, "most": False, "max": False, "maximum": False,
    "top": False, "biggest": False, "largest": False,
    "lowest": True, "least": True, "min": True, "minimum": True, "smallest": True,
}


_FORWARD_LOOKING_PHRASES = ("when next", "when again", "next time", "when will", "when do we",
                            "how soon", "when's the next", "when is the next", "when do we get")


def is_forward_looking_query(question: str) -> bool:
    """True for "when next do we get conditions like X?" style questions.

    A human asking this means "soon" -- not "sometime in the next 8 years, possibly a
    different season entirely" -- so the "recommend" intent's analog mode restricts to a
    near-term forward window by default when this fires (see `interface.py`), rather than the
    plain "closest match anywhere in the record, any direction in time" analog search.
    """
    q = question.lower()
    return any(p in q for p in _FORWARD_LOOKING_PHRASES)


_TARGET_CLAUSE_MARKERS = ("similar to", "similar weather to", "similar conditions to",
                          "comparable to", "resembling", "resembles")


def split_target_clause(question: str) -> str | None:
    """The substring right after a "similar to"/"comparable to"/... marker, else None.

    Needed for the "recommend" intent's analog mode when NEITHER named period has a day number
    attached, so `resolve_day` can't tell them apart by day-adjacency -- "a day to go out in
    January 2010 similar to June 2010": `resolve_period` on the whole text would pick January by
    dict order (jan sorts before jun), even though the user's own "similar to June" phrasing
    makes June unambiguously the target, not January. Deliberately narrower than the "like"
    keyword also used to detect the `recommend` intent -- "like" alone is too common a word
    ("a day I'd **like** to go out") to safely mark a comparison clause.
    """
    q = question.lower()
    matches = [(q.find(m), len(m)) for m in _TARGET_CLAUSE_MARKERS if m in q]
    if not matches:
        return None
    start, mlen = min(matches)
    return question[start + mlen:]


_SCOPE_RESTRICT_PHRASES = ("within", "during", "throughout", "stay within", "staying within",
                           "restrict", "restricted", "only in", "only during", "same month",
                           "that month", "same year", "that year", "no further than")


def resolve_explicit_scope_restriction(question: str) -> bool:
    """True if the question explicitly names a bounding period as a restriction, distinct from
    just stating a specific day's own date -- for the "recommend" intent's analog mode.

    "recommend days similar to 6 April 2013 **in the month of** April 2013" names "April" TWICE:
    once as part of the date, once as an explicit "search only within this month" scope -- so
    analog candidates should be restricted to April 2013. A bare "similar to 13 February 2013"
    names February once and means search the *whole record* for analogs, not just February.
    Detected via either an explicit restriction phrase ("within", "same month", ...) or a month
    name repeated in the question (the same tell as the worked example above).
    """
    q = question.lower()
    if any(p in q for p in _SCOPE_RESTRICT_PHRASES):
        return True
    return any(len(re.findall(rf"\b{name}\b", q)) >= 2 for name in MONTHS)


def resolve_secondary_period(question: str, target_month: int, target_year: int):
    """A period named in `question` OTHER than the target day's own month/year, else None.

    For the "recommend" intent's analog mode: "recommend a day like 5 September 2012 in
    November 2012" names a SECOND, different period (November 2012) as an explicit search
    restriction -- and it doesn't have to share the target's month OR year ("...in June 2016"
    works the same way). `resolve_period` itself can't see this: given two different month
    names it always returns whichever comes first by dict order (jan..dec), not by where it
    actually appears in the text, so it silently drops "November" here.

    Works by stripping ONE occurrence of the target's own month name and year, then running
    `resolve_period` on what's left -- whatever period (if any) is still named there is the
    restriction, found by value (not text position), so it works regardless of word order.
    """
    q = question.lower()
    stripped = q
    for name, num in MONTHS.items():
        if int(num) == target_month and re.search(rf"\b{name}\b", stripped):
            stripped = re.sub(rf"\b{name}\b", " ", stripped, count=1)
            break
    stripped = re.sub(rf"\b{target_year}\b", " ", stripped, count=1)
    start, end, label = resolve_period(stripped)
    return (start, end, label) if start or end else None


def resolve_generic_direction(question: str, columns) -> tuple[str, bool] | None:
    """A named sensor + a generic high/low word -> (sensor, ascending), else None.

    Fallback for `resolve_direction` when the sensor isn't one of the six adjective-covered
    ones -- e.g. "recommend the highest pressure days" (no natural adjective for "pressure").
    """
    sensors = resolve_sensors(question, columns)
    if not sensors:
        return None
    q = question.lower()
    for word, ascending in GENERIC_DIRECTION_WORDS.items():
        if re.search(rf"\b{word}\b", q):
            return sensors[0], ascending
    return None


def resolve_recurring_month(question: str) -> int | None:
    """A bare month name with **no year** -> its number (1-12), else None.

    Used by the "recommend" intent's ranking mode: "hot days in June" (no year) means every
    June across the whole record, not a single month+year slice -- `resolve_period` already
    handles the year-given case ("June 2015"), so this only fires when it wouldn't.
    """
    q = question.lower()
    if re.search(r"\b(?:19|20)\d{2}\b", q):
        return None
    for name, num in MONTHS.items():
        if re.search(rf"\b{name}\b", q):
            return int(num)
    return None


# Fixed explanations of the chart markers — facts the system knows, so answering a
# "what is the red dot?" question needs no computation and cannot be hallucinated. The
# LLM (when on) only rephrases these; the meanings themselves are code.
LEGEND_EXPLANATIONS = {
    "anomaly": "A solid red dot marks an anomaly — a reading flagged as unusual, where its "
               "residual z-score passed the detection threshold.",
    "borderline": "A dashed amber ring marks a borderline point — close to the anomaly threshold "
                  "but not flagged. It is shown to convey uncertainty: a near-miss, not a confirmed anomaly.",
    "max": "A solid orange dot marks the maximum — the point in view where the highest value occurred.",
    "min": "A solid blue dot marks the minimum — the point in view where the lowest value occurred.",
    "forecast": "The shaded band is the 80% forecast range, and the line inside it is the forecast mean.",
    "all": "Chart legend: the blue line is the observed data; a solid red dot is an anomaly (flagged "
           "unusual); a dashed amber ring is a borderline near-miss (close to the threshold, not flagged); "
           "a solid orange dot is the maximum and a solid blue dot the minimum; a shaded band is the 80% "
           "forecast range.",
}

_LEGEND_MARKER_WORDS = ("dot", "circle", "ring", "marker", "legend", "symbol", "colour", "color",
                        "band", "shaded", "shade")
_LEGEND_MEANING_WORDS = ("what", "mean", "meaning", "explain", "represent", "stand for", "which", "why")
# Match marker cues on WORD BOUNDARIES, not as raw substrings: "ring" must not fire inside
# "du(ring)", or an ordinary data question like "maximum temperature during July 2015" is
# mis-routed to the chart-legend explainer.
_LEGEND_MARKER_RE = re.compile(r"\b(?:" + "|".join(_LEGEND_MARKER_WORDS) + r")\b")
_LEGEND_BORDERLINE_RE = re.compile(r"\b(?:dashed|dotted|amber|hollow|ring|borderline)\b")


def resolve_legend_query(question: str) -> str | None:
    """Detect a question about the chart legend/markers -> a marker key, else None.

    Fires only when the text both names a marker cue (dot/circle/ring/legend/band/colour…) *and*
    asks for its meaning (what/mean/explain…), so ordinary data questions ("what was the max?")
    are not captured. Returns one of the ``LEGEND_EXPLANATIONS`` keys.
    """
    q = question.lower()
    is_marker = bool(_LEGEND_MARKER_RE.search(q))
    is_meaning = ("legend" in q) or any(w in q for w in _LEGEND_MEANING_WORDS)
    if not (is_marker and is_meaning):
        return None
    if _LEGEND_BORDERLINE_RE.search(q):
        return "borderline"
    if "red" in q:
        return "anomaly"
    if "orange" in q:
        return "max"
    if "blue" in q:
        return "min"
    if any(w in q for w in ("band", "shaded", "shade", "forecast range", "pink")):
        return "forecast"
    return "all"


_CAUSAL_WORDS = ("cause", "causes", "caused", "causing", "causation", "leads to", "lead to",
                 "responsible for", "due to", "because of", "what makes", "why does", "why is",
                 "why are", "why did")


def is_causal_query(question: str) -> bool:
    """True if the question asks about *cause* ("does X cause Y?", "what causes…?", "why does…?").

    The system measures correlation, not causation, so these are answered with an explicit
    correlation-not-causation caveat rather than silently treated as a relationship query.

    Uses \\b word boundaries (not plain substring search) -- "cause" is a literal substring of
    "because", so a bare `w in question` check flagged ANY question containing "because" (even
    with no "of", e.g. "I stayed in because it was cold, what was the average temperature?") as
    causal, hijacking it into the correlation branch and answering a question the user never
    asked instead of the one they did.
    """
    q = question.lower()
    return any(re.search(rf"\b{re.escape(w)}\b", q) for w in _CAUSAL_WORDS)


_CORRELATION_BREAKDOWN_PHRASES = (
    "why is it only", "why is that only", "why is it so", "why is that so",
    "why isn't it stronger", "why is it not stronger", "why not stronger", "why so weak",
    "why so strong", "break down", "breakdown", "explain the relationship",
    "explain that correlation", "explain this correlation", "why is the correlation",
    "why is that correlation",
)


def is_correlation_breakdown_query(question: str) -> bool:
    """True for "why is that correlation only moderate?" style follow-ups.

    Distinct from `is_causal_query`: that flags "why does X cause Y" (a claim about causation,
    answered with a caveat), this flags "why is the NUMBER the strength it is" (a request to
    explain the correlation's own magnitude -- R², rank correlation, seasonal breakdown). The
    two aren't mutually exclusive ("why is it only moderate" can trip both), and that's fine --
    each answers a genuinely different question the phrasing raised.
    """
    q = question.lower()
    return any(p in q for p in _CORRELATION_BREAKDOWN_PHRASES)


# Qualitative strength bins for |r| (Pearson or the 0-1 unsigned circular-linear measure) --
# the common convention (e.g. Evans 1996): checked high-to-low, first threshold met wins.
_CORRELATION_STRENGTH_BINS = (
    (0.80, "very strong"), (0.60, "strong"), (0.40, "moderate"), (0.20, "weak"),
)


def correlation_strength(r: float) -> str:
    """A plain-English strength label for a correlation coefficient's magnitude.

    Turns a bare number ("-0.572") into something a non-technical reader can act on ("a
    moderate negative correlation") without doing the interpretation themselves -- the
    "explainable" half of a trust-chain answer, not just the raw statistic. Only labels
    magnitude; sign (positive/negative) is the caller's job, since this also serves the
    0-1 unsigned circular-linear measure, which has no sign at all.
    """
    a = abs(r)
    for threshold, label in _CORRELATION_STRENGTH_BINS:
        if a >= threshold:
            return label
    return "very weak"


def detect_intent(question: str) -> str | None:
    """Classify a question into one intent via whole-word keyword matching.

    Word boundaries (``\\b``) stop short keywords from matching inside unrelated
    words — e.g. "most" must not fire inside "almost", nor "top" inside "stop".
    """
    q = question.lower()
    for intent, keywords in INTENTS.items():
        if any(re.search(rf"\b{re.escape(kw.strip())}\b", q) for kw in keywords):
            return intent
    return None


# Month names -> two-digit number, for "<month> <year>" questions.
MONTHS = {
    "jan": "01", "january": "01", "feb": "02", "february": "02", "mar": "03", "march": "03",
    "apr": "04", "april": "04", "may": "05", "jun": "06", "june": "06", "jul": "07", "july": "07",
    "aug": "08", "august": "08", "sep": "09", "sept": "09", "september": "09", "oct": "10",
    "october": "10", "nov": "11", "november": "11", "dec": "12", "december": "12",
}

# Comparison words -> operator, for threshold questions ("above 30", "colder than -5").
COMPARE = {
    ">": ["above", "exceed", "exceeds", "exceeded", "hotter than", "warmer than",
          "more than", "greater than", "higher than"],
    "<": ["below", "colder than", "cooler than", "less than", "lower than"],
}


SEASONS = {  # meteorological seasons -> (start_month, end_month)
    "spring": (3, 5), "summer": (6, 8), "autumn": (9, 11), "fall": (9, 11), "winter": (12, 2),
}


def _month_end(year: int, month: int) -> str:
    import calendar
    return f"{year:04d}-{month:02d}-{calendar.monthrange(year, month)[1]:02d}"


def resolve_period(question: str):
    """Detect a time window in free text -> (start_date, end_date, label).

    Understands a year ("2013"), a month+year ("july 2015"), a **season**+year
    ("summer 2012" -> Jun–Aug; "winter 2013" -> Dec 2012–Feb 2013), and a year
    range ("between 2012 and 2014"). Dates are ``YYYY-MM-DD`` strings for
    ``df.loc[start:end]``. Returns ``(None, None, "")`` when no period is named
    (=> whole record). This is the offline fallback; the LLM planner handles
    looser phrasing.
    """
    q = question.lower()
    # Dedupe repeated mentions of the SAME year (e.g. "6 April 2013 in the month of April
    # 2013" says "2013" twice but means one period, not a range) -- only genuinely different
    # year numbers count as a range.
    years = list(dict.fromkeys(re.findall(r"\b((?:19|20)\d{2})\b", q)))

    if len(years) >= 2:                                   # year range
        y0, y1 = sorted(int(y) for y in years[:2])
        return f"{y0}-01-01", _month_end(y1, 12), f"{y0}–{y1}"

    if len(years) == 1:
        y = int(years[0])
        for name, (m0, m1) in SEASONS.items():            # season + year
            if re.search(rf"\b{name}\b", q):
                if name == "winter":                      # Dec (y-1) .. Feb (y)
                    return f"{y - 1}-12-01", _month_end(y, 2), f"winter {y}"
                return f"{y}-{m0:02d}-01", _month_end(y, m1), f"{name} {y}"
        for name, num in MONTHS.items():                  # month + year
            if re.search(rf"\b{name}\b", q):
                return f"{y}-{num}-01", _month_end(y, int(num)), f"{name.capitalize()} {y}"
        return f"{y}-01-01", _month_end(y, 12), str(y)    # whole year

    return None, None, ""


def _coerce_date(value):
    """Validate an LLM-supplied date string -> 'YYYY-MM-DD' or None."""
    if not value:
        return None
    try:
        return str(pd.to_datetime(value).date())
    except (ValueError, TypeError):
        return None


def _day_label(month: int, day: int, year: int | None) -> str:
    import calendar
    name = calendar.month_name[month]
    return f"{day} {name} {year}" if year else f"{day} {name}"


def resolve_day(question: str):
    """Detect a specific calendar day -> (month, day, year_or_None, label), else None.

    Handles "26 january", "january 26", "26th of jan 2015", "26/01/2015", and
    "2015-01-26". The year is optional — without it the day is read across all years
    ("the weather on 26 January" -> every 26 Jan in the record).
    """
    q = question.lower()
    ymatch = re.search(r"\b((?:19|20)\d{2})\b", q)
    year = int(ymatch.group(1)) if ymatch else None

    iso = re.search(r"\b((?:19|20)\d{2})-(\d{1,2})-(\d{1,2})\b", q)
    if iso:
        y, mo, d = (int(g) for g in iso.groups())
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return mo, d, y, _day_label(mo, d, y)

    num = re.search(r"\b(\d{1,2})/(\d{1,2})(?:/((?:19|20)\d{2}))?\b", q)
    if num:
        d, mo = int(num.group(1)), int(num.group(2))
        y = int(num.group(3)) if num.group(3) else year
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return mo, d, y, _day_label(mo, d, y)

    for name, mon in MONTHS.items():
        if re.search(rf"\b{name}\b", q):
            m1 = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?{name}\b", q)
            m2 = re.search(rf"\b{name}\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", q)
            dm = m1 or m2
            if dm and 1 <= int(dm.group(1)) <= 31:
                d = int(dm.group(1))
                return int(mon), d, year, _day_label(int(mon), d, year)
            # This month was named with no adjacent day ("...in June 2016") -- keep looking
            # rather than giving up, since a SECOND month elsewhere in the text (dict order is
            # calendar order, not text order) may be the one that actually has a day attached,
            # e.g. "5 September 2012 in June 2016": "june" sorts before "september" in MONTHS
            # but has no day next to it, while "september" does.
            continue
    return None


_ORDINAL_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12,
}

# Guards resolve_relative_period against a chained ordinal+relative construct it can't resolve
# ("third week of last year", "second month of this year"): without this, e.g. "last year" inside
# "third week of last year" would match on its own and silently answer with the wrong (too-broad)
# scope, dropping "third week" with no signal anything was ignored -- worse than not answering at
# all. `resolve_ordinal_unit_in_year` only handles a literal year, not a nested relative unit, so
# this combination is genuinely unsupported; refusing it here lets it fall through cleanly instead.
_CHAINED_ORDINAL_PREFIX = re.compile(
    r"\b(?:last|" + "|".join(_ORDINAL_WORDS) + r"|\d{1,3}(?:st|nd|rd|th))"
    r"\s+(?:month|week|day|hour)\s+(?:of|in)\b"
)


def resolve_relative_period(question: str, anchor_ts):
    """Detect a bare relative period ("last month", "this hour", "now") -> (start, end, label).

    Anchored to `anchor_ts` (the dataset's own last timestamp, e.g. `df.index.max()`) rather than
    the real clock -- a static historical record has no real "now". Only fires when **no explicit
    year** is named in the question; a year present changes the meaning entirely (see
    `resolve_ordinal_unit_in_year`, which resolves inside that year instead). Also refuses when the
    question is a chained ordinal+relative construct ("third week of last year") that neither this
    function nor `resolve_ordinal_unit_in_year` fully resolves -- see `_CHAINED_ORDINAL_PREFIX`.
    Returns ``(None, None, "")`` when nothing (cleanly) relative is named.
    """
    q = question.lower()
    if re.search(r"\b((?:19|20)\d{2})\b", q):
        return None, None, ""
    if _CHAINED_ORDINAL_PREFIX.search(q):
        return None, None, ""
    anchor_ts = pd.Timestamp(anchor_ts)

    if re.search(r"\bnow\b", q):
        return str(anchor_ts), str(anchor_ts), "now"

    if re.search(r"\bthis hour\b", q):
        return str(anchor_ts.floor("h")), str(anchor_ts), "this hour"
    if re.search(r"\blast hour\b", q):
        end = anchor_ts.floor("h") - pd.Timedelta(microseconds=1)
        start = anchor_ts.floor("h") - pd.Timedelta(hours=1)
        return str(start), str(end), "last hour"

    if re.search(r"\bthis week\b", q):
        return str(anchor_ts - pd.Timedelta(days=7)), str(anchor_ts), "this week"
    if re.search(r"\blast week\b", q):
        end = anchor_ts - pd.Timedelta(days=7) - pd.Timedelta(microseconds=1)
        start = anchor_ts - pd.Timedelta(days=14)
        return str(start), str(end), "last week"

    if re.search(r"\bthis month\b", q):
        start = anchor_ts.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return str(start), str(anchor_ts), f"{anchor_ts.strftime('%B %Y')} (to date)"
    if re.search(r"\blast month\b", q):
        prev = anchor_ts.replace(day=1) - pd.Timedelta(days=1)
        return f"{prev.year}-{prev.month:02d}-01", _month_end(prev.year, prev.month), \
            prev.strftime("%B %Y")

    if re.search(r"\bthis year\b", q):
        start = anchor_ts.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        return str(start), str(anchor_ts), f"{anchor_ts.year} (to date)"
    if re.search(r"\blast year\b", q):
        y = anchor_ts.year - 1
        return f"{y}-01-01", _month_end(y, 12), str(y)

    return None, None, ""


def _qualifier_to_n(qual: str) -> int | None:
    """'last' -> None (caller resolves the final index); a word/numeric ordinal -> its integer."""
    if qual == "last":
        return None
    if qual in _ORDINAL_WORDS:
        return _ORDINAL_WORDS[qual]
    return int(re.match(r"\d+", qual).group())


def _month_range_in_year(n: int | None, year: int):
    import calendar
    m = n if n is not None else 12
    if not 1 <= m <= 12:
        return None, None, ""
    return f"{year}-{m:02d}-01", _month_end(year, m), f"{calendar.month_name[m]} {year}"


def _week_range_in_year(n: int | None, year: int):
    jan1 = pd.Timestamp(year=year, month=1, day=1)
    dec31 = pd.Timestamp(year=year, month=12, day=31)
    max_week = -(-((dec31 - jan1).days + 1) // 7)  # ceil division
    w = n if n is not None else max_week
    if not 1 <= w <= max_week:
        return None, None, ""
    start = jan1 + pd.Timedelta(days=(w - 1) * 7)
    end = min(start + pd.Timedelta(days=6), dec31)
    return str(start.date()), str(end.date()), f"week {w} of {year}"


def _day_range_in_year(n: int | None, year: int):
    jan1 = pd.Timestamp(year=year, month=1, day=1)
    dec31 = pd.Timestamp(year=year, month=12, day=31)
    n_days = (dec31 - jan1).days + 1
    d = n if n is not None else n_days
    if not 1 <= d <= n_days:
        return None, None, ""
    day = jan1 + pd.Timedelta(days=d - 1)
    return str(day.date()), str(day.date()), f"day {d} of {year} ({day.strftime('%-d %B')})"


def _hour_range_in_year(n: int | None, year: int):
    jan1 = pd.Timestamp(year=year, month=1, day=1)
    n_hours = int((pd.Timestamp(year=year, month=12, day=31, hour=23) - jan1).total_seconds() // 3600) + 1
    h = n if n is not None else n_hours
    if not 1 <= h <= n_hours:
        return None, None, ""
    start = jan1 + pd.Timedelta(hours=h - 1)
    end = start + pd.Timedelta(minutes=59, seconds=59)
    return str(start), str(end), f"hour {h} of {year} ({start.strftime('%-d %B, %H:00')})"


_ORDINAL_RANGE_FNS = {
    "month": _month_range_in_year,
    "week": _week_range_in_year,
    "day": _day_range_in_year,
    "hour": _hour_range_in_year,
}


def resolve_ordinal_unit_in_year(question: str):
    """Detect "<ordinal> <unit> of/in <year>" -> (start, end, label), else (None, None, "").

    Fires only when an explicit year is present -- a relative/ordinal qualifier combined with a
    named year resolves *inside that year* ("last month 2013" -> December 2013, "second month 2016"
    -> February 2016), never against the dataset's own timeline (see `resolve_relative_period` for
    the no-year case). Checked before `resolve_period` by the caller, since `resolve_period` would
    otherwise swallow the bare year into a whole-year default and miss the qualifier entirely.
    `week`/`day`/`hour` count consecutive fixed-size blocks from 1 January of that year -- not ISO
    calendar weeks, which have year-boundary edge cases.
    """
    q = question.lower()
    ymatch = re.search(r"\b((?:19|20)\d{2})\b", q)
    if not ymatch:
        return None, None, ""
    year = int(ymatch.group(1))
    qualifiers = "|".join(["last", *_ORDINAL_WORDS])
    for unit, fn in _ORDINAL_RANGE_FNS.items():
        m = re.search(rf"\b({qualifiers}|\d{{1,3}}(?:st|nd|rd|th))\s+{unit}\b", q)
        if m:
            n = _qualifier_to_n(m.group(1))
            return fn(n, year)
    return None, None, ""


def resolve_any_period(question: str, anchor_ts=None):
    """Best-available period across ordinal, explicit, and relative forms, in that priority order.

    Tries `resolve_ordinal_unit_in_year` first (a relative/ordinal qualifier + explicit year), since
    `resolve_period` would otherwise swallow a bare year into a whole-year default and miss the
    qualifier. Falls back to `resolve_period` (explicit year/month/season/range), then -- only if
    `anchor_ts` is given -- `resolve_relative_period` (bare relative phrase, dataset-anchored).
    """
    start, end, label = resolve_ordinal_unit_in_year(question)
    if start or end:
        return start, end, label
    start, end, label = resolve_period(question)
    if start or end:
        return start, end, label
    if anchor_ts is not None:
        return resolve_relative_period(question, anchor_ts)
    return None, None, ""


_HOUR_WORDS = {"noon": 12, "midday": 12, "midnight": 0}
_HOUR_BUCKETS = {"morning": (6, 12), "afternoon": (12, 18), "evening": (18, 21), "night": (21, 6)}


def resolve_hour(question: str):
    """Detect an hour-of-day sub-filter -> (start_hour, end_hour, label), else (None, None, "").

    A half-open [start_hour, end_hour) range of 24h-clock hours; `end_hour < start_hour` marks a
    bucket that wraps past midnight ("night" = 21-06). Independent of whatever day/period/relative
    scope is also named in the question -- this only narrows by clock hour; the caller applies it as
    an extra mask via `hour_mask`.
    """
    q = question.lower()

    for word, h in _HOUR_WORDS.items():
        if re.search(rf"\b{word}\b", q):
            return h, h + 1, word

    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", q)
    if m:
        h12 = int(m.group(1))
        if 1 <= h12 <= 12:                # a 12-hour clock only has 1-12; "13pm" etc is malformed
            h = h12 % 12
            if m.group(3) == "pm":
                h += 12
            return h, h + 1, m.group(0)

    m = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", q)
    if m:
        h = int(m.group(1))
        return h, h + 1, m.group(0)

    m = re.search(r"\b(\d{1,2})\s*o'?clock\b", q)
    if m:
        h = int(m.group(1))
        if 0 <= h <= 23:
            return h, h + 1, m.group(0)

    for word, rng in _HOUR_BUCKETS.items():
        if re.search(rf"\b{word}\b", q):
            return rng[0], rng[1], word

    return None, None, ""


def hour_mask(idx: pd.DatetimeIndex, start_hour: int, end_hour: int):
    """Boolean mask selecting rows whose clock hour falls in [start_hour, end_hour) -- wraps past
    midnight when `end_hour < start_hour` (e.g. "night" = 21-06)."""
    if start_hour < end_hour:
        return (idx.hour >= start_hour) & (idx.hour < end_hour)
    return (idx.hour >= start_hour) | (idx.hour < end_hour)


_COMPARE_WORDS = ["compare", "comparison", "versus", "vs", "difference", "hotter", "warmer",
                  "colder", "cooler", "higher", "lower", "than"]


def resolve_compare(question: str):
    """Detect a two-period comparison -> (yearA, yearB, stat), else None.

    Fires when the question names two years *and* a comparison word
    ("compare 2013 and 2014", "was 2013 hotter than 2014?"). `stat` is max for
    hot/high wording, min for cold/low, otherwise mean.
    """
    q = question.lower()
    years = sorted({int(y) for y in re.findall(r"\b((?:19|20)\d{2})\b", q)})
    if len(years) < 2:
        return None
    if not any(re.search(rf"\b{re.escape(w)}\b", q) for w in _COMPARE_WORDS):
        return None
    if re.search(r"\b(hot|warm|high|max)", q):        # hot/hotter/hottest, warm/warmer, high/higher…
        stat = "max"
    elif re.search(r"\b(cold|cool|low|min)", q):      # cold/colder, cool/cooler, low/lower…
        stat = "min"
    else:
        stat = "mean"
    return years[0], years[1], stat


def resolve_threshold(question: str):
    """Detect a comparison + numeric value -> (op, value), else None.

    The threshold is taken as the number that *follows* the comparison word, so
    "above 30 in 2013" reads 30 (not the year 2013). This position-based rule also
    fixes the old year-range edge case: "above 2000" now correctly reads 2000 as a
    genuine threshold (a year-range number is accepted only when it sits directly
    after the comparison word), while "above average in 2015" yields no threshold.
    """
    q = question.lower()
    # Earliest comparison word wins, and we remember where it ends (op_end).
    best: tuple[int, int, str] | None = None
    for o, words in COMPARE.items():
        for w in words:
            m = re.search(rf"\b{re.escape(w)}\b", q)
            if m and (best is None or m.start() < best[0]):
                best = (m.start(), m.end(), o)
    if best is None:
        return None
    _, op_end, op = best
    after = [(m.start(), float(m.group()))
             for m in re.finditer(r"-?\d+(?:\.\d+)?", q) if m.start() >= op_end]
    # Prefer the first non-year-range number after the comparison word (a clear threshold
    # like 30 or -5), so a trailing scope year ("above 30 in 2013") is ignored.
    non_year = [v for _, v in after if not (v.is_integer() and 1900 <= v <= 2100)]
    if non_year:
        return (op, non_year[0])
    # Only a year-range number follows: accept it as a genuine threshold ("above 2000")
    # solely when it sits directly after the comparison word (nothing but spaces between),
    # which separates it from a scope year ("above average in 2015" -> no threshold).
    for pos, v in after:
        if q[op_end:pos].strip() == "":
            return (op, v)
    return None


def _canonical_scope(question: str, columns):
    """Describe the time-scope a question names as (kind, text), or None.

    ``kind`` is ``"day"`` or ``"period"`` and ``text`` re-parses back to the same scope. A
    specific day is preferred over a period, because a day label ("26 January 2015") also
    contains a month/year that ``resolve_period`` would otherwise read as a whole month.
    """
    day = resolve_day(question)
    if day is not None:
        return ("day", day[3])                       # the human day label, e.g. "26 January 2015"
    start, end, label = resolve_period(question)
    if start or end:
        return ("period", label)
    return None


_FOLLOWUP_PREFIXES = ("and ", "and,", "also ", "then ", "what about", "how about", "what of")
_FOLLOWUP_CONTAINS = ("what about", "how about", "what of")


def is_followup_cue(question: str) -> bool:
    """True if the wording signals a continuation of the previous turn ("and…", "what about…").

    This cue is what lets an otherwise self-contained-looking follow-up ("and the coldest?", which
    names an intent *and*, via "cold", a sensor) still inherit the previous turn's scope instead of
    silently re-reading the whole record.
    """
    q = question.strip().lower()
    return q.startswith(_FOLLOWUP_PREFIXES) or any(c in q for c in _FOLLOWUP_CONTAINS)


def resolve_followup(question: str, history, columns) -> str | None:
    """Resolve an elliptical follow-up into a standalone question deterministically, else None.

    When the new message is elliptical — it lacks an intent and/or a scope of its own but
    introduces at least one *new* piece (a year/period/day, or a different sensor) — the missing
    pieces are carried forward from the most recent turn and a canonical standalone question is
    composed (e.g. history "hottest in 2013" + "what about 2014?" -> "max T (degC) in 2014").

    This gives the interface conversational memory **without an LLM**, so follow-ups work in the
    default offline mode (the previous behaviour only remembered context with Ollama running). It
    is fully deterministic and only rewrites the wording — every number is still computed by the
    query engine. Returns ``None`` when there is nothing new to carry context onto, or when the
    pieces are insufficient to route a standalone question (left to the LLM/fallback).
    """
    if not history:
        return None
    prev = history[-1].get("resolved") or history[-1].get("question")
    if not prev:
        return None

    cur_intent = detect_intent(question)
    cur_sensors = resolve_sensors(question, columns)
    cur_scope = _canonical_scope(question, columns)
    if not (cur_intent or cur_sensors or cur_scope):
        return None                                  # nothing new -> not a resolvable follow-up

    intent = cur_intent or detect_intent(prev)
    prev_sensors = resolve_sensors(prev, columns)
    sensor = cur_sensors[0] if cur_sensors else (prev_sensors[0] if prev_sensors else None)
    scope = cur_scope or _canonical_scope(prev, columns)
    if intent is None or (sensor is None and scope is None):
        return None                                  # not enough to route a standalone question

    parts = [INTENTS[intent][0]]                      # a canonical keyword the parser routes back
    if sensor is not None:
        parts.append(sensor)
    if scope is not None:
        kind, text = scope
        parts.append(text if kind == "day" else f"in {text}")
    return " ".join(parts)
