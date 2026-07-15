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
    # humidity / water vapour
    "relative humidity": "rh (%)", "humidity": "rh (%)", "humid": "rh (%)",
    "specific humidity": "sh (g/kg)",
    "water vapour concentration": "H2OC (mmol/mol)", "water vapor concentration": "H2OC (mmol/mol)",
    "water vapour": "H2OC (mmol/mol)", "water vapor": "H2OC (mmol/mol)",
    # vapour pressures (specific phrases before the generic "vapour pressure")
    "saturation vapour pressure": "VPmax (mbar)", "saturation vapor pressure": "VPmax (mbar)",
    "vapour pressure deficit": "VPdef (mbar)", "vapor pressure deficit": "VPdef (mbar)",
    "actual vapour pressure": "VPact (mbar)", "actual vapor pressure": "VPact (mbar)",
    "vapour pressure": "VPact (mbar)", "vapor pressure": "VPact (mbar)",
    # pressure / density
    "pressure": "p (mbar)",
    "air density": "rho (g/m**3)", "density": "rho (g/m**3)",
    # wind (direction/gust phrases before the generic "wind")
    "wind direction": "wd (deg)", "direction": "wd (deg)",
    "wind gust": "max. wv (m/s)", "gust": "max. wv (m/s)",
    "wind speed": "wv (m/s)", "wind": "wv (m/s)",
}

INTENTS = {
    "max": ["max", "maximum", "highest", "hottest", "peak", "warmest", "top", "biggest", "most",
            "hot", "record high"],
    "min": ["min", "minimum", "lowest", "coldest", "smallest", "least", "cold", "record low"],
    "mean": ["mean", "average", "typical", "normally", "usual"],
    "spread": ["std", "deviation", "variability", "spread", "variance", "vary", "fluctuate"],
    "correlation": ["correlation", "correlate", "related", "relationship", "relate", "versus",
                    " vs ", "depend", "affect", "influence", "connection", "connected", "link",
                    "linked", "associated", "associate"],
    "missing": ["missing", "gap", "gaps", "absent", "incomplete"],
    "forecast": ["forecast", "predict", "prediction", "projection", "next week", "next month",
                 "next few days", "coming days", "future", "will it", "going to be", "outlook"],
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
        start = q.find(word)
        while start != -1:
            if not any(consumed[start:start + len(word)]):
                for i in range(start, start + len(word)):
                    consumed[i] = True
                hits.append((start, col))
                break
            start = q.find(word, start + 1)
    found: list[str] = []
    for _, col in sorted(hits):  # order by appearance in the question
        if col not in found:
            found.append(col)
    for col in columns:  # also catch exact column names mentioned (e.g. "rh (%)")
        if col.lower() in q and col not in found:
            found.append(col)
    return found


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


def resolve_legend_query(question: str) -> str | None:
    """Detect a question about the chart legend/markers -> a marker key, else None.

    Fires only when the text both names a marker cue (dot/circle/ring/legend/band/colour…) *and*
    asks for its meaning (what/mean/explain…), so ordinary data questions ("what was the max?")
    are not captured. Returns one of the ``LEGEND_EXPLANATIONS`` keys.
    """
    q = question.lower()
    is_marker = any(w in q for w in _LEGEND_MARKER_WORDS)
    is_meaning = ("legend" in q) or any(w in q for w in _LEGEND_MEANING_WORDS)
    if not (is_marker and is_meaning):
        return None
    if any(w in q for w in ("dashed", "dotted", "amber", "hollow", "ring", "borderline")):
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
    """
    return any(w in question.lower() for w in _CAUSAL_WORDS)


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
    years = re.findall(r"\b((?:19|20)\d{2})\b", q)

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
            break  # a month was named but no day -> not a specific-day question
    return None


def resolve_relative_period(question: str, anchor_ts):
    """Detect a bare relative period ("last month", "this hour", "now") -> (start, end, label).

    Anchored to `anchor_ts` (the dataset's own last timestamp, e.g. `df.index.max()`) rather than
    the real clock -- a static historical record has no real "now". Only fires when **no explicit
    year** is named in the question; a year present changes the meaning entirely (see
    `resolve_ordinal_unit_in_year`, which resolves inside that year instead). Returns
    ``(None, None, "")`` when nothing relative is named.
    """
    q = question.lower()
    if re.search(r"\b((?:19|20)\d{2})\b", q):
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


_ORDINAL_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12,
}


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
