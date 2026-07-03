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
    "max": ["max", "maximum", "highest", "hottest", "peak", "warmest", "top", "biggest", "most"],
    "min": ["min", "minimum", "lowest", "coldest", "smallest", "least"],
    "mean": ["mean", "average", "typical", "normally"],
    "spread": ["std", "deviation", "variability", "spread", "variance", "vary", "fluctuate"],
    "correlation": ["correlation", "correlate", "related", "relationship", "relate", "versus",
                    " vs ", "depend", "affect", "influence", "connection"],
    "missing": ["missing", "gap", "gaps", "absent", "incomplete"],
    "forecast": ["forecast", "predict", "prediction", "projection", "next week", "next month",
                 "next few days", "coming days", "future", "will it", "going to be"],
    "trend": ["trend", "trending", "rising", "falling", "increasing", "decreasing", "increase",
              "decrease", "warming", "cooling", "going up", "going down", "over time"],
    "compare": ["compare", "comparison", "versus", "vs", "difference between"],
    "anomaly": ["anomaly", "anomalies", "unusual", "abnormal", "weird", "strange", "outlier",
                "odd", "spike", "wrong", "problem"],
    "weather": ["weather", "conditions", "climate"],
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

    Four-digit numbers in the year range are treated as periods, not thresholds.
    """
    q = question.lower()
    op = next((o for o, words in COMPARE.items()
               if any(re.search(rf"\b{re.escape(w)}\b", q) for w in words)), None)
    if op is None:
        return None
    nums = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", q)]
    nums = [n for n in nums if not (n.is_integer() and 1900 <= n <= 2100)]
    return (op, nums[0]) if nums else None
