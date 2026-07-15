# Time-Constraint Parsing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Teach v1's query interface to understand hour-of-day ("3pm", "morning") and relative/ordinal time phrases ("last month", "last hour", "second month 2016", "last week of 2013") — currently these are either silently dropped or resolve to nothing, per `v1/docs/superpowers/specs/2026-07-15-time-constraint-parsing-design.md`.

**Architecture:** Three new pure, deterministic parsing functions in `src/interface_parsing.py` (`resolve_relative_period`, `resolve_ordinal_unit_in_year`, `resolve_hour`) plus a combinator (`resolve_any_period`) and a mask helper (`hour_mask`), wired into `src/interface.py`'s existing scope-resolution chain. No ML/LLM involved — same regex/keyword style as the existing `resolve_period`/`resolve_day`.

**Tech Stack:** Python 3.12, pandas, pytest. No new dependencies.

## Global Constraints

- Deterministic parsing only — no ML/LLM in the truth path (the LLM, if enabled, only *phrases* already-computed facts, never decides scope). No new third-party dependencies.
- Run tests with `python3.12 -m pytest tests/ -q` from `v1/`. Baseline before this work: **197 passed, 1 skipped** — every task must keep this passing and net-additive.
- Every new function needs a docstring stating what the returned value means, not just its type (matches the rest of `interface_parsing.py`).
- Week/day/hour "ordinal unit of year" phrases count fixed-size blocks from **1 January** of that year — explicitly **not** ISO calendar weeks (documented in the spec's Risks section to avoid year-boundary edge cases).
- No `Co-Authored-By` trailer in commits (existing project convention).
- `%-d` strftime directive is used deliberately (Linux dev environment, matches existing `query_engine.fmt_when`).

---

### Task 1: `resolve_relative_period` — bare relative phrases, dataset-anchored

**Files:**
- Modify: `src/interface_parsing.py` (insert after `resolve_day`, i.e. after line 291, before `_COMPARE_WORDS` at line 294)
- Create: `tests/test_interface_parsing.py`

**Interfaces:**
- Produces: `resolve_relative_period(question: str, anchor_ts) -> tuple[str | None, str | None, str]` — same `(start, end, label)` shape as `resolve_period`. Uses the module-private `_month_end(year, month)` already defined in `interface_parsing.py`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_interface_parsing.py`:

```python
"""Tests for the new time-constraint parsers: relative/ordinal periods and hour-of-day."""

import pandas as pd

from src.interface_parsing import resolve_relative_period

ANCHOR = pd.Timestamp("2017-01-01 00:10:00")


def test_relative_now():
    assert resolve_relative_period("what is it now", ANCHOR) == (str(ANCHOR), str(ANCHOR), "now")


def test_relative_this_hour():
    start, end, label = resolve_relative_period("temperature this hour", ANCHOR)
    assert start == str(ANCHOR.floor("h"))
    assert end == str(ANCHOR)
    assert label == "this hour"


def test_relative_last_hour():
    start, end, label = resolve_relative_period("temperature last hour", ANCHOR)
    assert start == str(ANCHOR.floor("h") - pd.Timedelta(hours=1))
    assert end == str(ANCHOR.floor("h") - pd.Timedelta(microseconds=1))
    assert label == "last hour"


def test_relative_this_week():
    start, end, label = resolve_relative_period("wind speed this week", ANCHOR)
    assert start == str(ANCHOR - pd.Timedelta(days=7))
    assert end == str(ANCHOR)
    assert label == "this week"


def test_relative_last_week():
    start, end, label = resolve_relative_period("wind speed last week", ANCHOR)
    assert start == str(ANCHOR - pd.Timedelta(days=14))
    assert end == str(ANCHOR - pd.Timedelta(days=7) - pd.Timedelta(microseconds=1))
    assert label == "last week"


def test_relative_last_month_is_december_2016():
    assert resolve_relative_period("temperature last month", ANCHOR)[:2] == ("2016-12-01", "2016-12-31")


def test_relative_this_month_is_partial_january():
    start, end, _label = resolve_relative_period("temperature this month", ANCHOR)
    assert start == "2017-01-01 00:00:00"
    assert end == str(ANCHOR)


def test_relative_last_year_is_2016():
    # ANCHOR is 2017-01-01 00:10 -- "last year" is the last full calendar year, 2016.
    assert resolve_relative_period("temperature last year", ANCHOR)[:2] == ("2016-01-01", "2016-12-31")


def test_relative_this_year_is_partial_2017_to_date():
    # ANCHOR's own year is 2017 (a 10-minute stub), consistent with "this month" resolving to a
    # partial January 2017 rather than December 2016.
    start, end, _label = resolve_relative_period("temperature this year", ANCHOR)
    assert start == "2017-01-01 00:00:00"
    assert end == str(ANCHOR)


def test_relative_period_yields_nothing_when_a_year_is_present():
    assert resolve_relative_period("temperature last month 2013", ANCHOR) == (None, None, "")


def test_relative_period_yields_nothing_for_unrelated_question():
    assert resolve_relative_period("what was the hottest day?", ANCHOR) == (None, None, "")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd v1 && python3.12 -m pytest tests/test_interface_parsing.py -v`
Expected: `ImportError: cannot import name 'resolve_relative_period'` (function doesn't exist yet).

- [ ] **Step 3: Implement `resolve_relative_period`**

In `src/interface_parsing.py`, insert immediately after `resolve_day` (after line 291, before the `_COMPARE_WORDS = [...]` block at line 294):

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd v1 && python3.12 -m pytest tests/test_interface_parsing.py -v`
Expected: all 11 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add v1/src/interface_parsing.py v1/tests/test_interface_parsing.py
git commit -m "v1: add resolve_relative_period (dataset-anchored last/this hour/week/month/year, now)"
```

---

### Task 2: `resolve_ordinal_unit_in_year` — ordinal/relative qualifier + explicit year

**Files:**
- Modify: `src/interface_parsing.py` (insert immediately after Task 1's `resolve_relative_period`)
- Modify: `tests/test_interface_parsing.py`

**Interfaces:**
- Consumes: `_month_end(year, month)` (existing, module-private).
- Produces: `resolve_ordinal_unit_in_year(question: str) -> tuple[str | None, str | None, str]` — same `(start, end, label)` shape.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_interface_parsing.py`:

```python
from src.interface_parsing import resolve_ordinal_unit_in_year


def test_ordinal_last_month_of_year_is_december():
    assert resolve_ordinal_unit_in_year("what was the wind speed last month 2013")[:2] == \
        ("2013-12-01", "2013-12-31")


def test_ordinal_word_month_of_year():
    assert resolve_ordinal_unit_in_year("temperature in the second month 2016")[:2] == \
        ("2016-02-01", "2016-02-29")  # 2016 is a leap year, Feb has 29 days


def test_ordinal_numeric_week_of_year():
    expected_start = pd.Timestamp("2013-01-01") + pd.Timedelta(days=33 * 7)
    start, _end, label = resolve_ordinal_unit_in_year("what happened in the 34th week of 2013")
    assert start == str(expected_start.date())
    assert label == "week 34 of 2013"


def test_ordinal_last_week_of_year_ends_on_dec_31():
    _start, end, _label = resolve_ordinal_unit_in_year("last week of 2013")
    assert end == "2013-12-31"


def test_ordinal_last_day_of_year():
    assert resolve_ordinal_unit_in_year("temperature on the last day of 2013")[:2] == \
        ("2013-12-31", "2013-12-31")


def test_ordinal_first_hour_of_year():
    start, _end, _label = resolve_ordinal_unit_in_year("what was the wind on the first hour of 2013")
    assert start == "2013-01-01 00:00:00"


def test_ordinal_yields_nothing_without_a_year():
    assert resolve_ordinal_unit_in_year("last month") == (None, None, "")


def test_ordinal_yields_nothing_for_a_bare_year():
    assert resolve_ordinal_unit_in_year("temperature in 2013") == (None, None, "")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd v1 && python3.12 -m pytest tests/test_interface_parsing.py -v -k ordinal`
Expected: `ImportError: cannot import name 'resolve_ordinal_unit_in_year'`.

- [ ] **Step 3: Implement `resolve_ordinal_unit_in_year`**

In `src/interface_parsing.py`, insert immediately after `resolve_relative_period` (added in Task 1):

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd v1 && python3.12 -m pytest tests/test_interface_parsing.py -v`
Expected: all 19 tests PASS (11 from Task 1 + 8 new).

- [ ] **Step 5: Commit**

```bash
git add v1/src/interface_parsing.py v1/tests/test_interface_parsing.py
git commit -m "v1: add resolve_ordinal_unit_in_year (last/Nth month/week/day/hour of <year>)"
```

---

### Task 3: `resolve_hour`, `hour_mask`, and the `resolve_any_period` combinator

**Files:**
- Modify: `src/interface_parsing.py` (insert after Task 2's block, before `_COMPARE_WORDS`)
- Modify: `tests/test_interface_parsing.py`

**Interfaces:**
- Consumes: `resolve_period` (existing), `resolve_ordinal_unit_in_year` (Task 2), `resolve_relative_period` (Task 1).
- Produces:
  - `resolve_hour(question: str) -> tuple[int | None, int | None, str]` — `(start_hour, end_hour, label)`, half-open `[start_hour, end_hour)`; `end_hour < start_hour` marks a midnight-wrapping bucket.
  - `hour_mask(idx: pd.DatetimeIndex, start_hour: int, end_hour: int) -> pd.Series[bool]`.
  - `resolve_any_period(question: str, anchor_ts=None) -> tuple[str | None, str | None, str]` — tries ordinal → explicit → relative, in that order.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_interface_parsing.py`:

```python
from src.interface_parsing import hour_mask, resolve_any_period, resolve_hour


def test_resolve_hour_pm_clock_time():
    assert resolve_hour("temperature at 3pm")[:2] == (15, 16)


def test_resolve_hour_am_clock_time():
    assert resolve_hour("wind speed at 9am")[:2] == (9, 10)


def test_resolve_hour_24h_clock():
    assert resolve_hour("temperature at 15:00")[:2] == (15, 16)


def test_resolve_hour_oclock():
    assert resolve_hour("temperature at 3 o'clock")[:2] == (3, 4)


def test_resolve_hour_noon_and_midnight():
    assert resolve_hour("temperature at noon")[:2] == (12, 13)
    assert resolve_hour("temperature at midnight")[:2] == (0, 1)


def test_resolve_hour_time_of_day_buckets():
    assert resolve_hour("temperature this morning")[:2] == (6, 12)
    assert resolve_hour("temperature this afternoon")[:2] == (12, 18)
    assert resolve_hour("temperature this evening")[:2] == (18, 21)
    assert resolve_hour("temperature at night")[:2] == (21, 6)


def test_resolve_hour_yields_nothing_for_unrelated_question():
    assert resolve_hour("what was the hottest day?") == (None, None, "")


def test_hour_mask_normal_range():
    idx = pd.DatetimeIndex([f"2016-01-01 {h:02d}:00" for h in range(24)])
    mask = hour_mask(idx, 15, 16)
    assert list(idx[mask].hour) == [15]


def test_hour_mask_wraps_past_midnight():
    idx = pd.DatetimeIndex([f"2016-01-01 {h:02d}:00" for h in range(24)])
    mask = hour_mask(idx, 21, 6)
    assert sorted(idx[mask].hour) == [0, 1, 2, 3, 4, 5, 21, 22, 23]


def test_resolve_any_period_prefers_ordinal_over_bare_year_default():
    start, end, _label = resolve_any_period("second month 2016", ANCHOR)
    assert (start, end) == ("2016-02-01", "2016-02-29")


def test_resolve_any_period_falls_back_to_explicit_period():
    assert resolve_any_period("hottest in 2013", ANCHOR)[:2] == ("2013-01-01", "2013-12-31")


def test_resolve_any_period_works_without_an_anchor_for_explicit_period():
    assert resolve_any_period("hottest in 2013") == ("2013-01-01", "2013-12-31", "2013")


def test_resolve_any_period_yields_nothing_for_relative_phrase_without_an_anchor():
    assert resolve_any_period("last month") == (None, None, "")


def test_resolve_any_period_falls_back_to_relative_with_an_anchor():
    assert resolve_any_period("last month", ANCHOR)[:2] == ("2016-12-01", "2016-12-31")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd v1 && python3.12 -m pytest tests/test_interface_parsing.py -v -k "hour_ or any_period"`
Expected: `ImportError: cannot import name 'resolve_hour'`.

- [ ] **Step 3: Implement `resolve_hour`, `hour_mask`, `resolve_any_period`**

In `src/interface_parsing.py`, insert immediately after Task 2's block (still before `_COMPARE_WORDS`):

```python
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
        h = int(m.group(1)) % 12
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd v1 && python3.12 -m pytest tests/test_interface_parsing.py -v`
Expected: all tests in the file PASS (19 from Tasks 1-2 + 16 new = 35).

- [ ] **Step 5: Commit**

```bash
git add v1/src/interface_parsing.py v1/tests/test_interface_parsing.py
git commit -m "v1: add resolve_hour, hour_mask, and resolve_any_period combinator"
```

---

### Task 4: Wire the new parsers into `Interface`

**Files:**
- Modify: `src/interface.py`
- Modify: `tests/test_interface.py`

**Interfaces:**
- Consumes: `resolve_any_period`, `resolve_hour`, `hour_mask` (Task 3); `resolve_ordinal_unit_in_year`, `resolve_relative_period` (re-exported for callers/tests, matching the module's existing re-export pattern).
- Produces: `Interface._understand(question)` now returns an 8-tuple `(intent, sensors, start, end, label, thr, day, hour)` (was 7-tuple) — `hour` is `(start_hour, end_hour, label)` from `resolve_hour`. `Interface._anchor` (new attribute) = `df.index.max()`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_interface.py` (after the existing fixtures, near the top):

```python
@pytest.fixture
def year_df():
    idx = pd.date_range("2016-01-01 00:00", "2017-01-01 00:00", freq="h")
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {"T (degC)": rng.normal(5, 3, len(idx)), "wv (m/s)": rng.normal(3, 1, len(idx))},
        index=idx,
    )


def test_interface_anchors_last_month_to_dataset_end(year_df):
    iface = interface.Interface(year_df)
    assert iface._anchor == year_df.index.max()
    ans = iface.ask("what was the mean temperature last month?")
    expected = year_df.loc["2016-12-01":"2016-12-31"]
    assert ans.value["mean"] == pytest.approx(expected["T (degC)"].mean())


def test_interface_ordinal_month_with_year_beats_bare_year_default(year_df):
    iface = interface.Interface(year_df)
    ans = iface.ask("what was the mean temperature in the second month 2016?")
    expected = year_df.loc["2016-02-01":"2016-02-29"]
    assert ans.value["mean"] == pytest.approx(expected["T (degC)"].mean())
    assert ans.value["mean"] != pytest.approx(year_df["T (degC)"].mean())


def test_interface_hour_submask_narrows_a_named_period(year_df):
    iface = interface.Interface(year_df)
    ans = iface.ask("what was the mean temperature last month at 3pm?")
    scoped = year_df.loc["2016-12-01":"2016-12-31"]
    expected = scoped[scoped.index.hour == 15]
    assert ans.value["mean"] == pytest.approx(expected["T (degC)"].mean())


def test_interface_last_hour_is_the_final_clock_hour_of_the_record(year_df):
    iface = interface.Interface(year_df)
    ans = iface.ask("what was the mean temperature last hour?")
    expected = year_df.loc["2016-12-31 23:00":"2016-12-31 23:59:59.999999"]
    assert ans.value["mean"] == pytest.approx(expected["T (degC)"].mean())


def test_interface_now_is_the_single_latest_reading(year_df):
    iface = interface.Interface(year_df)
    ans = iface.ask("what was the mean temperature now?")
    assert ans.value["mean"] == pytest.approx(year_df["T (degC)"].iloc[-1])


def test_interface_night_bucket_wraps_past_midnight(year_df):
    iface = interface.Interface(year_df)
    ans = iface.ask("what was the mean wind speed at night in december 2016?")
    scoped = year_df.loc["2016-12-01":"2016-12-31"]
    expected = scoped[(scoped.index.hour >= 21) | (scoped.index.hour < 6)]
    assert ans.value["mean"] == pytest.approx(expected["wv (m/s)"].mean())


def test_interface_hour_only_question_scopes_across_whole_record(year_df):
    iface = interface.Interface(year_df)
    ans = iface.ask("what was the mean temperature at 3pm?")
    expected = year_df[year_df.index.hour == 15]
    assert ans.value["mean"] == pytest.approx(expected["T (degC)"].mean())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd v1 && python3.12 -m pytest tests/test_interface.py -v -k "anchor or ordinal_month or hour_submask or last_hour_is or now_is or night_bucket or hour_only"`
Expected: FAIL — `AttributeError: 'Interface' object has no attribute '_anchor'` (or similar, since none of the wiring exists yet).

- [ ] **Step 3: Update imports in `src/interface.py`**

In `src/interface.py`, replace the import block (lines 27-42):

```python
from .interface_parsing import (  # noqa: F401  re-exported for callers and tests
    INTENTS,
    LEGEND_EXPLANATIONS,
    SENSOR_ALIASES,
    _coerce_date,
    detect_intent,
    hour_mask,
    is_causal_query,
    is_followup_cue,
    resolve_any_period,
    resolve_compare,
    resolve_day,
    resolve_followup,
    resolve_hour,
    resolve_legend_query,
    resolve_ordinal_unit_in_year,
    resolve_period,
    resolve_relative_period,
    resolve_sensors,
    resolve_threshold,
)
```

- [ ] **Step 4: Set the dataset anchor in `Interface.__init__`**

In `src/interface.py`, in `Interface.__init__` (lines 80-88), add `self._anchor` right after `self.df = df`:

```python
    def __init__(self, df: pd.DataFrame, use_llm: bool = False, model: str = llm.DEFAULT_MODEL,
                 detector: str = DEFAULT_DETECTOR):
        self.df = df
        self._anchor = df.index.max()  # the dataset's own "now" -- a static record has no real one
        self.retriever = rag_system.Retriever()
```

- [ ] **Step 5: Use `resolve_any_period` in `_routable` and `_needs_context`**

In `src/interface.py`, replace `_routable` (lines 98-108):

```python
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
```

Replace `_needs_context` (lines 110-121):

```python
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
```

- [ ] **Step 6: Extend `_understand` to resolve hour and use `resolve_any_period`**

In `src/interface.py`, replace `_understand` (lines 151-184):

```python
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
                        start, end = s2 or e2, e2 or s2
                        label = plan.get("period") or f"{start} to {end}"
                if thr is None and plan.get("op") in (">", "<") and plan.get("value") is not None:
                    thr = (plan["op"], float(plan["value"]))
        return intent, sensors, start, end, label, thr, day, hour
```

- [ ] **Step 7: Update the `_understand` call site and apply the hour mask in `_answer`**

In `src/interface.py`, `_answer` (starting at line 390), change the unpacking line (line 398):

```python
        intent, sensors, start, end, label, thr, day, hour = self._understand(question)
```

Then, immediately after the existing no-data guard (originally lines 474-478 — `if data is None or not len(data): return Answer(...)`), insert the hour sub-filter:

```python
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
```

(This sits right before the `# Weather report:` comment that currently follows the no-data guard.)

- [ ] **Step 8: Apply the same hour mask in `_scoped` (chart data)**

In `src/interface.py`, replace `_scoped` (lines 274-282):

```python
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
```

- [ ] **Step 9: Run the new and full test suite**

Run: `cd v1 && python3.12 -m pytest tests/test_interface.py -v -k "anchor or ordinal_month or hour_submask or last_hour_is or now_is or night_bucket or hour_only"`
Expected: 7 new tests PASS.

Run: `cd v1 && python3.12 -m pytest tests/ -q`
Expected: previous baseline (197 passed, 1 skipped) **plus** the 35 new parsing tests **plus** the 7 new interface tests — no failures, no regressions.

- [ ] **Step 10: Commit**

```bash
git add v1/src/interface.py v1/tests/test_interface.py
git commit -m "v1: wire relative/ordinal-period and hour-of-day parsing into Interface"
```

---

### Task 5: Document the new phrasing in `README.md`

**Files:**
- Modify: `v1/README.md` (interface/usage section)

- [ ] **Step 1: Add a subsection listing supported phrasing**

In `v1/README.md`, in the section that documents the natural-language interface (near where `resolve_period`/`resolve_day` examples already live), add:

```markdown
### Time constraints the interface understands

Beyond explicit dates ("26 January 2015", "July 2015", "summer 2012"), the interface understands:

**Relative to the dataset's own end** (the record runs to 1 Jan 2017; there's no real "now" in a
static historical dataset, so these anchor to the record's last reading instead):
`now`, `this hour` / `last hour`, `this week` / `last week`, `this month` / `last month`,
`this year` / `last year`.

**A relative/ordinal qualifier + an explicit year** resolves *inside that year*, not against the
dataset's end — e.g. "last month 2013" → December 2013, "second month 2016" → February 2016,
"last week of 2013", "34th week of 2013", "last day of 2013", "first hour of 2013". Week/day/hour
count fixed-size blocks from 1 January of that year (not ISO calendar weeks).

**Hour-of-day**, applied on top of whatever day/period/relative scope is also named — clock times
("3pm", "15:00", "3 o'clock", "noon", "midnight") and time-of-day buckets ("morning" 06–12,
"afternoon" 12–18, "evening" 18–21, "night" 21–06, wraps past midnight). Asked alone (no day/period),
an hour filter reads across the whole record — e.g. "what's the hottest it's ever been at 3pm?".

Examples: `"temperature last month"`, `"wind speed last hour"`, `"second month 2016"`,
`"temperature at 3pm on 26 January 2015"`, `"mean wind speed at night in December 2016"`.
```

- [ ] **Step 2: Commit**

```bash
git add v1/README.md
git commit -m "v1: document relative/ordinal period and hour-of-day query phrasing"
```

---

### Task 6: Full verification and worklog entry

**Files:**
- Modify: `WORKLOG.md`

- [ ] **Step 1: Run the full test suite and capture real output**

Run: `cd v1 && python3.12 -m pytest tests/ -q`
Record the exact summary line (e.g. `N passed, 1 skipped`).

- [ ] **Step 2: Run `verify_results.py` to confirm nothing else broke**

Run: `cd v1 && python3.12 verify_results.py`
Record the exact tail line (e.g. `29/29 checks passed`).

- [ ] **Step 3: Live-test a handful of the new phrases through `Interface.ask()`**

Run:

```bash
cd v1 && python3.12 -c "
import pandas as pd
from src import preprocess
from src.interface import Interface

df = preprocess.load_and_clean('data/jena_climate_2009_2016.csv')
iface = Interface(df)
for q in [
    'what was the mean temperature last month?',
    'what was the wind speed last hour?',
    'what was the mean temperature in the second month 2016?',
    'what was the temperature at 3pm on 26 January 2015?',
    'what was the mean wind speed at night in December 2016?',
]:
    print(q, '->', iface.ask(q).computed)
"
```

Record the real printed output (not paraphrased).

- [ ] **Step 4: Append a dated `WORKLOG.md` entry with the real results**

Add a new `### v1 interface: implemented relative/ordinal period + hour-of-day parsing` subsection
under the existing `## 2026-07-15` heading (created earlier this session), containing: the real
pytest summary line, the real `verify_results.py` tail, and the five worked examples' real output
from Step 3.

- [ ] **Step 5: Commit**

```bash
git add WORKLOG.md
git commit -m "v1: record verification results for relative/ordinal/hour parsing"
```
