# Design: v1 Time-Constraint Parsing (hour-of-day, relative & ordinal periods)

_Date: 2026-07-15 · Version: v1 (Jena temperature/weather)_

## Context

The query interface (`src/interface_parsing.py`, `src/interface.py`) was reported to "make mistakes"
with time-scoped questions (e.g. wind speed / temperature bound to a specific hour, date, month, "last
month", "last hour"). Reading the parser and testing it directly confirmed two real gaps (recorded in
`WORKLOG.md`, 2026-07-15):

1. **Hour-of-day is silently dropped.** `resolve_day` parses "3pm on 26 January 2015" as the day (26 Jan
   2015) but discards "3pm" — the answer covers the whole day, not the requested hour, with no signal
   that anything was ignored.
2. **Relative phrases don't resolve.** "last month", "last hour", "yesterday" return `None`/empty from
   both `resolve_period` and `resolve_day`. This either falls through to the generic "I couldn't map
   that" fallback, or — worse — silently answers using the **entire 2009–2016 record**.

Confirmed as *not* part of the gap: numeric dates like "1/3/2012" already parse correctly via
`resolve_day`'s existing day/month/year regex.

**Core principle preserved:** all parsing stays deterministic (regex/keyword, no ML/LLM in the truth
path), matching the rest of `interface_parsing.py`. A third-party date-parsing library and LLM-based
resolution were both considered and rejected — the LLM is optional (may not be running) and the
project's architecture requires scope decisions to never depend on it; a date-parsing library's "now" is
real wall-clock time by default and its interpretations are a black box relative to the rest of the
parser, cutting against this project's explainability focus.

## Scope

In scope: three new parsing functions in `interface_parsing.py`, their wiring into `interface.py`
(`_understand`, `_scoped`, chart/where-label text), unit tests for every phrase, and a `README.md`
subsection documenting supported phrasing.

Out of scope (deferred, not requested): free-form phrasing beyond the list below ("round about
teatime"), multi-unit combinations beyond hour+day/period/relative/ordinal (e.g. "3pm last week of
2013" is fine since hour is always a sub-filter; "the third week of last year" — two relative/ordinal
qualifiers chained — is not).

---

## Part A — New parsing functions (`src/interface_parsing.py`)

### `resolve_relative_period(question, anchor_ts) -> (start, end, label)`

Bare relative phrases (**no year present** in the question), anchored to `anchor_ts` — the dataset's own
last timestamp (`df.index.max()`, i.e. `2017-01-01 00:10`), since a static 2009–2016 record has no real
"now". Same return shape as `resolve_period`, so it slots into the same `(start, end, label)` consumers.

| Phrase | Resolves to |
|---|---|
| `now` | the single most recent reading (`start == end == anchor_ts`) |
| `this hour` | the top of the anchor's clock hour up to the anchor itself (partial) |
| `last hour` | the full clock hour immediately before that |
| `this week` | rolling 7 days ending at the anchor (partial view of "this week") |
| `last week` | the rolling 7-day window immediately before that |
| `this month` | the 1st of the anchor's month up to the anchor (partial month, Jan 2017) |
| `last month` | the full previous calendar month (Dec 2016) |
| `this year` | 1 Jan of the anchor's year up to the anchor (partial year, 2017 — a 10-minute stub, since the anchor is the very first reading of the year) |
| `last year` | the full previous calendar year (2016) |

Checked in `_understand` **after** `resolve_period`/`resolve_day` (an explicit date always wins) and
**after** `resolve_ordinal_unit_in_year` (a year present changes the meaning entirely — see below).

### `resolve_ordinal_unit_in_year(question) -> (start, end, label)`

A relative/ordinal qualifier **combined with an explicit year** resolves *inside that year*, not against
the dataset anchor — e.g. "last month 2013" → December 2013, "second month 2016" → February 2016. This
is unambiguous once the year is known, so no clarification prompt is needed.

- **month**: word-ordinals `first`…`twelfth` (months only run 1–12, so numeric ordinals aren't needed)
  plus `last` → 12. `"<ordinal> month (of|in) <year>"` → that calendar month.
- **week**: `first`, `last`, or a numeric ordinal (`"34th week of 2013"`) → the Nth consecutive 7-day
  block from 1 Jan of that year (**not** an ISO calendar week — ISO week numbering has year-boundary
  edge cases, e.g. week 1 of a year can start in the prior December; a fixed 7-day-block-from-Jan-1
  definition is simpler, fully deterministic, and will be stated plainly in the README so nobody assumes
  ISO semantics). `last` → the final such block (may be shorter than 7 days on non-leap-divisible years).
- **day**: `first`, `last`, or a numeric ordinal (day-of-year, `"200th day of 2013"`) → `1 Jan + (N-1)
  days`. `last` → 31 December.
- **hour**: `first`, `last`, or a numeric ordinal (hour-of-year) → `1 Jan 00:00 + (N-1) hours`, a
  one-hour window. `last` → 23:00–24:00 on 31 December.

Checked in `_understand` **before** `resolve_relative_period` whenever a 4-digit year is present
alongside a relative/ordinal qualifier.

### `resolve_hour(question) -> (start_hour, end_hour) | None`

An hour-of-day **sub-filter**, independent of which day/period/relative/ordinal scope was resolved —
same role `resolve_threshold` plays alongside `resolve_period` today. Returns a half-open `[start_hour,
end_hour)` pair (ints, 24h clock; `end_hour` may be `< start_hour` for a bucket that wraps past
midnight, e.g. "night").

- **Clock times**: `"3pm"`, `"15:00"`, `"3 o'clock"`, `"noon"` (12–13), `"midnight"` (0–1) → a one-hour
  window.
- **Time-of-day buckets**: `"morning"` (6–12), `"afternoon"` (12–18), `"evening"` (18–21), `"night"`
  (21–6, wraps).

When no day/period/relative/ordinal scope is *also* named, the hour filter applies across the whole
record (e.g. "what's the hottest it's ever been at 3pm?" reads every 3pm reading in the record) —
consistent with how a dayless "26 January" already reads across all years.

---

## Part B — Wiring (`src/interface.py`)

`_understand()`'s scope resolution becomes an ordered chain, first match wins:

1. `resolve_day` (explicit calendar day)
2. `resolve_period` (explicit year/month/season/range)
3. `resolve_ordinal_unit_in_year` (ordinal/relative qualifier **+ explicit year**)
4. `resolve_relative_period` (bare relative qualifier, dataset-anchored)
5. none → whole record

`resolve_hour()` is then applied as an **additional mask** on top of whichever scope won (steps 1–5),
narrowing the resulting rows by `idx.hour`. `_scoped()` (chart data) and the `where`-label text
(`_slice`/`_answer`) use the same chain + hour mask so charts and answer text never disagree.

`Interface.__init__` gains `self._anchor = df.index.max()`, computed once, passed into
`resolve_relative_period` on each call (kept as an explicit parameter rather than implicit global state,
so the function stays pure/testable per the module's existing "no I/O, no hidden state" convention).

---

## Part C — Tests

New cases in `tests/test_interface_parsing.py` (or a new file if it's cleaner given the volume):

- Every relative phrase in the table above, using a **fixed fake `anchor_ts`** (no real-clock
  dependency) — asserts exact `(start, end, label)`.
- Every ordinal-in-year case (`first`/`last`/numeric, all four units) — self-contained, no anchor needed.
- Every hour format (clock times + buckets), including the midnight-wrap case for "night".
- `Interface`-level tests: hour mask actually narrows `self.df` rows; a question combining an ordinal
  qualifier + year resolves inside that year (not the dataset anchor); a bare relative phrase resolves
  against the dataset anchor.

## Part D — Docs

New subsection in `v1/README.md`'s interface section listing every supported relative phrase, ordinal
form, and hour format with a worked example each — including the explicit "week = 7-day block from 1
Jan, not ISO week" definition so the semantics are never assumed.

---

## Testing strategy

`python3.12 -m pytest tests/` stays green (currently 197 passing, 1 skipped; new tests add to this).
`python3.12 verify_results.py` unaffected (doesn't touch parsing). No notebook changes required — the
data dictionary/columns are unaffected, only the query-time parser.

## Results logging & verification

After implementation, append a dated entry to `WORKLOG.md` recording: the real `pytest` summary line
(not paraphrased), a few worked examples run live through `Interface.ask()` showing the new phrases
resolving correctly, and confirmation the existing 197/1 baseline still passes unmodified.

## Risks

- **Week definition deviates from ISO 8601** — a deliberate simplification (see Part A), documented in
  both this spec and the README so it's never silently assumed to be ISO-compliant.
- **Chained relative/ordinal qualifiers** ("the third week of last year") are out of scope — will fall
  through to the existing "I couldn't map that" fallback rather than partially/incorrectly resolving.
- **`resolve_hour` across the whole record** (no day/period given) can be a large, slow-to-compute mask
  on 8 years of 10-minute data for some intents (e.g. anomaly detection) — acceptable for this
  dataset's size (~350k rows), flagged here in case it needs profiling later.
