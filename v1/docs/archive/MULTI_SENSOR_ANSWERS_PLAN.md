# Multi-sensor answers, correlation clarification, wider memory window: design

Drafted **2026-08-22**. Design only — no code changed yet. See `CHAT_MEMORY_PLAN.md` for the
separate, still-open **client-side** (web `index.html`) unbounded-history issue; this plan only
touches the **server-side** `Interface.history` window, which is a different variable.

## Diagnosis — verified live against the running app

**Bug 1: single-statistic intents silently drop every sensor after the first.**

`Interface._answer()` sets `sensor = sensors[0] if sensors else "T (degC)"` once, near the top
(`interface.py:977`), and every single-statistic intent branch downstream (`max`, `min`, `mean`,
`spread`, `summary`, `trend`, `threshold`, `forecast`, `daytype`) reads that one variable. When a
question names more than one sensor, only the first is ever answered. Confirmed live:

```
Q: what was the mean humidity and temperature in april 2009?
🔢 Computed: rh (%) in April 2009 mean=69.18, std=19.19, range=69.57.
🧭 Understood: sensor=rh (%), T (degC); scope=April 2009
```

The 🧭 line proves the router found both sensors; the 🔢 line silently answers only one, with no
indication temperature was dropped. This is a transparency gap, not just a missing feature: the
system claims to have understood the full question and answers a smaller one.

**Bug 2 (same root cause, correlation-specific): 3+ sensors silently truncate to the first two.**

```
Q: how does temperature, humidity and pressure relate?
🔢 Computed: correlation(T (degC), rh (%)) = -0.572 (moderate negative correlation).
🧭 Understood: sensor=T (degC), rh (%), p (mbar)
```

Pressure is dropped with no indication. `get_correlation(df, a, b)` (`query_engine.py:102`) is
inherently pairwise, which is fine, but the interface never says so.

**Not a bug:** the `weather` (day-report) intent already returns a fixed multi-sensor summary
(temperature, humidity, dew point, pressure, wind) regardless of which words were used, so
"how is the weather... humidity and temperature" already reads as multi-sensor. It is not filtered
to only the named sensors, but nothing is silently dropped either, since it never claimed to be
sensor-specific in the first place.

**Memory window:** `self.history` is capped at the 6 most recent turns
(`self.history[-6:]`, four call sites: `interface.py:694, 748, 788, 817`). Confirmed by inspection;
this is a deliberate, working cap, not a bug — the ask is simply to widen it.

## Plan

**1. Multi-sensor loop for single-statistic intents.** In `_answer()`, when `len(sensors) > 1` and
the intent is one of the single-statistic intents (max, min, mean, spread, summary, trend,
threshold, forecast, daytype — i.e. every intent currently reading the shared `sensor` variable,
excluding `correlation`, `compare`, and `weather`), loop over each matched sensor, calling the
existing per-sensor logic unchanged once per sensor, and combine the resulting per-sensor lines
into one answer (one 🔢 line per sensor), in the style the `weather` intent already uses. The
single-sensor path (`len(sensors) <= 1`, today's only path, and the one every existing test and
the 72-question benchmark exercises) is untouched.

Context and chart, for the multi-sensor case: the 📚 Context line shows the first named sensor's
description only, to avoid stacking several paragraphs of context under one answer. The web chart
continues to plot the first named sensor only, matching today's single-sensor behaviour; charting
every named sensor together is a possible future extension, not part of this change.

**2. Correlation: stay pairwise, but say so.** When 3+ sensors are named on a correlation question,
use the first two (as today) and state explicitly that correlation compares two sensors at a time
and which two were used, instead of silently dropping the rest. No loop here: correlation measures
a relationship between two things, which does not reduce to "compute it once per sensor" the way
independent single-sensor facts do. An all-pairs matrix is a possible future extension, not part of
this change.

**3. Memory window: 6 → 10.** Change the four `[-6:]` slices in `interface.py` to `[-10:]`. No other
change; does not touch the separate client-side (web) history-growth issue tracked in
`CHAT_MEMORY_PLAN.md`.

## What does not change

- The faithfulness gate needs no change: it already extracts and checks every number in the phrased
  text, regardless of how many there are.
- No change to `query_engine.py`, `rag_system.py`, or the `weather` intent.
- No change to the client-side web history handling (separate, already-tracked issue).

## Testing

- New case: "what was the mean humidity and temperature in april 2009?" → expect two 🔢 lines
  (temperature and humidity), each matching the existing single-sensor computation independently
  verified elsewhere (`== pandas`).
- New case: the 3-sensor correlation example above → expect the explicit "compares two sensors at a
  time" wording plus the unchanged numeric correlation for the first two.
- New case: history length after 11 turns == 10 (was 6).
- All existing single-sensor tests must still pass unmodified, since that code path does not change.

## Risk

Low. The single-sensor path, which is what every currently-drafted report chapter and the planned
Chapter 4 evaluation describes, is unchanged. This is additive: new behaviour only activates when
more than one sensor is named, a case nothing currently tested or written up depends on.
