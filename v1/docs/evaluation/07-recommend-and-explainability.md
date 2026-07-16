# 7. `recommend` and correlation explainability — does the newest layer actually work?

## What's being evaluated

Two related additions built 16 July 2026: the `recommend` intent (analog similarity search —
"a day like 13 February 2013" — and directional ranking — "the hottest days in June" — plus a
forward-looking "when next" mode for both), and an explainability layer on top of `correlation`
(qualitative strength labels, and a "why is that only moderate?" breakdown). None of the other six
pages in this folder cover this code at all — it didn't exist when they were written.

This page follows the same discipline as the rest of the folder (independently recompute, don't
trust the system's own output back at itself) but the *shape* of what needs checking is different
from anywhere else here: the interesting claims aren't just "is the number right" (the truth
layer already guarantees that at the query-engine level) but "does the system correctly restrict
its own search space," "is its honesty caveat actually triggered by real mismatches, not just
silent," and "does an explanatory add-on trace back to real evidence."

## Method

Four checks were added to `verify_results.py` (#19–22), each recomputing something a *different*
way rather than re-asking the system to confirm its own answer:

1. **Analog restriction correctness** — ask a scoped analog question through the real
   `Interface`, then independently confirm every returned candidate date actually falls inside
   the named month by filtering the raw column directly (not by trusting the interface's own
   restriction logic).
2. **Directional honesty threshold** — recompute the "isn't genuinely hot" threshold from raw
   pandas (`df[sensor].resample("1D").mean().quantile(0.75)`) and check it matches what the
   system reported, rather than just checking the boolean flag.
3. **Correlation breakdown** — recompute R² and the summer-season correlation from a fresh
   pandas `groupby`-style filter, not by calling `query_engine.correlation_breakdown` a second
   time (which would only prove the function agrees with itself).
4. **The `is_causal_query` fix** (found and fixed the same day, see below) — run the exact
   reported bug pattern through the real `Interface.ask()` and confirm the intent routes
   correctly, not just that the underlying parsing function returns the right boolean in
   isolation.

## Real evidence — all four pass on the real dataset (reproduced 16 July 2026)

```bash
python3.12 verify_results.py
```

```
✓  recommend analog restriction: every candidate independently confirmed inside April 2013
✓  recommend honesty threshold == independently-recomputed 75th percentile of daily means
✓  correlation breakdown R² and summer split == independent pandas recomputation
✓  bare 'because' no longer hijacks an unrelated question into the correlation intent
```

35/35 checks now pass overall (up from 31/31), all four new ones included in the pass/fail count
(not advisory) since all four are deterministic and reproducible, unlike the LLM-dependent checks
elsewhere in this file.

## A real bug this evaluation work found: `is_causal_query` was hijacking unrelated questions

Not found by writing the verification check first — found by testing a concrete example before
adding the check, per this project's own stated discipline (§`README.md` point 1: "define ground
truth externally... never grade the code against its own output"). `is_causal_query` did plain
substring matching (`"cause" in question.lower()`), and "cause" is a literal substring of
"because". Any question containing "because" — even with no causal claim at all — was silently
routed into the `correlation` intent instead of whatever was actually asked:

```python
>>> iface.ask("I stayed in because it was cold, what was the average temperature in 2013?")
# intent: correlation
# "This system measures correlation... not causation... Ask 'how does X relate to Y?'"
# -- never answers the average-temperature question actually asked
```

Fixed with `\b` word-boundary matching (the convention the rest of `interface_parsing.py` already
uses everywhere else — this function had drifted from it). Re-verified: the same sentence now
correctly returns `intent: mean`. This is the second time a plain-substring keyword match has
caused a real routing bug this project (the first: `resolve_sensors` matching "thin" inside
"within" — see `WORKLOG.md`, 16 July 2026) — worth treating as a standing lesson for any future
keyword list added to this codebase: use `\b` boundaries by default, treat plain substring search
as the exception that needs justifying, not the default.

## Honesty caveat and forward-looking mode, evaluated by construction

Two design properties of the `recommend` intent are inherently about *when the system says
nothing is a good match*, not about a single number being right or wrong — evaluated by
constructing cases where the honest answer is known in advance:

- **The multi-sensor honesty caveat must catch a mismatch in ANY requested sensor, not just the
  first.** Verified with a synthetic fixture engineered so the first sensor (temperature) has NO
  seasonal signal and the second (pressure) carries the whole mismatch — the caveat correctly
  names pressure, not temperature (`tests/test_interface_recommend.py`,
  `test_recommend_honesty_caveat_catches_a_mismatch_in_a_sensor_other_than_the_first`). On the
  real dataset, restricting "similar to June 2013" to January 2013 (default 5-sensor comparison)
  flagged **pressure** as the worst mismatch (1.43× the record's typical spread) even though
  temperature that day was actually a reasonable match (0.81×) — exactly the case a
  first-sensor-only check would have missed.
- **"When next" in directional mode must not fabricate a fake future date.** This is a static
  historical record (2009–2016) with no real "now" beyond its own last row — a forward-looking
  directional question ("when next will it be really hot?") has no genuine future window to
  search. Verified the system says so explicitly rather than either staying silent or inventing a
  plausible-looking date: *"This is a historical record (ends 01 Jan 2017), so there's no real
  'next' to point to — but going by the pattern, hot days... happen most around July, August."*

## Honest caveats

- These four checks are a genuinely different discipline from the ~350 unit tests covering the
  same features (`tests/test_interface_recommend.py`, `test_query_engine.py`,
  `test_interface_intent.py`) — the unit tests use small synthetic fixtures engineered to isolate
  one behaviour each; these checks run on the real 70k-row Jena dataset and recompute independently
  from raw pandas. They're complementary, not redundant: the unit tests catch a much wider range of
  edge cases (they're the ones that actually caught the `is_causal_query` and `resolve_sensors`
  bugs during development), while these four are the smaller set of claims worth re-proving against
  real data specifically, the same role every other check in `verify_results.py` plays for its own
  component.
- Not every feature built 16 July 2026 has a dedicated independent-recomputation check here — the
  target/restriction-swap parsing (`split_target_clause`), the forward-looking window default (3
  months), and the correlation strength-label boundaries are covered thoroughly at the unit-test
  level but not duplicated into `verify_results.py`, since they're deterministic parsing/labelling
  logic with no real-dataset-specific claim to independently re-verify (unlike, say, the honesty
  threshold, which depends on the actual distribution of December vs. the whole record).
- Like `06-query-routing-coverage.md`, this page's evidence is entirely self-authored — the four
  checks were designed and run by the same person who wrote the code they test. The mitigation is
  the same one used throughout this folder: recompute independently from raw pandas rather than
  re-calling the system's own functions, so a check can't pass merely because the code agrees with
  itself.
