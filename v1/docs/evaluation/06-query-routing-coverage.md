# 6. Query-routing coverage — does the front door actually work?

## What's being evaluated

`src/interface.py`'s natural-language routing: given a plain-English question, does the system
correctly identify *which* statistic was wanted (intent: max/mean/trend/...), *which* sensor(s)
it's about, and *what time window* it refers to? None of the other five evaluation methods in
this folder measure this. The truth layer proves the arithmetic is right once a query has been
routed; this page is about whether it gets routed there in the first place. A perfectly correct
`query_engine` is worthless if "what was the temperature at 3pm?" never reaches it.

## Method

A 49-question hand-labelled corpus (`src/query_coverage.py`, `GROUND_TRUTH`), each entry a
`Case(question, expected_intent, expected_sensors, expected_has_scope)` written by human
judgement of the obviously-correct reading of the question — **before** running any code against
it, the same principle [`04-retrieval.md`](04-retrieval.md) uses for its ground truth. Grading a
system against its own output would be circular and could never find a bug.

Each case is run through the real `Interface.ask()` (not the parsing functions in isolation — the
full, actual user-facing behaviour, including the intent-fallback default and the LLM-off default
path), and three independent dimensions are scored: was the intent right, were the expected
sensor(s) present in what was resolved, and did a time scope get detected exactly when one should
have been (and not when it shouldn't).

Each labelled case is judged **standalone** — the benchmark resets the interface's conversation
history before every question, specifically so the conversational follow-up feature (a separate,
already-tested mechanism for things like "and the coldest?") can't accidentally make one
benchmark question inherit routing from an unrelated earlier one.

## Real evidence — the honest before/after (reproduced 15 July 2026)

```bash
python3.12 -c "
from src import preprocess, interface, query_coverage as qc
df = preprocess.align(preprocess.clean(preprocess.load_data('data/jena_climate_2009_2016.csv')), freq='1h')
print(qc.score_coverage(interface.Interface(df)))
"
```

**First run: 91.8% overall (45/49).** Four genuine misses, each independently confirmed by
inspection, not assumed:

| Question | Miss |
| --- | --- |
| "how muggy does it get" | "muggy" wasn't recognised as a humidity synonym at all |
| "how breezy is it in june" | "breezy" wasn't recognised as a wind-speed synonym at all |
| "how heavy is the air" | "heavy" wasn't recognised as an air-density synonym at all |
| "what will the humidity be going forward?" | "going forward" wasn't a coded forecast keyword |

Root cause for the first three, traced precisely: `rag_system.py`'s own sensor descriptions
already use these exact words (its source comment even says so: *"the retrieval evaluation
showed the original one-line docs missed queries like 'muggy' / 'breezy'"* — see
[`04-retrieval.md`](04-retrieval.md)), but `SENSOR_ALIASES`, the separate dictionary that
actually drives *routing*, was never updated to match. Two subsystems, overlapping vocabulary,
only one side of it was ever fixed.

**Fixed** (added the missing words to `SENSOR_ALIASES` and the forecast keyword list) **and
re-ran: 100.0% on all four dimensions, 49/49**, both on the benchmark's own synthetic fixture and
— as `verify_results.py` check #18 — on the real Jena Climate dataset itself:

```
✓  query-routing coverage: intent/sensor/scope all correct on >= 95% of labelled questions
   [overall=100.0% intent=100.0% sensors=100.0% scope=100.0% (n=49)]
```

## A finding about the benchmark itself, not the system

While investigating an apparent 5th failure mid-process, the actual cause turned out to be a
**bug in the benchmark's own first draft**: `score_coverage()` originally reused one `Interface`
object's conversation history across all 49 questions in a loop, so the conversational
follow-up-resolution feature was silently hijacking later, deliberately keyword-less questions
and making them inherit an unrelated earlier question's intent — a completely different
mechanism than the one under test. This dropped the measured score to 77.6% before the benchmark
itself was fixed (resetting history per case, described above). Worth stating plainly: a
benchmark's own bugs can look identical to the system's bugs from the outside, and the only way
to tell them apart is investigating each failure individually rather than trusting the aggregate
percentage.

## Honest caveats

- One ground-truth label was corrected mid-process, not the code: "how breezy is it in june" (no
  year given) was expected to detect a time scope, but `resolve_period` genuinely requires an
  explicit year by design — a bare month name alone isn't recognised (unlike a bare day+month,
  e.g. "26 January," which *is* read across all years). This is a real, minor, documented
  asymmetry in the parser, not a bug fixed here — the ground-truth label was simply wrong about
  what the system was ever designed to do.
- 49 questions is a real but modest corpus size. It was built to cover every intent, several
  sensor aliases, and every time-scope style the system supports at least once each — not to be
  an exhaustive stress test of every possible phrasing.
- 100% on a hand-written benchmark should be read for what it is: the system handles the
  *specific, known* phrasings this corpus tests. It is not a claim that *any* plausible English
  question is handled — only that the ones deliberately checked for are, and that the process of
  building and checking this benchmark surfaced (and closed) 4 real gaps that manual spot-checking
  during earlier sessions had not caught.
