# 8. External-LLM benchmark — are we better than "just ask an LLM"?

## What's being evaluated

Every other page here asks "is the system internally correct?". This one asks the question a
sceptical reader actually has: **why build a deterministic truth layer at all, when you could just
paste the data into a chatbot and ask?** So we run a like-for-like contest — the *same* question,
with the *same* data, given to our application and to general-purpose LLMs — and score all of them
against one independent ground truth.

This is a *comparative* evaluation (relative to external baselines), not a proof of internal
correctness (that's [`01-truth-layer.md`](01-truth-layer.md)). It measures the **value added** by
the engineering, using other AI tools as the benchmark.

## Method

Harness: `src/benchmark_external.py`; runners under `docs/evaluation/benchmark/`.

- **Gold set** (`GOLD`): 8 natural-language questions — max/min/mean across five sensors
  (temperature, humidity, pressure, wind speed, dew point).
- **Fairness by design.** Each question is scoped to a **single day** (144 ten-minute readings),
  and that day's raw readings are **embedded verbatim in the prompt**. No contestant is guessing
  blind — the data is right there. This isolates *computation over the data* (where our engine has
  a real advantage) from *access to the data* (which would be an unfair advantage). The identical
  prompt string is reused for every contestant.
- **Ground truth** = the oracle: the exact answer recomputed independently from the raw CSV with
  the same decisions the app makes — drop duplicate timestamps (the dataset has 183 in 2014 alone),
  treat `-9999` as missing, compute on the native 10-minute grid. See the oracle cross-check that
  established these (max/min/mean all match the app to the digit).
- **Scoring**: exact within ±0.01 of ground truth.
- **Contestants**:
  - *our app* — full pipeline, LLM off (pure deterministic parser + query engine);
  - *local LLMs* — the three models pulled in Ollama, run automatically (`run_local.py`);
  - *hosted LLMs via OpenRouter free tier* — real large models (e.g. Nemotron 550B/120B, OpenAI
    `gpt-oss-20b`, Gemma-4-31B) called automatically with the same prompts (`run_openrouter.py`).
    The API is OpenAI-compatible; needs a free key (`OPENROUTER_API_KEY`). The exact `kimi`/`glm`/
    `deepseek` ids can be passed via `--models` when they're on the free roster or a small balance
    is available.
  - *commercial tools* (ChatGPT, Claude, Kimi, GLM) with **file upload** — run by hand via the
    generated **manual pack** (`make_manual_pack.py` → `manual_pack.md`, `answers_template.csv`;
    score with `score_manual.py`). The pack tells the user to upload the full CSV and ask the
    natural question — the tools' real code-interpreter path, and the setting most likely to expose
    the duplicate-timestamp / resolution traps.

## Real evidence (automated leg, 30 July 2026)

```bash
python3.12 docs/evaluation/benchmark/run_local.py
```

| Contestant | Accuracy | Notes |
| --- | --- | --- |
| **our app** | **8 / 8 (100%)** | deterministic; every answer == oracle |
| Gemma-4-E4B (7 GB, Q6) | 6 / 8 (75%) | got **every extremum** right; **missed both means** |
| llama3.2 (3B) | 1 / 8 (12%) | small errors on most (37.13 vs 37.28); one lucky exact |
| Gemma-3-1B ("thinking") | 0 / 8 (0%) | unusable: returned −1.00, 23.49, 23.00 … |

Per-question detail and every raw model reply are saved to
`docs/evaluation/benchmark/results_local.csv` for audit.

## Real evidence (hosted LLMs via OpenRouter free tier, 30 July 2026)

```bash
OPENROUTER_API_KEY=... python3.12 docs/evaluation/benchmark/run_openrouter.py
```

Same 8 questions, same in-prompt data, scored against the same oracle. Five free-tier models:

| Contestant | Accuracy | Notes |
| --- | --- | --- |
| **our app** | **8 / 8 (100%)** | deterministic; exact, instant, offline |
| nvidia/nemotron-3-super-120b | 8 / 8 (100%) | perfect incl. both means; 13–92 s/question |
| inclusionai/ling-3.0-flash | 8 / 8 (100%) | perfect incl. both means; fastest (1–41 s) |
| nvidia/nemotron-3-ultra-550b | 7 / 8 (88%) | one **API error** (malformed response), not a reasoning miss; slow (54–160 s) |
| openai/gpt-oss-20b | 6 / 8 (75%) | missed both near-aggregate cases (95.9 vs 96.2; 71.32 vs 71.98); one call **477 s** |
| google/gemma-4-31b | 1 / 8 (12%) | **rate-limited (429) on 6/8** — infrastructure, not capability |

Raw replies saved to `docs/evaluation/benchmark/results_openrouter.csv`.

**Reading.** Unlike the tiny *local* models, capable *hosted* models **can** match the app on this
task — including the means that broke local Gemma-4. So against a strong baseline the app's edge is
no longer "the only one that can compute"; it is that it is exact **and** instant **and** reliable
**and** needs no network or quota. The failure modes that showed up here — a malformed API response,
a 477 s latency spike, rate-limiting that wiped out a whole model's run, and gpt-oss still slipping
on aggregation — are exactly the operational risks a deterministic local engine does not carry.

## Real evidence (routing / understanding — the worksheet, 30 July 2026)

The benchmarks above test *computation*. This one tests *understanding*: mapping a plain-English
question to the right (intent, sensor(s), time-scope). Each model gets the identical classification
contract (`llm.independent_intent_judgment`); scored against the hand-labelled
`query_coverage.GROUND_TRUTH` (the blind worksheet in `blind-worksheet/worksheet.csv`), n = 49.

```bash
python3.12 docs/evaluation/benchmark/run_openrouter_routing.py --models inclusionai/ling-3.0-flash:free
```

| Contestant | Intent | Sensors | Scope | Overall |
| --- | --- | --- | --- | --- |
| **our app** (deterministic parser) | 100% | 100% | 100% | **100%** |
| inclusionai/ling-3.0-flash | 59% | 73% | 78% | **53%** |

A second hosted model (nemotron-3-super-120b) was attempted for a sturdier comparison but is
**excluded**: by then the OpenRouter **free-tier daily request cap** was exhausted, so all 49 calls
returned HTTP 429 and the model never actually answered — a quota artifact, not a 0% routing score.
It can be re-run for free after the daily reset. (This is a limitation of the *free* measurement
setup, not of the method.)

**Reading.** On *understanding* the same model that scored 8/8 on *computation* gets only ~half the
routing right. The errors are systematic: it collapses single-sensor **"summary"** questions into
the whole-day **"weather"** report intent (e.g. "the water vapour concentration", "vapour pressure
deficit last year"), and misreads implicit sensors ("wind gust last week" → `max`, not a
`max. wv (m/s)` summary). A general LLM is good at *computing over data you hand it* but much weaker
at *deciding what a messy question is actually asking* — which is the parser's job.

**Honest caveat.** The app's 100% is flattered because this ground-truth corpus and the parser
co-evolved (the self-authored-labels limitation already flagged in `06-query-routing-coverage.md`).
The LLM judge is genuinely independent, so the *gap* is real and informative even though the app's
absolute score is not an unbiased estimate of open-world routing accuracy.

## What this shows (the honest reading)

1. **The data being present is not enough.** Even with all 144 numbers in the prompt, the general
   models make arithmetic/reading mistakes. The deterministic engine cannot — that's the point of
   computing numbers in code, not in a language model.

2. **Aggregation is where LLMs fail hardest.** The strongest local model (Gemma-4-E4B) recovered
   *every* max and min correctly but got *both* means wrong (e.g. 28.36 vs 29.38): picking the
   largest of a list is easy for an LLM; mentally averaging 144 numbers is not. This is a clean,
   reproducible illustration of why "let the LLM read the table" is unsafe for anything beyond
   trivial lookups.

3. **Model size dominates.** The 1B "thinking" model is not merely inaccurate but incoherent
   (negative pressures, constant 23.x answers) — a reminder that a small *local* model, which is
   what a privacy-preserving offline deployment can actually run, is exactly the case where an
   external truth layer matters most.

4. **Scope / honesty.** This is 8 questions on single-day windows — a demonstration, not a
   large-sample study, and the local models here are small. The commercial tools (with code
   interpreters over the full file) are the harder baseline and are expected to score higher; the
   manual-pack leg exists to measure them fairly rather than assert a result we didn't run. Even
   there, the interesting outcome is *not* "who scores 100%" but *whether* a general tool silently
   averages to hourly or keeps the duplicate timestamps — the two preprocessing decisions our
   oracle had to pin down, and which a general tool has no reason to get right by default.

## Verification — the app as a confirmed-correct reference (30 July 2026)

The whole benchmark scores everyone against the app/oracle, so that reference must itself be proven
correct. `verify_benchmark.py` does this locally (no API cost), three ways:

```bash
python3.12 docs/evaluation/benchmark/verify_benchmark.py
```

**1. App == independent oracle on every gold question** — the exact answers, confirmed against the
raw dataset:

| # | Question | App = oracle (dataset) |
| --- | --- | --- |
| 1 | highest temperature on 4 July 2015 | **37.28 °C** ✔ |
| 2 | lowest temperature on 4 July 2015 | **20.15 °C** ✔ |
| 3 | average temperature on 4 July 2015 | **29.38 °C** ✔ |
| 4 | maximum humidity on 2 October 2012 | **96.20 %** ✔ |
| 5 | minimum pressure on 15 January 2014 | **982.34 mbar** ✔ |
| 6 | highest wind speed on 10 February 2016 | **9.90 m/s** ✔ |
| 7 | highest dew point on 20 August 2011 | **12.61 °C** ✔ |
| 8 | average humidity on 2 October 2012 | **71.98 %** ✔ |

8/8 — the app's answer equals the value recomputed independently from the raw CSV (with the same
de-duplication + sentinel handling).

**2. Re-grading with the app as judge reproduces the oracle grades exactly.** Every contestant's
saved answer was re-scored against the *app's* value instead of the oracle's; all 9 contestants got
identical verdicts either way (**zero disagreements**), so no "✓/✗" in the tables above depended on
which reference was used.

**3. App routing == hand-labelled truth**: intent/sensors/scope/overall all 100% (n = 49).

Verdict: the app is a confirmed-correct reference for the entire benchmark — its numbers match the
dataset, it grades others the same way the independent oracle does, and it routes the labelled
questions correctly.

## Reproduce

```bash
# confirm the app is a correct reference (local, no cost)
python3.12 docs/evaluation/benchmark/verify_benchmark.py

# automated: our app vs local Ollama models
python3.12 docs/evaluation/benchmark/run_local.py

# hosted free LLMs via OpenRouter (needs a free key)
OPENROUTER_API_KEY=sk-or-... python3.12 docs/evaluation/benchmark/run_openrouter.py
#   or target specific models: --models moonshotai/kimi-k2,z-ai/glm-4.6

# commercial tools (manual): generate pack, fill it in, score it
python3.12 docs/evaluation/benchmark/make_manual_pack.py
#   -> upload data/jena_climate_2009_2016.csv to each tool, ask the questions in manual_pack.md,
#      record numbers in answers_template.csv
python3.12 docs/evaluation/benchmark/score_manual.py docs/evaluation/benchmark/answers_template.csv
```
