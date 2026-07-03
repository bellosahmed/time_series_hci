# v1 Improvements Log (report-ready before/after)

_A running record of the post-feedback improvements to v1, written so it can be lifted
straight into the report. Each entry states the problem, the change, and before→after
results. Work is on branch `supervisor-feedback-improvements`._

Groups: **G1** correctness fixes · **G2** local-LLM explainer (Ollama) · **G3** `interface.py`
split · **G4** more detectors + honest evaluation.

---

## G1 — Correctness fixes

### G1.1 — Wind direction treated as a circular variable

**Problem (before).** `wd (deg)` (wind direction, 0–360°) was analysed with *linear* statistics —
plain z-score and Pearson correlation. These are wrong for a circular quantity, where 359° and 1°
are 2° apart, not 358°. Symptoms in the multi-sensor table:

| `wd (deg)` metric | BEFORE (linear) |
| --- | --- |
| z-score anomalies (thr 3.0) | 0 (angular outliers invisible) |
| strongest correlation | `p (mbar)`, r = −0.074 (spuriously weak) |
| top linear \|r\| partners | p (0.074), rho (0.069), wv (0.067) — all near zero |

The near-zero correlations and zero anomalies were artefacts of applying linear methods to an
angle, not a real finding.

**Change.** Added circular statistics in `analysis.py` and routed `wd (deg)` through them:
- `circular_zscore_anomalies()` — angular deviation from the circular mean, scaled by circular std.
- `circular_linear_corr()` — Mardia circular-linear correlation coefficient (in [0, 1]).
- `analyze_all_sensors()` now detects circular columns (default `{"wd (deg)"}`) and uses the
  circular metrics for them, linear for the rest.

**Result (after).** The `wd (deg)` row of the multi-sensor table changes; every other sensor is
unchanged.

| `wd (deg)` metric | BEFORE (linear) | AFTER (circular) |
| --- | --- | --- |
| z-score anomalies (thr 3.0) | 0 (artefact) | 0 (**genuine** — wind blows from a broad range of directions over a year, so the circular spread is wide and no hour is a 3σ angular outlier) |
| strongest correlation | `p (mbar)`, r = −0.074 (spurious) | **`max. wv (m/s)`, coeff = 0.247** (physically sensible: prevailing direction relates to gust strength) |
| method | Pearson r on raw degrees | Mardia circular–linear coefficient in [0, 1] |

The headline: wind direction's reported strongest association moved from a *spurious near-zero
pressure correlation* (an artefact of applying Pearson's r to a wrapped angle) to a *physically
meaningful* association with gust speed. New tests: `test_circular_zscore_detects_true_outlier_that_linear_misses`,
`test_circular_linear_corr_detects_and_ranges`, `test_analyze_all_sensors_wd_uses_circular`.

### G1.2 — Guard functions that require a DatetimeIndex

**Problem (before).** `compare_imputation` (`method="time"`), `analysis.decompose` and
`forecasting.arima_residuals` silently rely on a `DatetimeIndex`; on a plain index they raise an
opaque pandas error. **Change.** Added an explicit `DatetimeIndex` check in `compare_imputation`
(method `"time"`) that raises a clear `TypeError`. **Result (after).** Calling it on a
`RangeIndex` series now raises `"compare_imputation(method='time') requires a DatetimeIndex; got
RangeIndex."` instead of a cryptic pandas failure. New test:
`test_compare_imputation_requires_datetime_index`.

### G1.3 — Compute the correlation matrix once in `analyze_all_sensors`

**Problem (before).** `analyze_all_sensors` called `top_correlations()` per sensor, each of which
recomputed the full `df.corr()` — the correlation matrix was rebuilt 14× on a 14-sensor frame.
**Change.** Compute `correlations(num)` once before the loop and index it per sensor.
**Result (after).** Identical numbers for all linear sensors; one matrix build instead of 14.

### G1.4 — Reconcile the notebook-kernel documentation

**Problem (before).** `CLAUDE.md` says to run notebooks with the `py312-tshci` kernel (the only
env with the dependencies), but the notebooks stored a generic `python3` kernelspec — so a plain
"Restart & Run All" could pick a kernel without the deps. **Change.** Repinned the kernelspec of
notebooks 01 and 02 to `py312-tshci` (metadata only; outputs untouched). **Result (after).** Both
now open on the correct kernel by default, matching the doc. (Notebooks 00/03/04 have unrelated
uncommitted edits and should be repinned when next touched.)

---

## G2 — Local-LLM explainer (Ollama) actually wired in

**Problem (before).** The Ollama explanation layer was already coded (`llm.py`: `generate`,
`explain`, `classify`, `plan`, `faithfulness`; `interface.py` routes through it when
`use_llm=True`). But the default model was the bare tag **`"llama3.2"`**, while the only model
pulled locally is **`llama3.2:3b`**. Ollama does not auto-resolve the bare name, so every call
returned **HTTP 404**, and `interface._phrase` swallowed it with a bare `except Exception: pass`
and silently fell back to the offline template. Net effect: **`use_llm=True` produced output byte-
identical to `use_llm=False`** — the LLM was never actually being used, and nothing surfaced the
misconfiguration.

**Change.**
- Pinned the model to **`llama3.2:3b`** everywhere: `llm.DEFAULT_MODEL`, `interface.Interface`
  default (now references `llm.DEFAULT_MODEL`), `apps/chat.py`, `apps/web/app.py`.
- Added two regression tests (`tests/test_llm.py`) that, when Ollama is up, assert
  `DEFAULT_MODEL` is actually pulled and that `explain` returns real prose (not the template) with
  **no invented numbers** — this would have caught the 404 silent-fallback.

**Result (after).** `use_llm=True` now returns genuine local-LLM explanations that differ from the
template, with every number preserved (the `faithfulness` guard reports `invented: []`):

| Question | TEMPLATE (before / fallback) | LOCAL LLM `llama3.2:3b` (after) |
| --- | --- | --- |
| hottest temperature? | `🔢 Computed: max T (degC) was 37.04 on 4 Jul 2015 16:00.` | "The hottest temperature recorded during this period was **37.04°C**, on **4 July 2015 at 4pm**." |
| corr(temp, humidity)? | `🔢 Computed: correlation(T, rh) = -0.572.` | "There is a strong negative correlation between temperature and humidity — as one goes up, the other tends to go down." |

The core guarantee holds: the number `37.04` is computed by the query engine; the model only
rewords it. This closes the "LLM explains" half of the system's thesis. Full suite: **60 passed,
1 skipped**; `verify_results` **20/20** (its LLM-faithfulness check now exercises a real call).

---

## G3 — Split `interface.py` (separate parsing from orchestration)

**Problem (before).** `interface.py` had grown to **585 lines** doing two distinct jobs: (1) pure
natural-language *parsing* (intent/sensor/date/threshold via keyword + regex) and (2) *orchestration*
(routing to the query engine, RAG context, phrasing, charting). One large file mixing both concerns
is harder to read, test, and change safely.

**Change.** Extracted the entire pure-parsing layer into a new module **`src/interface_parsing.py`**
(224 lines): `SENSOR_ALIASES`, `INTENTS`, `resolve_sensors`, `detect_intent`, `resolve_period`,
`resolve_day`, `resolve_compare`, `resolve_threshold`, and their helper tables (`MONTHS`, `SEASONS`,
`COMPARE`). `interface.py` (now **390 lines**) imports and **re-exports** them, so the public API is
unchanged — `interface.resolve_sensors`, `interface.detect_intent`, etc. still resolve exactly as
before for `apps/web/app.py`, `apps/chat.py`, and `tests/test_interface.py`. No behaviour change; a
pure structural refactor.

**Result (after).**

| | Before | After |
| --- | --- | --- |
| `interface.py` | 585 lines (parsing + orchestration) | **390 lines** (orchestration only) |
| parsing layer | inline | **`interface_parsing.py`**, 224 lines, independently testable |
| public API | `interface.<parser>` | unchanged (re-exported) |

Verified: the 48 interface tests still pass, full suite **60 passed / 1 skipped**, `verify_results`
**20/20**, and a live **web smoke-test** (launched `apps/web/app.py`, POSTed `/api/ask`) returned
correct answers in both template mode ("max T (degC) was 37.04…") and LLM mode ("Yes, 2013 was
hotter than 2014 by a margin of 1.25°C…").

---

## G4 — More detectors + honest real-event evaluation

**Problem (before).** The detector comparison used only **synthetic** injected spikes (exact but
artificial labels), and covered z-score / rolling z-score / STL-residual / ARIMA / Isolation Forest.
Two common baselines (LOF, One-Class SVM) were missing, and no detector had ever been scored against
*real* documented events. The ARIMA order was also an unexplained hard-coded `(2, 1, 2)`.

**Change.**
- Added **`anomaly.lof_anomalies`** (Local Outlier Factor, density-based) and
  **`anomaly.ocsvm_anomalies`** (One-Class SVM, with a fit-subsample guard so the kernel SVM stays
  tractable on ~70k rows — fit on ≤15k sampled rows, score all). No new dependencies (both sklearn).
- Added **real ground-truth events** in `evaluation.py`: `KNOWN_EVENTS` / `known_events()` /
  `label_from_events()` — five documented Central-European events in the record (2009 winter cold
  snap, 2010 July heat, 2012 February cold wave, 2013 August heat, 2015 July heatwave), cross-checked
  against the record's own extremes. This complements the synthetic evaluation with *real* labels.
- Documented the fixed ARIMA `(2, 1, 2)` order (deliberate low order on the deseasonalised series;
  `order` exposed for tuning) in `forecasting.arima_residuals`.
- New tests for both detectors and the event labelling.

**Result (after).** All seven detectors scored against the **real** event labels (1,512 hours across
5 events, 2.16% of the record), point-adjusted (a detector gets credit for an event if it fires
anywhere inside the window):

| detector | PA precision | PA recall | PA F1 | points flagged |
| --- | --- | --- | --- | --- |
| z-score (global) | **0.965** | 0.841 | 0.899 | 157 |
| rolling z-score | 0.703 | 0.540 | 0.611 | 349 |
| STL-residual z-score | 0.736 | 0.905 | 0.812 | 526 |
| ARIMA residual | 0.750 | 0.778 | 0.764 | 411 |
| **Isolation Forest** | 0.804 | **1.000** | **0.891** | 699 |
| LOF *(new)* | 0.689 | **1.000** | 0.816 | 702 |
| One-Class SVM *(new)* | 0.233 | 1.000 | 0.378 | 5,140 |

Honest reading: **Isolation Forest is the best overall** (F1 0.891, perfect recall) and **global
z-score is the precision champion** (0.965 — it fires only 157 times yet catches 84% of events).
The new **LOF** is competitive (perfect recall, F1 0.816); the new **One-Class SVM honestly loses**
— perfect recall but precision 0.233, over-flagging 5,140 points on this univariate temperature
signal. Reporting a baseline that underperforms is deliberate: it shows the evaluation is real, not
cherry-picked. Full suite: **63 passed, 1 skipped**; `verify_results` **20/20**.
