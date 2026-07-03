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
