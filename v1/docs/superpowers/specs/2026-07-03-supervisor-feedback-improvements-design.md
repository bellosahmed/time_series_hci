# Design: v1 Supervisor-Feedback Improvements

_Date: 2026-07-03 · Version: v1 (Jena temperature/weather)_

## Context

Supervisor feedback on v1 (notes in `../supervisor update`, 1 July 2026) asked for: data
availability & correlation, distribution/graph shape, skewness & data imbalance, an explicit
account of how missing data is handled, the std→outlier link, verification that the algorithm
gives the right answer, function-return documentation, and — importantly — **verification across
humidity and the other sensors, not just temperature**.

Mapping the feedback to the code showed most items already exist (correlation, interpolation,
z-score/std, `verify_results.py`); the genuine gaps are: skewness/imbalance stats, a data-availability
report, and running the pipeline across all 14 sensors. The rest is description.

**Core principle preserved:** code computes all numbers; prose only explains them.

## Scope

In scope: A (new analysis code) + B (multi-sensor verification), then C (description/docs).
Out of scope (deferred): cloud, speed/profiling. No new models, no refactors beyond the additions.

## Sensors (all 14)

`p (mbar)`, `T (degC)`, `Tpot (K)`, `Tdew (degC)`, `rh (%)`, `VPmax (mbar)`, `VPact (mbar)`,
`VPdef (mbar)`, `sh (g/kg)`, `H2OC (mmol/mol)`, `rho (g/m**3)`, `wv (m/s)`, `max. wv (m/s)`,
`wd (deg)`. Feedback singled out humidity (`rh (%)`) — every one of these gets covered.

---

## Part A + B — Code (one slice)

### `src/analysis.py`

- **`distribution_stats(df) -> pd.DataFrame`**
  Per-sensor rows with columns: `mean, median, std, skew, kurtosis, min, max`.
  Uses pandas `.skew()` / `.kurt()` (Fisher, excess kurtosis). Answers "shape of the graph" +
  "skewness". Return meaning documented in the docstring.

- **`data_availability(raw_df, sentinels=(-9999.0,)) -> pd.DataFrame`**
  Computed on the **raw** (pre-clean) frame. Per-sensor rows: `n_total, n_missing, n_sentinel,
  n_real, available_pct`. `n_missing` = NaN count; `n_sentinel` = count of sentinel values;
  `n_real = n_total - n_missing - n_sentinel`; `available_pct = 100 * n_real / n_total`.
  Answers "availability of the data" and grounds the missing-data discussion.

- **`analyze_all_sensors(df, threshold=3.0) -> pd.DataFrame`** (B)
  One summary row per sensor: `skew, kurtosis, zscore_anomalies` (count from
  `anomaly.zscore_anomalies`), `top_corr_sensor`, `top_corr_r` (strongest |r| partner from
  `top_correlations`). This is the artifact that proves humidity/pressure/wind were all exercised,
  not just temperature.

### `src/evaluation.py`

- **`anomaly_imbalance(flags) -> dict`**
  `{n_anomaly, n_normal, n_total, anomaly_pct}` for a boolean flag Series. Answers "imbalance of
  the data" for the detector labels (anomalies are a tiny minority — the imbalance the supervisor
  meant).

### `src/preprocess.py`

- **`compare_imputation(series, method="time") -> pd.DataFrame`**
  On artificially held-out points, compares interpolation vs mean-fill (abs error per strategy) so
  the interpolation choice is *justified with numbers*, answering "how does the AI handle missing
  data — average or distribution or trend". Keeps it small; returns a per-strategy error summary.

### Tests — `tests/test_analysis.py` (new) + additions

- `distribution_stats`: skew≈0 for symmetric input, positive for right-skewed; columns present.
- `data_availability`: a frame with known NaNs + `-9999`s returns exact counts and `available_pct`.
- `analyze_all_sensors`: returns one row per input column with expected columns.
- `anomaly_imbalance`: counts and pct match a hand-built flag Series.
- `compare_imputation`: returns finite errors; interpolation ≤ mean-fill error on a smooth ramp.

### `verify_results.py`

Add independent checks (raising the 17/17 count) that recompute `data_availability` and
`distribution_stats` against direct pandas and assert equality — reinforcing "control the
algorithm → the right answer".

---

## Part C — Description (second slice, after A+B produces numbers)

### `report/DATA_QUALITY.md` (new)

Sections, each describing **real outputs** from A/B:
1. **Data availability** — the `data_availability` table; how much per sensor is real vs interpolated.
2. **Distribution & skewness** — the `distribution_stats` table; which sensors are skewed and what
   the shape means (e.g. wind speed right-skewed; temperature near-symmetric).
3. **Missing-data method** — interpolation (time-aware) chosen over mean/distribution fill, with the
   `compare_imputation` evidence; explains it follows the local trend rather than flattening to a mean.
4. **std → outlier threshold** — how z-score uses std(ddof=0) and why threshold 3.0/4.0, citing the
   existing `threshold_sensitivity` sweep.
5. **Multi-sensor verification** — the `analyze_all_sensors` table across all 14 sensors (the
   humidity/other-sensor answer).
6. **Ground-truth control** — surfaces `verify_results.py` (n/n) and query-engine==pandas as the
   "right answer" evidence.

### Notebook cells (minimal, to keep re-runs safe)

- `01_data_exploration_and_visualization`: add availability + skewness cells (call the new functions,
  show the tables).
- `02_methods_and_evaluation`: add the multi-sensor `analyze_all_sensors` table + a short std→threshold
  description cell.

### Docstring pass

On every touched function, ensure the docstring states **what the returned object means**, not just
its type (directly addresses "function description — when I return the data it means what").

---

## Testing strategy

`python -m pytest tests/` stays green (currently 48 pass / 1 skip; new tests add to this).
`python verify_results.py` passes at the new count. Notebooks run clean under **Restart & Run All**
on the `py312-tshci` kernel. Prose in `DATA_QUALITY.md` cites only numbers produced by the code.

## Results logging & verification (record the pass evidence)

After the work is done and verified, the actual pass output must be **recorded, not just claimed**:

1. **In the repo** — append a dated "Verification results" section to `report/DATA_QUALITY.md`
   containing the real captured output: the `pytest` summary line (e.g. `N passed, 1 skipped`) and
   the `verify_results.py` tail (`n/n checks passed`). Copy real terminal output, no paraphrase.
2. **In Notion** — create a new **`Daily Log — 3 July 2026`** entry under the `Msc_project` hub page
   (id `38639cc9-db44-81b3-a6bb-f1b0ea5c77cb`), following the existing daily-log pattern. It carries
   the same pass summary: date, pytest result, verify_results result, and a one-line status per
   feedback item A/B/C.
3. **Verify** — the numbers written to both places must come from a real run captured in this
   session, cross-checked against each other, before the work is declared complete
   (per verification-before-completion).

## Risks

- **Notebook re-runs** are the fragile/slow step — kept minimal; substantive description lives in
  `DATA_QUALITY.md` (plain markdown, no kernel).
- Column name `rho (g/m**3)` contains `**` — treat sensor names as opaque strings throughout.
