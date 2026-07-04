# Work Log — time_series_hci

A running log of what we built, tried, and decided. Newest entries at the top.
(Complements `v1/EVALUATION.md`, `v1/MODELS.md`, and `CLAUDE.md`.)

---

## 2026-07-02

### v1 tidy-up: dedupe deep modules + robust test bootstrap
- **Scope:** best-practices pass on **v1 only** (arrange / combine / document) — no behaviour change.
- **Combine (dedupe deep detectors):** the CNN- and LSTM-autoencoder scorers each carried their own
  copy of (a) the z-standardisation step and (b) the overlapping-window error-averaging loop. Both
  are now single shared helpers in `deep_anomaly.py` — `standardise()` and
  `average_overlapping_windows()` — imported by `deep_lstm.py`. Removed `deep_lstm._standardise` and
  two duplicated averaging loops.
- **Arrange:** added `v1/conftest.py` so the suite resolves `from src import ...` from any working
  directory (previously only worked as `python -m pytest` run from inside `v1/`; a bare `pytest` or a
  run from the repo root failed to import `src`).
- **Verify:** **48 passed, 1 skipped**; `verify_results.py` **17/17**. Confirmed the same results
  when running the suite from the repository root (the conftest fix).

### v1 made newcomer-readable: merge deep modules + rewrite README
- **Goal:** make v1 easy for a newcomer (and me) to understand — clear where to start, grouped,
  documented — without changing behaviour.
- **Combine:** merged `deep_anomaly.py` + `deep_lstm.py` into a single **`src/deep.py`** ("all deep
  detectors in one place": CNN-AE, LSTM-AE, LSTM-forecaster). Public API unchanged. Merged the two
  test files into `tests/test_deep.py`. Updated the notebook imports via aliasing
  (`deep_anomaly as da` → `deep as da`, `deep_lstm as dl` → `deep as dl`) so every existing `da.*` /
  `dl.*` call still resolves — no in-cell code changes and no re-run needed. src modules: 14 → 13.
- **Document:** rewrote `v1/README.md` as a guided onboarding doc — plain "what it does", a numbered
  **reading order** (README → walkthrough notebook → query_engine → interface → anomaly/reasoning), a
  **layers diagram** grouping the 13 modules into Data → Detect → Truth → Explain → Interface, and
  copy-paste run/test commands. Updated `CLAUDE.md` module table + test list to match `deep.py`.
- **Verify:** **48 passed, 1 skipped**; `verify_results.py` **17/17** (unchanged).

### v2 made newcomer-readable: declutter root + rewrite README
- **Goal:** same treatment as v1 — arrange / clean / document — no behaviour change. (v2 has a single
  deep module, so there was no deep-module merge to do here.)
- **Arrange (declutter root):** moved the 16 developer scripts off the top level into **`v2/tools/`** —
  14 `build_*_notebook.py` (notebook generators) + `get_trade_data.py` + `get_bilateral_demo.py`
  (Eurostat fetchers). Root now holds only `verify_results.py` beside the real structure
  (`src/ tests/ notebooks/ apps/ report/`). Fixed each script's output path to survive the move:
  builders now write to `Path(__file__).parents[1]/"notebooks"`, fetchers to
  `HERE.parent.parent/"data"`. Verified the builders resolve to `v2/notebooks/*.ipynb` via a
  write-stubbed dry run (executed notebooks left untouched) and the fetchers to `code/data/`.
- **Arrange:** added `v2/conftest.py` (same `src`-on-path fix as v1).
- **Document:** rewrote `v2/README.md` in the same onboarding style — "what it does", a numbered
  **reading order**, the **layers diagram** (Data → Detect → Truth → Explain → Interface) over the 12
  `src/` modules, corrected run commands (`tools/get_trade_data.py`, web app) and the test count
  (**79**). Updated `CLAUDE.md` + `GLOSSARY.md` references to `tools/get_trade_data.py`.
- **Verify:** **79 passed**; `verify_results.py` **13/13**; identical when run from the repo root
  (the conftest fix); executed notebooks unchanged.

## 2026-06-28

### v1 review, fixes, and notebook consolidation (9 → 6)
- **Scope:** this case study uses **v1 only** (temperature / Jena climate). Reviewed the whole v1
  codebase, fixed issues, and consolidated the notebooks.
- **Bugs / improvements (src):**
  - `reasoning.co_movement` hardcoded the word "temperature" in its output — wrong label when the
    queried sensor was anything else. Now uses the actual `target` sensor name.
  - `interface.detect_intent` matched intent keywords as **substrings** ("most" inside "almost",
    "top" inside "stop"). Switched to **word-boundary** regex matching (consistent with v2).
  - Added `tests/test_reasoning.py` (2 tests) + 2 word-boundary asserts to `test_interface`.
    **Tests: 23 → 25 passing** (1 skipped when Ollama is up).
- **Notebooks consolidated 9 → 5** (re-executed clean on the `py312-tshci` GPU kernel):
  - old `01_data_exploration` + `04_visualizations` → **`01_data_exploration_and_visualization`** (Part 1 EDA, Part 2 visual gallery)
  - old `02_evaluation` + `03_methods_demo` → **`02_methods_and_evaluation`** (Part 1 methods, Part 2 correctness)
  - old `06_deep_model` + `07_deep_comparison` + `08_model_comparison` → **`04_deep_models_comparison`** (Parts A/B/C)
  - old `05_interface_demo` → **`03_interface_reasoning`**
  - final order: 00 walkthrough · 01 EDA+viz · 02 methods+eval · 03 interface · 04 deep models
- **Kernel:** registered `py312-tshci` (→ `/usr/bin/python3.12`, has torch+CUDA) because the old
  `python3` kernel pointed at a missing `python` and `tshci` is the torch-less 3.14 venv.
- **Docs synced:** `notebooks/README.md`, `v1/README.md`, `CLAUDE.md` (notebook table + test counts).

### Interface: time-scoped + threshold questions (fixes "always 2015")
- **Problem reported:** every "hottest/coldest" question returned the all-time record (37.04 °C,
  2015-07-04) because the interface always called `get_max`/`get_min` over the **whole** dataset —
  it never read a year or a value from the question.
- **Fix (`src/interface.py` + `query_engine.py`):**
  - `resolve_period()` — parses a year (`2013`), month+year (`july 2015`), or range
    (`between 2012 and 2014`); `ask()` slices the dataframe to it before computing.
  - `resolve_threshold()` + `query_engine.threshold_events()` — answers "how often above 30?",
    "colder than -15 in 2012?" with exact counts + first/last occurrence.
  - Every answer now states the period it used (`max … in 2013 was …`).
- **Verified on real data:** hottest overall 37.04 (2015); in 2013 → 34.93; coldest 2010 → -17.71;
  Jan 2012 → 11.97; above 35 °C → 18 times; below -15 in 2012 → 92 times.
- **Tests:** +4 (`resolve_period`, `resolve_threshold`, year-scoped ask, threshold count) → **29 pass,
  1 skip.** Web app suggestion chips updated; notebook 03 re-executed.

### New docs + verification recheck
- `v1/report/HCI_USER_STUDY.md` — full **user-study protocol** (within-subjects: spreadsheet vs
  template vs AI; task success graded against ground truth; time, SUS, NASA-TLX, trust).
- `v1/report/WRITEUP.md` — dissertation **draft skeleton** (structure + measured evidence embedded;
  lit review/prose marked TODO — for later, not for marking yet).
- **Verification rechecked twice:** test suite 48 pass / 1 skip; `verify_results.py` **17/17** on both
  runs (exit 0); independent fresh-pandas spot-check matches the system exactly (max 37.04 @ 2015-07-04
  16:00, min −22.65, mean 9.4447, >35 °C = 18).

### Evaluation rigor (examiner-driven) + de-emphasised "novelty"
- **Forecast backtest** (`forecasting.backtest`, rolling-origin, MAE/RMSE/**MASE**): on T, MAE 3.55 °C,
  RMSE 4.28 °C, **MASE 0.83** (beats persistence). MAPE avoided (temp crosses 0 °C).
- **Parameter sensitivity** (`anomaly.threshold_sensitivity` / `contamination_sensitivity`):
  z-threshold 4.0 is the knee (526 flags, ~0.75%); contamination 0.01 conservative — choices justified.
- **LLM faithfulness** (`llm.faithfulness`, rounding-tolerant): measures dropped vs **invented**
  numbers. Live result: **0 invented** (no hallucinated figures — trust property holds); occasional
  benign omission when summarising multi-value facts.
- **Repeatable anomaly benchmark** (`evaluation.benchmark`, 10 seeds): strict P 0.77±0.13, R 0.66±0.07,
  F1 0.71; point-adjusted similar — mean±std, not a single run.
- **`verify_results.py` now 17/17** (added forecast-MASE, benchmark recall, LLM-no-invented checks).
- **EVALUATION.md** gained a "Quantitative rigor" section with the measured numbers.
- **De-emphasised novelty** (per user): softened the three "novelty" mentions to "cross-sensor
  explanation feature"; saved a memory that the core objective is the evaluated explainable trust
  system + HCI, not novelty.
- **Tests:** +5 (backtest, threshold/contamination sensitivity, faithfulness, benchmark) → **48 pass, 1 skip.**

### More: forecast, reasoning confidence, web charts, v1 verification
- **Forecast intent** — `forecasting.forecast()` fits ARIMA on the last year of daily means and
  returns a 14-day forecast with an 80% band; interface answers "what will it be next week?".
- **Reasoning confidence** — `describe_moment` now adds a verdict: many sensors moving together →
  "real weather event"; one alone → "possible sensor glitch".
- **Web charts** — `Answer.chart` carries a downsampled series (+ markers / forecast band); the web
  UI renders an inline SVG (anomaly markers, the extreme point, or history+forecast).
- **`verify_results.py` for v1** — **14 independent checks** (query engine == pandas, scoped queries,
  thresholds, known-event placement, reasoning verdict, forecast bounds, point-adjusted ≥ strict
  recall, confusion sums to N). All pass.
- **EVALUATION.md** — added an honest "Ground truth & honesty" section: exact stats are hard truth
  (== pandas); anomaly accuracy is on synthetic labels + corroborated against known real events
  (Jul 2015 heatwave, winter cold extremes).
- **Tests:** +3 (forecast shape, forecast intent, reasoning verdict) → **43 pass, 1 skip.**

### Capability + UX upgrades (reasoning novelty, trend, compare, cleaner web UI)
- **Wired the multi-sensor reasoning (the project's novelty) into the interface.** Anomaly answers
  now *explain why* via `reasoning.describe_moment` — e.g. "Tdew low (-2.9σ); T low (-2.7σ); rho high
  (+2.5σ)". It was built but previously only used in a notebook, not the NL interface.
- **New `trend` intent** — linear warming/cooling rate (e.g. "T (degC) rising at +0.40 per year").
- **New `compare` intent** — two-period questions ("was 2013 hotter than 2014?"): exact stat per year
  + which was warmer/colder. `resolve_compare()` fires on 2 years + a comparison word.
- **Richer weather report** — added dew point.
- **Robustness:** unknown-sensor questions now say "I don't track X" instead of defaulting to
  temperature; the day-climatology label clamps to years actually present (no more spurious "2017");
  `anomaly.anomaly_windows` now delegates to `query_engine.get_anomaly_windows` (one implementation).
- **Evaluation honesty:** added `evaluation.point_adjust` + `score_point_adjusted` (the fairer
  time-series anomaly metric, mirroring v2).
- **Cleaner web UI** (`apps/web/index.html`): light, high-contrast, larger type; friendly trust labels
  (📊 exact figures · ℹ️ background · 🧠 what it suggests) + a one-line trust banner; no duplicated
  facts in template vs AI mode; refreshed suggestion chips.
- **Tests:** +8 (compare, trend, missing-sensor, anomaly-reasoning, point-adjust, mocked `llm.plan`
  ×2) → **40 pass, 1 skip.** Notebook 03 re-executed.
- **Friendly dates:** added `query_engine.fmt_when` ("19 Aug 2012 13:00") used across answers and the
  reasoning text instead of raw ISO timestamps.

### Interface: specific-day "weather report" answers
- **Ask:** questions like "tell me the weather in 26 january" should give a daily report.
- **Added:** `resolve_day()` parses a specific date — "26 january", "january 26", "26th of jan 2015",
  "26/01/2015", "2015-01-26"; year optional. New `weather` intent + `query_engine.weather_report()`
  give a multi-sensor snapshot (temp avg/low/high, humidity, pressure, wind).
- **No year → climatology:** "26 January" with no year reports across all years (e.g. 192 readings);
  with a year it's that exact 24-hour day. Stats also scope to a day ("hottest on 26 January 2015").
- **Verified:** 26 Jan 2015 → avg 1.1 °C (low -0.5, high 2.7), humidity ~93%; 5 Jul 2014 → avg 22.6 °C.
  **+2 tests → 32 pass, 1 skip.** Web chips + notebook 03 updated.

### Interface: LLM-as-parser so it isn't limited to a keyword list
- **Ask:** make it understand *any* phrasing, not just fixed keywords.
- **Design (trust-preserving):** the local LLM (`llm.plan`) translates free text into a **structured
  plan** {intent, sensors, start/end dates, op, value} — it interprets language (seasons, ranges,
  loose phrasing) but **never computes a number**; the query engine still does all the maths.
- **Merge rule:** deterministic parser wins when confident (reliable keyword/explicit-year matches);
  the LLM only *fills gaps* it left open. Prevents the small 3B model from flipping a good parse
  (e.g. "typical"→mean becoming max).
- **Seasons** added to the offline resolver too: spring/summer/autumn/fall, and winter rolls back to
  the previous December (`winter 2013` = Dec 2012–Feb 2013).
- **Verified (llama3.2:3b):** "how warm did it get during the summer of 2012?" → max 35.52 (summer
  2012); "typical temperature toward the end of 2014" → mean 10.70 (2014); "brutally cold in early
  2010" → anomaly scan in 2010. **30 tests pass, 1 skip.**

### CUDA out-of-memory fix in notebook 04 (deep models)
- **Symptom:** notebook 04 OOM'd on the 8 GB GPU when run interactively.
- **Cause:** the scoring step pushed the *entire* ~70k-window tensor through the model in **one
  forward pass**; the merged notebook also trains 3 models per session without freeing GPU memory,
  so stale kernels left ~4.5 GB held and the big single allocation no longer fit.
- **Fix (src):** batched inference — `deep_anomaly._squared_errors` and a new
  `deep_lstm._predict_batched` run the forward pass in mini-batches (8192 / 4096), so peak GPU
  memory is bounded regardless of series length. Added `deep_anomaly._free_cuda()` and call it after
  each model's scoring to return memory to the driver.
- **Notebook:** setup cell now clears the cache and prints GPU headroom on start.
- **Result:** runs clean in ~35 s with only **3.2 GB free**; CNN-AE scores 70,129 points in 1.4 s.
  Deep tests (5) + full suite (25 pass, 1 skip) green.

---

## 2026-06-22

### v2 kick-off — new domain dataset: international trade (supply chain)
- **Objective today:** start v2 by getting a richer, supply-chain-flavoured time series — the user
  wanted **trade between countries** (not the Jena climate data). v2 is a fresh iteration and does
  **not** have to mirror v1's structure.
- **Source survey (offline-friendly):** no Kaggle (no CLI/creds). Probed UN Comtrade (free API now
  deprecated → web page), IMF DOTS (host unreachable), World Bank (OK, annual only), **Eurostat (OK,
  monthly)**. Picked Eurostat for monthly resolution + genuine country-to-country trade.
- **Dataset chosen:** Eurostat `ei_eteu27_2020_m` — *EU27 international trade, monthly, 2002–2026*.
  Pulled raw trade **value (million EUR, not seasonally adjusted)**, total goods, partner = WORLD,
  so seasonality and real shocks survive.
- **Built:** `code/v2/get_trade_data.py` — reproducible fetch via Eurostat statistics/1.0 JSON API
  (the SDMX endpoint ignored query filters; statistics/1.0 filters correctly). Each country's monthly
  EXPORTS and IMPORTS = one column ("channel"), analogous to a sensor in v1.
- **Result:** `code/data/eu_trade_monthly.csv` — **292 months × 54 columns** (27 EU countries ×
  exports/imports), 2002-01 → 2026-04, **0 missing cells**, 114 KB (gitignored like Jena).
- **Why it's good for the project:** real, *explainable* anomalies are baked in — **2008 crisis**
  (DE exports €88.6bn Oct'08 → €63.5bn Jan'09) and **COVID** (€109bn Mar'20 → €75.9bn Apr'20, ~30%
  drop). Great for Objective 2 (drift / event detection) and the explanation layer.
### v2 EDA (notebook 01) — done
- Structure decided: **hybrid** v2 — analysis steps as notebooks, engine/interface/llm as importable
  `src` (so the website + pytest still work). Scaffolded `v2/{src,notebooks,tests}`.
- Built `v2/notebooks/01_eda.ipynb` (authored via `build_eda_notebook.py`, executed clean, 0 errors).
  **Documented for readers:** 13 markdown cells (What/Why/How-to-read per step) + line-commented code,
  and **8 plots** — value distribution (raw vs log), total trade with balance shading, big-traders bar,
  full STL 4-panel, month boxplot, COVID small-multiples, correlation heatmap + EXP-vs-IMP scatter, and a
  residual z-score anomaly preview (flags 2008-04, COVID 2020-04/05/06, 2022-09, 2024-03).
  Added `v2/GLOSSARY.md` + a v2 section on the Notion Glossary page.
- **Findings:** 292 months × 54 cols, 0 missing, no index gaps, all values > 0 → minimal cleaning.
  Total EU exports grew **3.1×** (2002→2026), roughly exponential → **log-transform** before anomaly
  detection. Clear yearly seasonality (summer/Dec troughs), DE export seasonal swing ~€32.6 bn →
  **STL period = 12**. Real shocks confirmed: 2008 **−28.3%**, COVID **−30.7%** (built-in ground truth).
  Countries co-move strongly (mean export-growth corr **0.57**; top pairs ES–IT, AT–DE, CZ–HU/PL ≈ 0.91).
- Tooling note: fixed a broken system `rpds`/`jsonschema` by installing newer ones into the py3.12
  user site so `nbformat`/`nbconvert` import. Notebook kernel = `python3`.
### v2 bilateral DEMO — EU trade by partner country
- To explain *country-to-country* (bilateral) trade, pulled Eurostat `ext_lt_maineu` (Extra-EU trade
  by partner, annual 2002-2025) → `code/data/eu_trade_by_partner.csv` (24 years × 22 cols = 11 major
  partners × EXP/IMP). Script: `v2/get_bilateral_demo.py`.
- Caveat: reporter = EU as a whole, breakdown by partner country (EU→USA, EU→China, EU→Russia…).
  True individual pairs (FR→BE) need Comext `DS-018995`, which the public API doesn't serve (404);
  intra-EU members can't be "partners" in extra-EU datasets.
- Real, explainable bilateral events: **EU imports from Russia −78%** (€163.6bn 2021 → €36.3bn 2024,
  sanctions); EU exports to USA grew €246bn→€536bn; China/UK trends visible. (This is a teaching
  example; the main v2 dataset remains `eu_trade_monthly.csv`.)
### v2 Preprocess (notebook 02 + src/preprocess.py + tests) — done
- Hybrid in action: logic in **`v2/src/preprocess.py`** (load_trade, validate→QualityReport, countries,
  flow_columns, trade_balance, total_trade, add_features, log_transform, one-call `preprocess()`),
  demonstrated in **`v2/notebooks/02_preprocess.ipynb`** (12 cells, 2 plots, 0 errors), covered by
  **`v2/tests/test_preprocess.py`** (8 tests, all pass).
- Implements the EDA decisions: validate says **CLEAN ✓** (0/0/0); derived **balance** (FR deficit /
  BE surplus) + **total**; **log transform** makes crises comparable across years (FR_EXP change-std:
  raw 4,608 EUR m vs log 0.131 = scale-free).
- Saved `data/eu_trade_features.csv` (108 cols = 54 flows + 27 BAL + 27 TOT) for downstream notebooks.
### v2 Behaviour analysis (notebook 03 + src/analysis.py + tests) — done
- `v2/src/analysis.py`: trend_slope(s)/trend_slopes, STL decompose + trend/seasonal strength
  (Hyndman), rolling_stats, adf_test→ADFResult, acf_pacf, correlations(growth), top_pairs,
  lead_lag, coverage. Notebook `03_analysis.ipynb` (18 cells, 6 plots, 0 errors). Tests:
  `test_analysis.py` (8) → **16 tests total pass**.
- Findings: fastest export growth = CY 9.9, LV 8.9, PL 8.7 %/yr (catch-up economies); slowest =
  LU 0.4, FI 1.7, MT 1.9. STL strength FR trend 0.93 / seasonal 0.67, BE 0.96 / 0.44. **ADF:** log
  level non-stationary (p=0.81) → seasonal-diff stationary (p=0.0013) ⇒ ARIMA needs differencing
  (d≥1, D≥1). STL residuals have **fat tails** (skew −3.18, kurt 28) — driven by the crises, so the
  |z|>3 rule is usable and the tails ARE the anomalies. Co-movement high (ES-IT, AT-DE, CZ-HU/PL
  ≈0.91); DE vs PL lead-lag = 0 (move together).
- Folded in the two preprocessing caveats: full coverage (27/27 countries, n=292) and a note that
  the last 1-2 months are **provisional** (Eurostat revises them).
### v2 Anomaly detection (notebook 04 + src/anomaly.py + tests) — done
- `v2/src/anomaly.py`: three detector families — (1) **statistical** z-score on STL residual,
  (2) **forecasting** seasonal-naive error z-score, (3) **ML multivariate** Isolation Forest over
  all 54 channels' year-on-year growth (systemic-shock detector). Plus eval helpers: label_windows,
  Scores, score() (P/R/F1/confusion). Notebook `04_anomaly.ipynb` (16 cells, 4 plots, 0 errors).
  Tests `test_anomaly.py` (6) → **22 tests total pass**.
- Evaluated vs **real labelled crises** (2008-09→2009-04, 2020-03→2020-08; 14 months). Strict
  month-by-month: seasonal-naive best (P=0.62 R=0.36 F1=0.45), STL-z (0.50/0.21/0.30), IsoForest
  (0.36/0.36/0.36). Low recall is a labelling artifact (wide window vs detectors firing on the
  sharpest months) — so added **event-level detection**: all three catch **both** 2008 & COVID (±1mo).
- Honest notes captured: **precision artifact** (real-but-unlabelled 2022 energy shock & 2021 rebounds
  count as FPs → measured precision is a lower bound); three families fail differently (z-score assumes
  near-normal, seasonal-naive blind to drift, IsoForest needs multivariate view) → agreement = confidence.
### v2 Multi-country reasoning (notebook 05 + src/reasoning.py + tests) — done
- `v2/src/reasoning.py`: cross_section_z (months×27 standardised growth), systemic_timeline,
  ShockReport, shared_shock/explain_month (classifies shared-shock vs isolated vs normal +
  evidence + sentence), find_isolated_months, co_movement. Notebook `05_reasoning.ipynb`
  (14 cells, 4 plots, 0 errors). Tests `test_reasoning.py` (6) → **28 tests total pass**.
- Demonstrated: **Apr-2020 COVID** = 18/27 countries down (67%) → "shared shock" (top FR/RO/IT/SK/ES
  ≈ −5σ); **Jan-2009 GFC** = 23/27 down (85%); **isolated** example Aug-2014 (only CY +2.2σ).
  57 isolated months found overall.
- **Trust-aware output** (the LLM bridge): emits 🔢 COMPUTED facts (counts, z-scores, fraction, kind)
  separately from 🧠 INFERRED cause — the model may rephrase computed facts but never invents them.
- Deep-learning model shortlist recorded (for the later deep step): USE multivariate Autoencoder
  (systemic) + 1D CNN-AE baseline; LSTM-AE/forecaster/TCN/VAE optional; Transformer/GNN/TFT
  literature-only (292 pts too few). Classical baselines must be beaten to justify deep.
  Saved as a **side note**: `v2/DEEP_MODELS.md` (full table + recommendation; not yet implemented).
### v2 Forecasting (notebook 06 + src/forecasting.py + tests) — done
- `v2/src/forecasting.py`: SARIMA (1,1,1)(1,1,1)12 — fit_sarima, forecast()→Forecast(mean/band),
  **backtest()** (train on past, predict UNSEEN future, MAPE/RMSE/coverage), backtest_from(cutoff).
  Notebook `06_forecasting.ipynb` (12 cells, 4 plots, 0 errors). Tests `test_forecasting.py` (5)
  → **33 tests total pass**.
- **Predict-vs-memorise answered with a backtest:** France trade, last 24 unseen months →
  **MAPE 4.0%, 100% band coverage** = genuine prediction (SARIMA's ~5 params can't memorise 292 pts).
- **Shock backtest across COVID** (train ≤2019-12 → forecast 2020-21): MAPE 14.4%, **6 months fall
  BELOW the 95% band** → proves shocks are *detected, not predicted* (the forward early-warning idea).
- Forecast to 2030: France €116.4 bn → central €127.6 bn (+10%), band €78-208 bn (widens with horizon).
  Projected fastest growers: CY 9.9, LV 8.9, PL 8.7, BG 8.3, LT 8.0 %/yr.
- Limitations stated: shocks unpredictable, band widens, structural breaks, provisional tail.
- ANSWER to "when do we train models": SARIMA trained+validated here; deep models train later (deep
  notebook) with the same backtest discipline (small/regularised to avoid memorising). See DEEP_MODELS.md.
### v2 Query engine (notebook 07 + src/query_engine.py + tests) — done
- `v2/src/query_engine.py` — THE TRUTH LAYER: get_max/get_min (Extremum), get_mean_std,
  get_correlation, trade_balance, total_trade, biggest_surplus/deficit, growth, worst_drop,
  rank_countries, country_summary. Notebook `07_query_engine.ipynb` (16 cells, 0 errors).
  Tests `test_query_engine.py` (10) → **43 tests total pass**.
- **Verification: 8/8 checks confirm engine == pandas** (exact). Sample answers: FR deficit
  −7.5 bn (Mar-2024), DE biggest surplus +21.8 bn; PL exports +244% (2010→2024); worst monthly
  export drops ES −35% / DE −30.7% (both Apr-2020 COVID); top exporters DE>NL>IT>FR>BE; NL in
  surplus 100% of months.
- This is the foundation for the trust principle: code computes exact numbers; the interface/LLM
  (next) only phrase them.
### v2 Interface + local LLM (notebook 08 + src/interface.py + src/llm.py + tests) — done
- `v2/src/interface.py`: NL question → detect_intent (word-boundary match) + resolve country/flow/
  dates → `query_engine` → trust-aware `Answer` (🔢 computed / 🧠 inferred / 💬 LLM). Intents: max,
  min, mean, correlation, surplus/deficit/balance, growth, drop, rank, summary, shock(→reasoning).
  `src/llm.py`: stdlib-HTTP Ollama, is_available/list_models/pick_model/generate/explain; strict
  system prompt forbids changing numbers; graceful fallback to templates if Ollama down.
- Notebook `08_interface.ipynb` (14 cells, 0 errors). Tests `test_interface.py` (10) → **53 tests pass**.
- **Live LLM verified** (Ollama up; picked `llama3.2:3b`): it phrases the computed facts and keeps
  numbers EXACT (e.g. "Poland's exports … growing by 173% … Dec-2010 to Dec-2024"; Spain "-35% …
  peak of COVID-19") — adds context but never alters a figure. Trust principle proven end-to-end.
- **Bug fixed:** growth keyword "rise" matched inside "summa**rise**" (same class as v1 usual/unusual)
  → switched intent matching to word boundaries; added regression test.
- Sample answers: FR peak export 57,238 (2025-10); DE-FR export corr 0.79; ES worst drop -35% (2020-04);
  top exporters DE>NL>FR>IT>BE; NL in surplus 100% of months.
### v2 Hybrid RAG (notebook 09 + src/rag_system.py + tests) — done
- User chose **hybrid (lexical + semantic)**. Findings before building: `rank_bm25` not installed and
  the installed Ollama models **can't embed** (501 Not Implemented). So built fully offline, no new
  deps: **BM25 from scratch** (lexical) + **LSA** = TF-IDF→TruncatedSVD (semantic, sklearn). Same
  `search()` interface so neural embeddings (pull `nomic-embed-text`) can swap in later.
- `v2/src/rag_system.py`: knowledge_base() (20 context docs — countries, trade terms, 2008/COVID/2022
  events, methods; **no euro amounts**), BM25, SemanticLSA, HybridRetriever (min-max normalise +
  weighted combine + re-rank), explain() (🔢/📚/🧠). Notebook `09_rag.ipynb` (12 cells, 0 errors).
  Tests `test_rag.py` (7) → **60 tests total pass**.
- Verified semantic match: "pandemic lockdown halted commerce" → event_covid (0.97); "war disrupted
  energy and trade" → event_2022 (0.98) — meaning matched without literal keywords. Full trust-aware
  answer assembled: 🔢 ES -35% (2020-04, from query engine) + 📚 COVID/2008 context + 🧠 inference.
- Trust boundary kept: RAG supplies CONTEXT only; numbers always come from the query engine (KB has
  no euro amounts — unit-tested).
- Note on retrieval method (answer to user): v2 now uses **BM25 + LSA hybrid**; earlier interface used
  keyword routing only. Confusion-matrix confirmed (seasonal-naive DE: TP=5 FP=3 FN=9 TN=275, sums to
  292; FP=real-but-unlabelled 2009-tail/2021-rebound; FN=wide-window recovery tail).
### v2 Evaluation + report (notebook 10 + report/ + README) — done
- `v2/notebooks/10_evaluation.ipynb` (13 cells, 0 errors): detector comparison averaged over 27
  countries (STL-z F1 0.27, seasonal-naive 0.22, **IsoForest 0.36** — multivariate wins on average);
  event-level all three catch both crises; confusion-matrix heatmap; forecasting recap (normal MAPE
  **4.0%**/100% coverage, COVID MAPE 14.4%/6 breaches); v1↔v2 comparison table.
- Wrote **`v2/report/REPORT.md`** (full MSc draft: abstract, aim/objectives, dataset, methodology
  per step, results, evaluation discussion, limitations, v1↔v2, conclusion/future work) +
  **`v2/report/EVALUATION.md`** (all numbers) + **`v2/README.md`** (folder/module map + how to run).
- Note: user declined pulling `nomic-embed-text` (real neural embedded search) — LSA stays the
  semantic half; documented as future work. Ollama registry reachable + ollama CLI 0.30.7 if revisited.
- v2 status: **10 notebooks, 60 tests passing**, full pipeline (EDA→preprocess→analysis→anomaly→
  reasoning→forecasting→query engine→interface+LLM→hybrid RAG→evaluation) + report. Remaining
  optional: web app, deep models, clustering.

### v2 final batch — clustering, deep model, web UI, report finalised (user: "do 4 2 1 5")
- **#4 Clustering** (`src/clustering.py`, notebook 11, 6 tests): country_features (size, balance_ratio,
  growth, seasonality, volatility) → K-means. Silhouette best=2 (just isolates CY/MT outliers) so used
  **k=4** for interpretable groups: Eastern catch-up (BG/CZ/EE/LT/LV/PL/RO/SI/SK), Western established
  (AT/BE/DE/DK/ES/FI/FR/HU/IE/IT/NL/PT/SE), CY alone, EL/HR/LU/MT. Satisfies Objective 4.
- **#2 Deep model** (`src/deep_anomaly.py`, notebook 12, 4 tests, GPU cuda): multivariate autoencoder.
  **Honest negative result** — AE F1 **0.21** (catches COVID, misses 2008) **loses to** Isolation
  Forest 0.36. Overfits fast (more epochs → worse); early-stop at 60. Insight: systemic shocks are
  low-rank so an AE reconstructs them well → tree-based IsoForest better. Deep needs more data; IsoForest
  stays default. (DEEP_MODELS.md prediction confirmed.)
- **#1 Web UI** (`apps/web/app.py` + `index.html`): Flask chat — model selector, use-LLM toggle,
  suggestion chips, trust-labelled answers (🔢/📚/🧠/💬) incl. hybrid-RAG context. Auto free-port
  (3000→3001 when busy). Verified endpoints (/, /api/suggestions, /api/models, /api/ask).
- **#5 Report finalised**: REPORT.md (added clustering §3.9, deep §3.10/§4.5, web §3.11; v1↔v2 table;
  future work), EVALUATION.md (deep-vs-baseline table), README.md (12 notebooks, 70 tests, web app).
- **Status: 12 notebooks, 70 tests passing**, full pipeline + clustering + deep + web + report.
  User declined neural embedded search (nomic-embed-text) — kept as future work.

### v2 Chunking + BERT retrieval (notebook 13 + rag_system extensions + tests) — done
- User chose "both: chunker + BERT". Installed **sentence-transformers 5.6.0**; downloaded
  **all-MiniLM-L6-v2** (384-dim, ~90MB). HF Hub reachable.
- `src/rag_system.py` additions: **chunk_text()** (sentence-aware, overlapping), **BertRetriever**
  (neural embeddings; lazy-loads model; drop-in for SemanticLSA), and `HybridRetriever(semantic=
  "lsa"|"bert")`. Notebook `13_chunking_bert.ipynb` (12 cells, 0 errors). Tests +4 → **74 total pass**
  (BERT tests skip gracefully if lib absent via importorskip).
- **Key demo (BERT earns its keep):** paraphrase query "factories shut and shipping stopped worldwide"
  (no shared words) → BM25 & LSA return wrong country docs; **BERT returns event_covid + event_2008**.
  Chunking a long doc → 4 overlapping chunks; chunk-level BERT retrieval answers "how did governments
  react?" with the stimulus chunk.
- Honest note kept: for 20 atomic KB docs BM25/LSA already suffice; BERT + chunking pay off as the
  corpus grows (long docs, paraphrase). Retrieval still returns CONTEXT only.
- Docs updated: REPORT §3.8 + v1↔v2 table + future work, README (13 notebooks, BERT/chunking).
  New dep: sentence-transformers. **v2 now: 13 notebooks, 74 tests passing.**

### v2 results verification (verify_results.py + report/VERIFICATION.md) — done
- Built `v2/verify_results.py`: independently re-derives every headline claim from raw data and checks
  it (separate from unit tests — tests check the code, this checks the claims). **13/13 PASS.**
- Verified: data 292×54 clean; query engine == pandas (truth layer); ADF level→diff; all 3 detectors
  catch 2008 & COVID (event-level); confusion sums to 292 (TP5/FP3/FN9/TN275); forecast backtest MAPE
  **4.0%** (predicts not memorises); COVID breaches band; deep AE F1 0.21 ≤ IsoForest 0.36 (negative
  result holds); clustering 27→4; reasoning 2020-04 = shared shock; **RAG KB has no numbers**; **BERT
  finds COVID on paraphrase where BM25 fails**; **LLM keeps magnitude 35 + year 2020 exact**.
- Fixed an over-strict LLM check (was matching literal "-35.0"; LLM correctly wrote "35%"/"April 2020"
  — magnitude+date preserved, sign/decimal dropped = not a trust violation).
- Documented in `report/VERIFICATION.md` (one-command reproducible proof of the trust chain).

### v2 Evaluation upgrade (notebook 14 + src/evaluation.py + tests) — done
- Addresses the review's #1 weakness (low strict F1). `src/evaluation.py`: **point_adjust** (TS-anomaly
  standard — credit a whole anomaly segment if any point flagged), score_point_adjusted, **ensemble**
  (vote across detectors), **pr_curve** + pr_auc. Notebook `14_evaluation_upgrade.ipynb` (10 cells,
  0 errors). Tests `test_evaluation.py` (5) → **79 tests total pass**.
- **Big improvement:** point-adjusted F1 — STL 0.30→0.52, **seasonal-naive 0.45→0.90**, IsoForest
  0.36→0.76. **Ensemble (≥2 votes) point-adjusted: P=0.82, R=1.00, F1=0.90** (perfect recall, 0 FN).
  PR curves trace each detector's full threshold trade-off (PR-AUC).
- Takeaway: the earlier strict numbers were honest but pessimistic; under the standard metric the
  system detects the real 2008 & COVID crises strongly, and the ensemble is the best single choice.
- Fixed NumPy 2.x `trapz`→`trapezoid` in pr_auc. Docs updated: REPORT §4.1, EVALUATION.md (point-adj +
  ensemble table, 79 tests), README (14 notebooks, evaluation.py).

---

## 2026-06-21

### v1 restructured + documented
- Tidied the scattered v1 layout: `apps/` (chat.py + web/), `report/` (REPORT moved in from
  `code/report/`, plus EVALUATION/MODELS/FINETUNING), `src` / `tests` / `notebooks` unchanged.
- New docs: `v1/README.md` (folder map + how to run), `notebooks/README.md` (reading order),
  and **`notebooks/00_code_walkthrough.ipynb`** — tours every src module with runnable examples.
- Fixed path lines in moved `chat.py` / `web/app.py`; updated root README + CLAUDE.md.
- Verified: chat.py + web import OK, 23 tests pass, walkthrough executes.

### Web UI (Flask) — interactive chat in the browser
- Built `v1/web/app.py` (Flask) + `v1/web/index.html`: chat page with **model selector**,
  **Use-LLM toggle**, **clickable suggested questions**, and **trust-labelled answers**
  (computed/context/inferred shown).
- Reuses `interface`/`llm` directly; endpoints `/api/ask`, `/api/models`, `/api/suggestions`.
- **Port note:** the user wanted :3000, but a **FastAPI app already occupies :3000** — app.py
  auto-falls back to the next free port (currently **3030**). Run: `python3.12 v1/web/app.py [port]`.
- Verified all endpoints return correct answers; numbers stay exact.

### Interface: removed the keyword limitation
- **Problem:** the NL interface only understood fixed keywords (max/min/mean/correlation/missing/
  anomaly/summary).
- **Change:** expanded the keyword synonyms, and added an **LLM fallback** — when keywords don't
  match and `use_llm=True`, the local model classifies the question into an allowed intent + sensor
  (`llm.classify`), then the **query engine still computes the exact number**. Trust preserved.
- **Verified live:** "how warm did the air ever get?" → max → 37.04 °C; "is humidity connected to
  how warm it is?" → correlation(rh, T). 23 tests pass.
- **Bug found & fixed:** adding "usual" to the *mean* keywords matched "un**usual**" (anomaly) — a
  substring collision. Removed it.

### Ollama explanation layer (the final step) — implemented & verified
- `src/llm.py` talks to local Ollama (stdlib HTTP); strict prompt forbids changing numbers.
- `interface.Interface(use_llm=True)` phrases answers via the LLM, falls back to template if Ollama
  is down. Live demo kept numbers exact (37.04 °C, −0.963).
- Added `chat.py` runnable entry point (`python3.12 chat.py --llm`). `FINETUNING.md` = side note.
- Note: only the community Gemma models are installed; recommend `ollama pull llama3.2:3b` for a
  cleaner explainer.

### Evaluation: confusion matrix + precision clarified
- Added TP/TN/FP/FN (`evaluation.confusion`). Controlled signal: Isolation Forest perfect (40 TP,
  0 FP); z-score weakest (misses a few spikes STL absorbs).
- **Key finding:** real-data precision (~0.14) is an *artifact* — genuine background anomalies count
  as false positives against synthetic-only labels. Controlled-signal precision is 0.85–1.00.

### Models: full comparison + ARIMA
- Added ARIMA forecast-error detector (deseasonalised; ~3 s on 70k points).
- Six detectors compared (statistical / forecasting / classical / 3 deep) in `MODELS.md` and
  `notebook 08`. CNN-AE is the best simple default.

### GPU enabled
- Installed cu128 PyTorch for the RTX 5070 (Blackwell, sm_120). Deep models are device-aware.
- Verified: 21 tests pass on GPU; CPU/GPU equivalent; ~1.7× speedup (small models, so modest).

### Deep models
- CNN autoencoder, LSTM autoencoder, LSTM forecaster — unsupervised, per-point reconstruction/forecast
  error. All recall 1.00 on injected anomalies.

### Core pipeline (earlier in the day)
- preprocess (incl. −9999 sentinel cleaning) → analysis (STL) → anomaly (z-score, Isolation Forest)
  → clustering (K-means day-types) → **query engine** (exact, 6/6 vs pandas) → interface + RAG +
  reasoning. Dataset: Jena Climate 2009–2016 (70k hourly rows).
- Environment gotcha: Jupyter runs on **Python 3.12** (deps there), separate 3.14 venv.

---

## How to use this log
Add a dated entry whenever we build/try/decide something: what the problem was, what changed, what
the result was, and any bug or surprise. This feeds the report's methodology and discussion.
