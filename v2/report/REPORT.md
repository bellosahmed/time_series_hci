# Time-Series Data Analysis Through a Natural HCI Interface — v2 Report

**Domain:** International trade (EU monthly exports & imports) — a supply-chain time series.
**Date:** 22 June 2026.

> **Core principle:** computation is done by deterministic code (the *truth layer*); the local
> language model only *explains* — it never calculates or invents a number.

---

## Abstract

This project builds a local, privacy-preserving, explainable system for analysing multivariate
time-series data and answering natural-language questions about it. Version 2 applies the system to
**EU international trade** (27 countries × exports/imports, monthly 2002–2026). Unlike v1 (weather
sensors with *synthetic* anomalies), this dataset contains **real, datable anomalies** — the 2008
financial crisis, the 2020 COVID collapse, and the 2022 energy crisis — enabling a genuine
evaluation. The pipeline covers preprocessing, behaviour analysis, anomaly detection (statistical,
forecasting, and machine-learning families), multi-country reasoning, forecasting with out-of-sample
validation, an exact query engine, hybrid retrieval (lexical + semantic), and a local-LLM
explanation layer. All numbers are computed by code and verified against pandas; the LLM only
phrases them. The full system runs offline and is covered by 60 automated tests.

---

## 1. Aim and objectives

**Aim.** Design a local, trustworthy time-series analytics system that detects anomalies, explains
behaviour, computes exact results, and answers user questions interactively.

**Objectives (from the project knowledge base):**
1. Analyse time-series behaviour in single- and multi-variable data.
2. Detect abnormal behaviour, drift, and event-like patterns.
3. Compute exact statistics (max/min/correlation/…).
4. Use clustering / grouping to reveal similar behaviour.
5. Support a natural-language (HCI) interface.
6. Use retrieval (RAG) for context only.
7. Keep the whole workflow local and privacy-preserving.

---

## 2. Dataset

**Source:** Eurostat `ei_eteu27_2020_m` (EU27 international trade, monthly), pulled via the public
API and stored locally as `data/eu_trade_monthly.csv`.

| Property | Value |
| --- | --- |
| Grain | each country's trade with the world |
| Shape | 292 months × 54 columns (27 countries × {exports, imports}) |
| Period | Jan 2002 – Apr 2026 (monthly) |
| Units | trade value, € millions, not seasonally adjusted |
| Missing | none (0 cells, 0 gaps, all values > 0) |
| Naming | `CC_EXP` / `CC_IMP` (see `GLOSSARY.md`) |

The data is the v2 analogue of v1's sensors: each country-flow is a "channel". It contains three
real shocks, which serve as ground-truth anomalies for evaluation. A complementary **bilateral**
dataset (`eu_trade_by_partner.csv`, EU↔partner annually) is kept as an illustrative example
(e.g. EU imports from Russia fell ≈ 78 % after the 2022 sanctions).

---

## 3. Methodology

The pipeline is implemented as a *hybrid*: reusable logic in `src/` (importable, unit-tested) is
demonstrated and explained in numbered Jupyter notebooks. Every analytical choice is justified by a
prior measurement (the EDA/analysis findings drive the modelling).

### 3.1 Preprocessing (`src/preprocess.py`, notebook 02)
Load → validate (clean: 0 missing/gaps/non-positive) → derive **balance** (EXP−IMP) and **total**
(EXP+IMP) → optional **log transform** (trade grows ~exponentially; logs make a proportional change
the same size in any year, so detectors treat 2008 and 2020 consistently).

### 3.2 Behaviour analysis (`src/analysis.py`, notebook 03)
Per-country trend (log-slope ≈ growth rate); **STL decomposition** (period = 12) with trend/seasonal
strength; rolling statistics (volatility); **ADF** stationarity test (level non-stationary p≈0.81 →
seasonally-differenced stationary p≈0.001, prescribing ARIMA differencing); ACF/PACF; residual
normality (fat tails = the crises); cross-country correlation and lead-lag.

### 3.3 Anomaly detection (`src/anomaly.py`, notebook 04)
Three families chosen because they fail differently:
1. **Statistical** — z-score on the STL residual.
2. **Forecasting** — z-score on the seasonal-naive error (vs the same month last year).
3. **Machine-learning (multivariate)** — Isolation Forest over all 54 channels' year-on-year growth,
   flagging *systemic* months when many countries move together.

### 3.4 Multi-country reasoning (`src/reasoning.py`, notebook 05)
Classifies a flagged month as a **shared shock** (many countries, same direction) or an **isolated**
single-country event, with the evidence (counts, z-scores, contributors). Output is *trust-aware*:
computed facts (🔢) are separated from inferred cause (🧠).

### 3.5 Forecasting (`src/forecasting.py`, notebook 06)
SARIMA (1,1,1)(1,1,1)₁₂ with a 95 % prediction band. Validated by **temporal backtest**
(train on the past, predict unseen future) — the test that distinguishes genuine prediction from
memorisation. A second backtest *across* COVID demonstrates that shocks are *detected* (band breach),
not predicted.

### 3.6 Query engine (`src/query_engine.py`, notebook 07)
The **truth layer**: exact max/min, mean/std, correlation, balance, surplus/deficit, growth, worst
drop, ranking, country summary. Every function is unit-tested **equal to pandas**.

### 3.7 Interface + LLM (`src/interface.py`, `src/llm.py`, notebook 08)
Natural-language question → intent + country/flow/date resolution → query engine → trust-aware
answer. A local Ollama model optionally *phrases* the computed facts (a strict system prompt forbids
changing numbers); if Ollama is down, templates are used (fully offline).

### 3.8 Hybrid retrieval (`src/rag_system.py`, notebooks 09 & 13)
**BM25** (lexical) + a semantic retriever, min-max normalised and combined. Two semantic backends are
provided behind one `search()` interface: **LSA** (TF-IDF→SVD, offline) and **BERT**
(`sentence-transformers`, neural embeddings). A sentence-aware **chunker** readies the system for long
documents. On paraphrase queries (no shared words) only BERT retrieves the right document, confirming
the value of neural embeddings on harder queries. Retrieval returns *context* only — the knowledge
base contains no numbers.

### 3.9 Clustering (`src/clustering.py`, notebook 11)
Countries are summarised by five interpretable features (size, surplus/deficit ratio, growth,
seasonality, volatility) and grouped with K-means (Objective 4). At k=4 the groups are interpretable:
Eastern catch-up economies, established Western economies, and small/outlier states (Cyprus; Greece/
Croatia/Luxembourg/Malta).

### 3.10 Deep model (`src/deep_anomaly.py`, notebook 12)
A small, GPU-trained multivariate **autoencoder** over the 54-channel growth vector; reconstruction
error scores systemic anomalies. Benchmarked against the Isolation Forest baseline (§4.5).

### 3.11 Web interface (`apps/web/`)
A Flask browser app: ask in plain English, see the trust-labelled answer (🔢 computed · 📚 retrieved ·
🧠 inferred · 💬 LLM). Numbers come from the query engine; RAG adds context; the LLM is optional.
Run `python3.12 apps/web/app.py` (auto-selects a free port).

---

## 4. Results

### 4.1 Detector performance (vs real 2008 + COVID labels)
Averaged across all 27 countries (strict month-by-month):

| Detector | Family | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| STL z-score | statistical | 0.46 | 0.20 | 0.27 |
| Seasonal-naive | forecasting | 0.49 | 0.15 | 0.22 |
| Isolation Forest | ML (multivariate) | 0.36 | 0.36 | **0.36** |

**Event-level detection (±1 month):** all three families catch **both** the 2008 and COVID crises.
(On Germany alone, seasonal-naive scores P=0.62, R=0.36, F1=0.45.)

**Point-adjusted (the TS-anomaly standard, notebook 14):** crediting a whole anomaly segment when any
point is flagged, F1 rises to **0.52 / 0.90 / 0.76** (STL / seasonal-naive / Isolation Forest). An
**ensemble** (flag if ≥2 detectors agree) reaches **P=0.82, R=1.00, F1=0.90** — perfect recall, zero
false negatives. The strict numbers above are honest but pessimistic; under the standard metric the
system detects the real crises strongly.

### 4.2 Forecasting (out-of-sample)
- Normal 24-month backtest (France trade): **MAPE 4.0 %**, 95 % band coverage 100 % → genuine
  prediction, not memorisation.
- Across-COVID backtest: MAPE 14.4 %, with 6 months breaching the lower band → shocks are detected,
  not foreseen.

### 4.3 Truth layer
8/8 verification checks confirm the query engine equals pandas exactly (max, min, mean, std,
correlation, balance, growth, surplus). 70 automated tests pass overall.

### 4.5 Deep model vs classical baseline (honest negative result)
The autoencoder, early-stopped at 60 epochs, scores **F1 ≈ 0.21** (catches COVID, misses 2008) — it
**does not beat** the Isolation Forest (F1 0.36). More training makes it worse (it overfits and
reconstructs the crises). Systemic shocks are *low-rank* (all countries fall together), which an
autoencoder reconstructs well, so they don't always register as high error; a tree-based detector
suits this better. **Conclusion:** on 292 points the deep model underperforms the classical baseline —
the expected, defensible outcome; deep learning needs more data to pay off. The classical Isolation
Forest remains the v2 default.

### 4.4 Example findings
Germany is the largest exporter and runs the biggest surplus; the Netherlands is in surplus 100 % of
months; France runs persistent deficits; Poland's exports grew ≈ +173–244 % (2010→2024); the worst
single-month export drops were Spain −35 % and Germany −31 %, both April 2020.

---

## 5. Evaluation discussion

- **Recall vs the wide window.** Strict recall looks low because the labelled crisis window spans
  ~14 months while detectors fire on the *sharpest* months. Event-level detection (the appropriate
  metric for rare events) is 100 %.
- **The precision artifact.** Many "false positives" are *real, unlabelled* anomalies (the 2009
  recovery tail, the 2021 rebound, the 2022 energy crisis). Measured precision is therefore a **lower
  bound**; cross-method agreement and corroboration against documented history show the detectors are
  better than the headline numbers.
- **Why three detector families.** They have different blind spots (z-score assumes near-normal
  residuals — but they are fat-tailed; seasonal-naive is blind to gradual drift; Isolation Forest
  needs the multivariate view). Agreement across families is a confident anomaly.

---

## 6. Limitations

1. **Short series.** 292 monthly points — strong for classical methods, too few for large deep
   models (Transformers/GNNs); these are documented as future work (`DEEP_MODELS.md`).
2. **Unpredictable shocks.** Forecasting predicts the *normal band*, never the crisis.
3. **Provisional data.** The most recent 1–2 months are revised by Eurostat.
4. **Trust-but-blunt thresholds.** The z-score rule is reasonable but crude given fat tails.
5. **Bilateral scope.** Individual country-pairs (e.g. France→Belgium) require Comext, which the
   public API does not serve; the EU↔partner dataset is used as the bilateral example.

---

## 7. v1 ↔ v2

| Aspect | v1 (Jena climate) | v2 (EU trade) |
| --- | --- | --- |
| Anomaly ground truth | synthetic spikes | **real** events (2008/COVID/2022) |
| Multivariate reasoning | per-sensor z-scores | systemic shared-shock detection |
| Forecasting | ARIMA residual | SARIMA + **backtest** validation |
| Retrieval | TF-IDF | **hybrid BM25 + LSA + BERT + chunking** |
| Clustering | day-types | **countries** by trade behaviour |
| Deep model | CNN-AE/LSTM (won) | autoencoder (honestly **loses** to baseline on 292 pts) |
| Interface | CLI + web | CLI logic + **web app** (trust-labelled) |
| Truth layer | query engine | query engine (8/8 vs pandas) |
| Tests | ~24 | **70** |

---

## 8. Conclusion and future work

v2 demonstrates the full architecture on a real-world supply-chain dataset with genuine anomalies:
detection that catches documented crises, a forecaster validated out-of-sample, an exact and
verified query engine, hybrid retrieval, and a local LLM that explains without ever altering a
number. The system is offline, modular, tested, and honest about its limitations.

**Future work:** more data to let deep models pay off (weekly/daily trade, more series); scaling the
RAG knowledge base (where BERT + chunking, now implemented, earn their keep over BM25/LSA); bilateral
(Comext) trade flows; and a 1D CNN-AE variant.

_Supporting documents: `EVALUATION.md`, `GLOSSARY.md`, `DEEP_MODELS.md`, `README.md`, notebooks 01–12,
`apps/web/`._
