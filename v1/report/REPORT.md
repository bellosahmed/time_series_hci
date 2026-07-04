# Time-Series Data Analysis Through a Natural HCI Interface

**MSc Project Report (Draft)**
_Author: Bello Ahmed · Loughborough University · Last updated: 4 July 2026_

> This is a working draft generated from the implemented system and its measured results.
> Placeholders marked **[ADD]** are for you to complete (exact citations, personal reflection,
> supervisor-specific framing). Numbers come from the actual runs in `v1/notebooks/`.

---

## Abstract

This project designs and implements a **local, privacy-preserving, explainable** time-series
analytics system that lets a user ask natural-language questions about multivariate sensor data and
receive **trustworthy** answers. Its defining principle is a strict separation between *computation*
and *explanation*: all exact values (maxima, minima, correlations, anomaly windows, missing-value
counts) are produced by a deterministic **query engine**, while a language model is reserved purely
for phrasing — it never computes numbers, and a lightweight **faithfulness check** verifies this
automatically, rejecting any explanation that introduces a figure absent from the computed facts.
We name this reusable pattern **Verifiable NL Analytics** (deterministic truth layer + LLM
explanation + automated faithfulness verification). The system is evaluated on the Jena Climate
dataset (2009–2016; 70,129 hourly readings across 14 sensors). Anomaly detection is studied across
six methods spanning statistical, forecasting, classical-machine-learning and deep-learning families.
The query engine is shown to match an independent `pandas` ground truth exactly; detector quality is
reported with bootstrap confidence intervals and a paired significance test (precision 0.85–1.00 on
a controlled signal); and the local LLM introduced **0** fabricated numbers across the verification
suite. The result is a fully offline pipeline that combines exact numerical querying, anomaly
detection, multi-sensor reasoning and a verified natural-language interface. The same architecture is
shown to transfer to a second, unrelated domain (EU trade; Appendix B).

---

## 1. Introduction

### 1.1 Background
Sensor systems produce continuous streams of time-indexed measurements (temperature, pressure,
humidity, etc.). Two problems make these data hard to use: (i) detecting and *understanding*
abnormal behaviour, and (ii) allowing non-experts to interrogate the data without writing code.
Large language models (LLMs) make natural-language interaction attractive, but they are unreliable
for arithmetic — they can fabricate plausible-looking numbers. This undermines trust in any
analytics answer they produce.

### 1.2 Problem statement
How can a system answer natural-language questions about time-series sensor data **correctly and
trustworthily**, while running entirely **locally** for privacy?

### 1.3 Aim
Design a local, trustworthy time-series analytics system that detects anomalies, explains behaviour,
computes exact results, and answers user questions interactively.

### 1.4 Objectives
1. Analyse single- and multi-sensor time-series behaviour.
2. Detect abnormal behaviour, missing data, drift and event-like patterns.
3. Compute exact statistics (max/min, correlations, counts).
4. Use clustering to identify groups of similar behaviour.
5. Support an HCI interface for user questioning.
6. Use retrieval (RAG) for contextual grounding only.
7. Keep the whole workflow local and privacy-preserving.

### 1.5 Research questions
1. Which statistical, ML and deep-learning methods best detect unusual behaviour in the dataset?
2. How useful are decomposition and residual analysis for anomaly detection?
3. Can clustering reveal normal vs abnormal behaviour groups?
4. Can a local interface answer questions reliably when exact values are computed by code, not a model?
5. How can trust and explainability be improved in local time-series analytics?

### 1.6 Contributions
- **Verifiable NL Analytics** — a reusable architecture that pairs a deterministic *truth layer*
  with an LLM used *only* for explanation, and adds an **automated faithfulness check** that verifies,
  at answer time, that the explanation invented no numbers. Numerical trust is thus guaranteed by
  construction *and* checked, not merely asserted.
- A comparative study of **six** anomaly-detection methods on a common dataset and harness, reported
  with **bootstrap confidence intervals** and a **paired significance test** (not bare point estimates).
- A **multi-sensor reasoning** component that explains anomalies using cross-sensor behaviour.
- A fully **offline** natural-language interface with trust-labelled answers, shown to **transfer to a
  second, unrelated domain** (EU trade, Appendix B) — evidence the method is general, not dataset-specific.

### 1.7 Report structure
Section 2 reviews related work; Section 3 sets out the methodology; Section 4 the implementation;
Section 5 the results and evaluation; Section 6 discusses them; Sections 7–8 cover limitations,
conclusions and future work.

---

## 2. Literature Review

**Themes.** (1) real-world multi-sensor monitoring datasets; (2) statistical behaviour modelling;
(3) anomaly detection (statistical, classical ML, deep learning); (4) evaluation challenges;
(5) explainability and interaction; (6) local / privacy-preserving design.

**Key methods and references** _(verify exact citations — **[ADD]**)_:
- Isolation Forest — Liu, Ting & Zhou (2008).
- STL decomposition — Cleveland et al. (1990).
- LSTM for anomaly detection — Malhotra et al. (2015).
- TCN / 1D CNN for sequences — Bai, Kolter & Koltun (2018).
- Deep anomaly detection survey — Pang et al. (2021).
- Benchmark critique — Wu & Keogh (2021).
- Retrieval-Augmented Generation — Lewis et al. (2020).
- Datasets — M100 ExaData (Borghesi et al.), Vigoya data-centre temperature, RUAD.

**Gap.** Many works *detect* anomalies, but few combine **exact numerical querying + natural-language
explanation + an HCI interface + local, privacy-preserving deployment**. This project targets that
intersection, and additionally separates computation from explanation to guarantee numerical trust.

---

## 3. Methodology

### 3.1 Dataset
Jena Climate (2009–2016): 420,551 readings at 10-minute intervals, 14 numeric sensors. The target
sensor is air temperature `T (degC)`. Any time-indexed dataset would fit the pipeline; Jena is chosen
because it is multivariate, strongly seasonal, and contains real data-quality issues.

### 3.2 Preprocessing
Timestamps are parsed to a datetime index; the wind columns' `-9999` sentinel error codes are
converted to NaN; missing values are interpolated; the series is resampled to a common **hourly**
frequency (70,129 rows), which preserves daily and seasonal structure while remaining tractable.

### 3.3 Behaviour analysis
Rolling mean/standard deviation, **STL decomposition** (trend + seasonal + residual), trend slope,
and cross-sensor correlations.

### 3.4 Anomaly detection (six methods)
| Family | Method |
| --- | --- |
| Statistical | Z-score on the STL residual |
| Forecasting | ARIMA forecast-error (deseasonalised) |
| Classical ML | Isolation Forest |
| Deep (reconstruction) | 1D CNN Autoencoder |
| Deep (reconstruction) | LSTM Autoencoder |
| Deep (forecasting) | LSTM Forecaster |

All deep models are unsupervised: they learn the *normal* pattern and flag points that are poorly
reconstructed or poorly predicted.

### 3.5 Clustering
Each calendar day is summarised into a feature vector (mean/std/min/max per sensor) and grouped with
**K-means** to reveal day-types.

### 3.6 Query engine (the truth layer)
Deterministic Python functions compute exact max/min (with timestamps), mean/std/range, Pearson
correlation, missing-value counts, anomaly windows, cluster membership and per-sensor summaries.
**The LLM never performs these computations.**

### 3.7 Interface, retrieval and reasoning
A deterministic interpreter maps a question's intent and target sensor to a query-engine call. A
local **TF-IDF retriever** supplies sensor context (RAG). Answers are **trust-labelled** as
*computed*, *retrieved* or *inferred*. A **multi-sensor reasoning** module explains an anomalous
moment using each sensor's z-score and its correlation/co-movement with the target.

### 3.8 Evaluation approach
(i) **Correctness** — every query-engine output is compared to an independent `pandas` computation.
(ii) **Detection quality** — anomalies have no ground-truth labels, so *synthetic* anomalies are
injected and precision/recall/F1 measured, on both the real series and a controlled clean signal.
(iii) **Offline operation** — confirmed by using only local libraries and data.

---

## 4. Implementation

### 4.1 Architecture
```
Data → Preprocess → Behaviour analysis → Anomaly detection → Clustering
                                   ↓
                            Query engine (exact, truth)
                                   ↓
                     RAG retriever (context) + Reasoning
                                   ↓
                    Natural-language interface (trust-aware)
                                   ↓
              Local LLM — explanation only, faithfulness-checked
```

### 4.2 Modules (`v1/src/`)
`preprocess` · `analysis` · `anomaly` · `clustering` · `query_engine` · `deep` (CNN-AE,
LSTM-AE, LSTM forecaster) · `forecasting` (ARIMA) · `reasoning` · `rag_system`
· `interface` · `main`.

### 4.3 Technology
Python 3.12; pandas, NumPy, SciPy, scikit-learn, statsmodels, matplotlib; PyTorch for deep models;
Flask for the web UI; an optional local Ollama model for phrasing. Dependencies are pinned
(`requirements.txt`). Everything runs locally. Deep models are **GPU-aware** (auto-select CUDA): verified on an NVIDIA RTX 5070
(Blackwell) with the cu128 PyTorch build; results are equivalent to CPU.

### 4.4 Testing
**87 automated tests** pass (1 skipped) (`v1/tests/`): query-engine correctness vs pandas, interface
routing (incl. circular wind-direction handling and the LLM trust-gate), the Flask/CLI app layer,
the client-side chart/legend JavaScript (run in Node), synthetic evaluation with confidence
intervals, and the deep / ARIMA detectors. An independent
harness (`verify_results.py`) additionally recomputes results a different way and passes **21/21**
checks (plus one advisory LLM-faithfulness vote); `reproduce.sh` regenerates everything in one command.

---

## 5. Results & Evaluation

### 5.1 Data characteristics
Temperature ranges from −22.7 °C (07 Jan 2009) to 37.0 °C (04 Jul 2015), mean 9.4 °C. It correlates
+0.997 with potential temperature and **−0.96 with air density** (cold air is denser) — physically
correct, confirming the pipeline handles the data soundly.

### 5.2 Query-engine correctness
On the full dataset, every exact query matched the independent `pandas` ground truth **exactly**
(values and timestamps): the `verify_results.py` harness passes **21/21** cross-checks — max, min,
mean and correlation equal pandas; scoped and threshold queries equal the corresponding pandas slice;
per-sensor availability and skew are recomputed and agree — and the unit suite (**87 tests**) covers
the same guarantees. This establishes the trust guarantee the architecture depends on.

### 5.3 Anomaly-detection comparison
| Model | Type | Flags | Time | Precision* | Recall* | F1* |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Z-score (residual) | Statistical | 526 | ~4.0 s | 0.24 | 0.45 | 0.32 |
| ARIMA forecast | Forecasting | 411 | ~2.7 s | 0.20 | 1.00 | 0.34 |
| Isolation Forest | Classical ML | 699 | ~0.3 s | 0.14 | 1.00 | 0.25 |
| CNN Autoencoder | Deep (recon) | 702 | ~2.5 s | 0.14 | 1.00 | 0.25 |
| LSTM Autoencoder | Deep (recon) | 702 | ~1.0 s | 0.14 | 1.00 | 0.25 |
| LSTM Forecaster | Deep (forecast) | 702 | ~2.5 s | 0.14 | 1.00 | 0.25 |

\* From injecting 100 synthetic anomalies into the **real** series. Precision is understated because
genuine background anomalies are flagged but count as false positives against synthetic-only labels.

**Fair precision (controlled clean signal, no hidden anomalies):**
| Model | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Z-score | 0.85 | 0.82 | 0.84 |
| ARIMA | 0.98 | 1.00 | 0.99 |
| Isolation Forest | 1.00 | 1.00 | 1.00 |
| CNN Autoencoder | 0.91 | 1.00 | 0.95 |
| LSTM Autoencoder | 0.93 | 1.00 | 0.96 |
| LSTM Forecaster | 0.93 | 1.00 | 0.96 |

**Real-event recall (non-synthetic).** As an honest complement to the synthetic-spike numbers, each
detector is also scored against the **five documented Central-European events** (2009 cold snap, 2010
July heat, 2012 February cold wave, 2013 August heat, 2015 heatwave): an event counts as caught if the
detector flags any point inside its window. The z-score/STL-residual detector catches **4 of 5** (it
misses the 2015 heatwave — STL absorbs the sustained warmth, so the residual does not spike), a result
verified in `verify_results.py`. This measures detection on *real* anomalies rather than injected ones.

Flagged residual anomalies have a mean |residual| of **8.09 °C vs 1.00 °C** overall — i.e. they are
genuinely extreme. Methods overlap partially because each sees something different: z-score and
ARIMA react to large residual/forecast errors, Isolation Forest to unusual multi-sensor
combinations, and the deep models to unusual shapes/sequences.

### 5.4 Clustering
K-means (k=4) on daily features recovers temperature-ordered day-types (means ≈ 1.4 / 11.5 / 16.4 /
19.2 °C), aligning with the seasons — confirming the clustering is meaningful.

### 5.5 Interface and reasoning
Example: *"What was the hottest temperature?"* → "🔢 max T (degC) was 37.04 at 2015-07-04 16:00"
plus retrieved sensor context. Reasoning at that moment correctly identifies a hot, dry day
(vapour-pressure deficit +8.4 σ).

**Reading the chart (legend).** Answers that plot a series attach a small chart with a colour-coded
legend so every marker is self-explanatory: the **blue line** is the observed data; a **solid red dot**
is an anomaly (flagged unusual); a **dashed amber ring** is a *borderline* near-miss (close to the
threshold but not flagged — shown to convey uncertainty rather than a hard yes/no); a **solid orange
dot** is the maximum and a **solid blue dot** the minimum; a **shaded band** is the 80 % forecast range.
The legend lists only the markers actually present, and the interface can answer questions *about* it
(e.g. *"what is the red dot?"*) from a fixed knowledge base — the meaning is a known fact, so, like every
other answer, it needs no computation and cannot be fabricated.

**Conversational follow-ups.** With the local model enabled, the interface keeps a short rolling memory
of recent turns, so elliptical follow-ups resolve in context: after *"show the anomalies in 2013"*, the
question *"what about 2014?"* is rewritten to a standalone question and answered for 2014, and *"what is
the red dot?"* is answered as a legend question. Crucially, the model only rewrites the *language*; the
deterministic query engine still computes every number, preserving the trust guarantee. Two guards
contain the risk of a bad rewrite: the rewriter is invoked **only for elliptical follow-ups** (a
question that already parses to an intent plus a sensor or period is answered directly, unchanged),
and a rewrite is **accepted only if the deterministic parser can route it** — a garbled rewrite is
rejected and the original question kept. The rewrite that is used is also shown to the user
("understood as…"). Without the model the interface remains single-shot (each question answered
independently) — the documented no-model limitation.

### 5.6 Offline operation and GPU
The full pipeline runs offline using only local libraries. Deep-model training is verified on both
CPU and GPU with equivalent results (~1.7× GPU speedup at this model size).

### 5.7 Statistical rigor (confidence intervals and significance)
Point estimates alone are not defensible on a small labelled set, so detector quality is now reported
with uncertainty. On the labelled synthetic-spike protocol, the z-score/STL-residual detector achieves
**recall 0.78 (95 % bootstrap CI 0.64–0.90)** and **precision 0.89 (0.76–0.98)** (2 000 resamples,
seeded). A paired **McNemar test** comparing Isolation Forest and the z-score detector on the same
points rejects the null of equal error patterns (**p ≈ 1×10⁻¹⁹**, 64 discordant points). Averaged
over **8 random seeds**, the detector's performance is recall **0.87 (95 % CI 0.73–0.97)**, precision
**0.92 (0.83–1.00)**, F1 **0.90 (0.80–0.99)** — a seed-robust estimate with an explicit interval.
Every interval also carries its labelled sample size `n`, so its width is read in context; the
intervals are deliberately wide because the labelled set is necessarily small, and are framed as
**indicative** rather than confirmatory. This McNemar
difference must be read with the class imbalance in mind (§5.3): per-point accuracy is dominated by
the ~99 % normal class, and Isolation Forest's fixed contamination prior over-flags this slice, so the
test confirms the two detectors make *significantly different* errors rather than crowning a winner —
consistent with the "no single best detector" finding. Both functions (`evaluation.bootstrap_ci`,
`evaluation.mcnemar_test`) are unit-tested and re-run inside the verification harness.

### 5.8 Circular sensors (a correctness fix in the truth layer)
Wind direction `wd (deg)` is a compass bearing that wraps at 360°, so linear statistics are invalid on
it. Previously the interface answered *"maximum wind direction"* with **359.8°** — a meaningless linear
extremum that ignores the wrap. The interface now routes circular sensors through circular statistics
and instead reports the **prevailing (circular-mean) direction 196.6° (S), circular spread 71.3°**, and
uses an unsigned circular–linear coefficient for correlation. The handling is complete across the
interface: **trend** and **forecast** on a bearing are declined with an explanation (a linear slope is
dominated by the 0°/360° wrap; ARIMA cannot model a wrapping series), and a directional **threshold**
count is still exact but carries a wrap caveat. This closes a genuine hole in the "exact numbers"
guarantee: the truth layer must be correct — or honestly decline — for *every* sensor, not only the
linear ones.

### 5.9 LLM faithfulness (the trust guarantee, measured)
The central claim — the LLM explains but never computes — is verified rather than assumed. Every LLM
answer is passed through a faithfulness check that compares the figures in the explanation against the
computed facts (rounding tolerated); if the model introduces a number that matches no computed value,
the interface **discards the phrasing and falls back to the exact template**, so a fabricated figure
cannot reach the user. The check enforces both decimals *and* larger integers — so a hallucinated
**count** ("87 windows") or a **wrong year** is caught — while tolerating legitimate reformatting
(clock times such as `16:00`→`4pm` are stripped, and single-digit counts ≤ 12 are left tolerant, a
documented residual gap). Across the verification questions, run as a **majority-of-3
vote** (the 3-billion-parameter model is stochastic), the LLM introduced **0 invented numbers**. This
turns "trust us, it only explains" into a measured, reproducible result.

---

## 6. Discussion

The strict computation/explanation split is the project's most important design decision: it removes
the largest trust risk of LLM-based analytics (hallucinated numbers) by construction. The anomaly
study shows there is **no single best detector** — the value is comparative coverage across method
families, with the deep models adding shape-/sequence-awareness at small cost. The precision result
is a methodological finding in itself: without labels, precision is hard to measure honestly, and
synthetic injection on real data confounds it with genuine background anomalies — a point the
benchmark-critique literature raises. Multi-sensor reasoning demonstrates how exact cross-sensor
statistics can support *explanations* of anomalies, not just detections.

**Verifiable NL Analytics as the core contribution.** Beyond the temperature application, the reusable
idea is an architecture: a deterministic *truth layer* answers every quantitative question, an LLM is
confined to *explanation*, and an *automated faithfulness check* verifies at answer time that the
explanation added no numbers of its own. The measured result — 0 invented numbers, with a template
fallback whenever the check fails (§5.9) — makes numerical trust an enforced property rather than a
hope. This directly targets the currently prominent problem of LLM hallucination in data analysis, and
because the truth layer is domain-agnostic the pattern transfers unchanged to other datasets (Appendix
B). We therefore frame the contribution not as "a weather chatbot" but as a general, verifiable
method for trustworthy natural-language analytics.

---

## 7. Limitations and Threats to Validity

### 7.1 Limitations
- **No real anomaly labels** in Jena, so detection accuracy is measured via synthetic injection and a
  set of *documented real-event* windows (2010 July heat, 2012 February cold wave, etc.) used as
  approximate ground truth — not a gold-standard labelled benchmark.
- **Threshold/contamination** parameters are chosen priors (justified by sensitivity sweeps), not learned.
- **Interface routing** is keyword-based with an optional LLM planner. Coverage is measured by a
  paraphrase test suite (23 realistic rephrasings across all intents route correctly, including
  plurals and comparative forms), and truly novel phrasing falls through to the LLM planner — but it
  still handles a *defined* set of intents, not fully arbitrary language.
- **Correlation, not causation.** The system quantifies association, not cause. Causal questions
  ("does X cause Y?", "what causes…?") are answered honestly: the correlation is given (when a sensor
  pair is identified) with an explicit *correlation-is-not-causation* caveat, or, for an unpaired
  question, with a statement that causal inference is out of scope. Genuine causal discovery is future
  work. The question vocabulary is likewise a defined set (max/min/mean/spread/trend/compare/
  correlation/threshold/anomaly/forecast/weather/summary + legend), not open-ended analytics.
- **Small explanation model.** The local LLM is a 3B model; its qualitative prose is weaker than a
  large model's. The impact is bounded, not eliminated: the faithfulness gate and prompt constraints
  guarantee the *numbers* (not the surrounding prose), and generation is now **reproducible** — a fixed
  seed and low temperature make the same facts yield the same phrasing across runs (verified: two
  seeded runs produce identical output), so the model's stochasticity no longer affects repeatability.
  A larger local model can be substituted via the existing `model` parameter with no code change.

### 7.2 Threats to validity
- **Construct validity.** Synthetic spikes may not resemble real anomalies; we mitigate by *also*
  evaluating on documented real events and by reporting the honest precision caveat (§5.3).
- **Statistical-conclusion validity.** Labelled points are few, so we report **bootstrap CIs and a
  paired McNemar test** rather than bare point estimates (§5.7); results are framed as indicative.
- **External validity.** A single primary domain risks over-fitting conclusions to weather. We address
  this at two levels. First, the **computational core is dataset-agnostic**: a generalisation test suite
  runs the *actual* v1 modules (query engine, anomaly detector, interface) on a non-weather domain
  (simulated server metrics — `cpu_load`, `mem_gb`) and confirms exact statistics and anomaly detection
  still hold. The dataset-specific parts are the ingestion/preprocessing adapter and the interface's
  sensor vocabulary (aliases), which are swapped per domain. Second, the **whole architecture** is shown
  transferred to a real, unrelated domain — EU trade (**Appendix B**) — where a sibling implementation
  reproduces the same guarantees. Together these separate "the method generalises" (evidenced) from
  "this exact vocabulary generalises" (it does not, by design).
- **Reproducibility & deployment.** All randomness is seeded, dependencies are pinned
  (`requirements.txt`), and `reproduce.sh` regenerates every counted result; the stochastic LLM check
  is advisory only. The dataset is gitignored (licensing/size), so a missing file now raises a clear,
  actionable error pointing to where to obtain it. The web UI uses Flask's built-in server, adequate
  for the project's local single-user scope; `create_app()` is factored so it can be served behind a
  production WSGI server (e.g. waitress/gunicorn) unchanged.
- **No human evaluation yet.** The interface's usability and trust benefits are argued from design and
  exact-answer correctness, **not yet measured with users**. A within-subjects study (spreadsheet vs
  template vs AI mode, 8 ground-truth-graded tasks, SUS/trust) is designed and ready to run
  (`HCI_USER_STUDY.md`); running it is the primary item of future work.

---

## 8. Conclusion and Future Work

The project delivers a working, offline, trustworthy time-series analytics core: exact querying,
six-method anomaly detection (reported with confidence intervals and a significance test), clustering,
a natural-language interface with an **integrated, faithfulness-checked local LLM**, RAG context and
multi-sensor reasoning, all evaluated, and shown to transfer to a second domain (Appendix B). Future
work:
1. **Run the HCI user study** (the designed within-subjects protocol) to measure usability, task
   accuracy and trust with real participants — the single highest-value next step and the natural
   completion of the "HCI" aim.
2. Add a **labelled or semi-synthetic benchmark** for proper precision/recall beyond documented events.
3. Explore a **multi-sensor deep model** (e.g. a graph network) to strengthen cross-sensor reasoning.
4. Extend the **Verifiable NL Analytics** pattern with agentic tool-use (multi-step questions,
   still numberless) and streaming/online detection.

---

## References
**[ADD]** Format per your department's gu(e.g. Harvard/IEEE). Core entries to include: Isolation
Forest (2008); STL (1990); Malhotra et al. (2015); Bai et al. (2018); Pang et al. (2021); Wu & Keogh
(2021); Lewis et al. (2020); the M100 ExaData, Vigoya and RUAD dataset papers.

## Appendix A — How to run
```bash
cd code/v1
python -m src.main --data ../data/jena_climate_2009_2016.csv --sensor "T (degC)"
python -m pytest tests/          # 87 pass (1 skipped)
python verify_results.py         # 21/21 independent checks
bash reproduce.sh                # both of the above, with an environment banner
```
Notebooks `00`–`04` reproduce the code walkthrough, exploration/visualisation, methods and evaluation,
the interface and reasoning, and the deep-model comparison.

## Appendix B — Generalisation to a second domain (EU trade)

A central claim of *Verifiable NL Analytics* (§1.6, §6) is that the architecture is domain-agnostic:
only the data changes, not the truth-layer / explanation / verification design. To test this beyond
weather, the **identical pipeline** (preprocess → analyse → detect → forecast → query engine →
interface → faithfulness-checked LLM) was applied to an unrelated dataset — **EU monthly international
trade**: 292 months × 54 series (27 EU countries × {exports, imports}), 2002–2026, in € millions, a
domain with real, documented shocks (the 2008 financial crisis, COVID-2020, the 2022 energy shock).

**Findings (from the second system's own independent verification, 13/13 checks):**

| Aspect | Weather (v1) | EU trade (Appendix B) |
| --- | --- | --- |
| Query engine == pandas | ✅ exact | ✅ exact |
| Anomaly detection | catches 2015 heatwave, 2012 cold wave | catches the **2008 crisis** and **COVID-2020** |
| Forecast honesty | ARIMA backtest MASE < 1.5 | SARIMA backtest **MAPE 4.0 %** (normal); 14.4 % across COVID → shocks *detected*, not predicted |
| Deep vs classical (honest) | no single best detector | autoencoder **F1 0.21 ≤ Isolation Forest 0.36** — reported as a loss, not hidden |
| LLM faithfulness | 0 invented numbers | 0 invented numbers |

The pattern transfers without modification to its logic: exact answers remain exact, detectors surface
the domain's true events, forecasts remain honest about shocks, and the LLM still never invents a
figure. This is evidence that the contribution is a **general method**, not a weather-specific artefact.
(The trade system is a sibling implementation; full details in its own report.)
