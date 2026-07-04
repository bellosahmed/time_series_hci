# Time-Series Data Analysis Through a Natural HCI Interface

**MSc Project Report (Draft)**
_Author: Bello Ahmed · Loughborough University · Last updated: 21 June 2026_

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
for phrasing — it never computes numbers. The system is evaluated on the Jena Climate dataset
(2009–2016; 70,129 hourly readings across 14 sensors). Anomaly detection is studied across six
methods spanning statistical, forecasting, classical-machine-learning and deep-learning families.
The query engine is shown to match an independent `pandas` ground truth exactly, and the detectors
achieve high recall on injected anomalies (precision 0.85–1.00 on a controlled signal). The result
is a fully offline pipeline that combines exact numerical querying, anomaly detection, multi-sensor
reasoning and a natural-language interface.

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
- A **trust-separated architecture**: deterministic computation vs language-based explanation.
- A comparative study of **six** anomaly-detection methods on a common dataset and harness.
- A **multi-sensor reasoning** component that explains anomalies using cross-sensor behaviour.
- A fully **offline** natural-language interface with trust-labelled answers.

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
                       [Local LLM — explanation only, future]
```

### 4.2 Modules (`v1/src/`)
`preprocess` · `analysis` · `anomaly` · `clustering` · `query_engine` · `deep` (CNN-AE,
LSTM-AE, LSTM forecaster) · `forecasting` (ARIMA) · `reasoning` · `rag_system`
· `interface` · `main`.

### 4.3 Technology
Python; pandas, NumPy, scikit-learn, statsmodels, matplotlib; PyTorch for deep models. Everything
runs locally. Deep models are **GPU-aware** (auto-select CUDA): verified on an NVIDIA RTX 5070
(Blackwell) with the cu128 PyTorch build; results are equivalent to CPU.

### 4.4 Testing
21 automated tests (`v1/tests/`): query-engine correctness vs pandas, interface routing, synthetic
evaluation, and the deep / ARIMA detectors.

---

## 5. Results & Evaluation

### 5.1 Data characteristics
Temperature ranges from −22.7 °C (07 Jan 2009) to 37.0 °C (04 Jul 2015), mean 9.4 °C. It correlates
+0.997 with potential temperature and **−0.96 with air density** (cold air is denser) — physically
correct, confirming the pipeline handles the data soundly.

### 5.2 Query-engine correctness
On the full dataset, every exact query matched the independent `pandas` ground truth **exactly**
(values and timestamps) — 7/7 numeric checks, plus 8/8 unit tests. This establishes the trust
guarantee the architecture depends on.

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

### 5.6 Offline operation and GPU
The full pipeline runs offline using only local libraries. Deep-model training is verified on both
CPU and GPU with equivalent results (~1.7× GPU speedup at this model size).

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

---

## 7. Limitations
- **No real anomaly labels** in Jena, so detection accuracy is measured only via synthetic injection
  and sanity checks.
- **Single dataset** — generalisation to other domains is untested.
- **Threshold/contamination** parameters are chosen priors, not learned.
- **Interface routing** is keyword-based; it handles a defined set of intents, not arbitrary phrasing.
- **The explanation LLM is not yet integrated** — answers are currently template-phrased.
- No formal statistical-significance testing or hyperparameter search.

---

## 8. Conclusion and Future Work

The project delivers a working, offline, trustworthy time-series analytics core: exact querying,
six-method anomaly detection, clustering, a natural-language interface, RAG context and multi-sensor
reasoning, all evaluated. Future work:
1. **Integrate a local LLM (Ollama)** to phrase the (already trust-labelled) grounded facts — the
   `rag_system.explain` interface is ready for it.
2. Add a **labelled or semi-synthetic benchmark** for proper precision/recall.
3. Explore a **multi-sensor deep model** (e.g. a graph network) to strengthen cross-sensor reasoning.
4. Build a **Streamlit UI** over the interface for an interactive demo.
5. Test on **additional datasets** to assess generalisation.

---

## References
**[ADD]** Format per your department's gu(e.g. Harvard/IEEE). Core entries to include: Isolation
Forest (2008); STL (1990); Malhotra et al. (2015); Bai et al. (2018); Pang et al. (2021); Wu & Keogh
(2021); Lewis et al. (2020); the M100 ExaData, Vigoya and RUAD dataset papers.

## Appendix A — How to run
```bash
cd code/v1
python -m src.main --data ../data/jena_climate_2009_2016.csv --sensor "T (degC)"
python -m pytest tests/
```
Notebooks `01`–`08` reproduce exploration, evaluation, the methods, visualisations, the interface,
the deep models and the model comparison.
