# Time-Series Data Analysis Through a Natural HCI Interface

**MSc Project Report (Draft)**
_Author: Bello Ahmed · Loughborough University · Last updated: 6 July 2026_

> This is a working draft generated from the implemented system and its measured results. All numbers
> come from the actual runs in `v1/notebooks/` and the test/verification suites. Remaining author
> tasks are personal reflection and any supervisor-specific framing; reference formatting should be
> confirmed against the department's exact style guide.

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

This work sits at the intersection of six strands: multi-sensor monitoring data, statistical
behaviour modelling, anomaly detection, evaluation methodology, explainable/interactive systems, and
local privacy-preserving deployment.

**Monitoring datasets.** Multi-sensor environmental and infrastructure monitoring produces long,
seasonal, multivariate series that carry real data-quality faults. Comparable annotated collections
include the data-centre temperature dataset of Vigoya *et al.* (2020) and the M100 ExaData
supercomputer-telemetry campaign (Borghesi *et al.*, 2023). The Jena Climate record used here shares
their multivariate, strongly-seasonal character while adding documented `-9999` sensor faults, making
it a realistic testbed for a pipeline that must both clean and reason about the data.

**Statistical behaviour modelling.** Seasonal-Trend decomposition using Loess (STL; Cleveland *et
al.*, 1990) separates a series into trend, seasonal and residual components, and classical
Box–Jenkins ARIMA modelling (Box and Jenkins, 1970) captures autocorrelated structure. Both underpin
the residual- and forecast-error anomaly signals used in this project.

**Anomaly detection.** Approaches span statistical residual tests; distance/density methods such as
the Local Outlier Factor (Breunig *et al.*, 2000) and the One-Class SVM (Schölkopf *et al.*, 2001);
the tree-based Isolation Forest (Liu, Ting and Zhou, 2008); shape-based discords via the Matrix
Profile (Yeh *et al.*, 2016); and deep unsupervised detectors — LSTM encoder–decoders (Malhotra *et
al.*, 2015) and temporal convolutional networks (Bai, Kolter and Koltun, 2018). Pang *et al.* (2021)
survey the deep family. This project implements a representative detector from each family and
compares them on an equal footing rather than advocating a single method.

**Evaluation methodology.** Wu and Keogh (2021) show that many time-series anomaly benchmarks are
flawed and create an "illusion of progress", which motivates this project's reliance on *synthetic
injection with known labels*, point-adjusted metrics, bootstrap confidence intervals and a paired
significance test rather than a single headline score.

**Explainability, retrieval and interaction.** Retrieval-Augmented Generation (Lewis *et al.*, 2020)
grounds language-model output in retrieved evidence; the retriever here is evaluated with classical
information-retrieval ranking — TF-IDF weighting after Sparck Jones (1972), BM25 (Robertson and
Zaragoza, 2009) and Latent Semantic Analysis (Deerwester *et al.*, 1990) — alongside a dense
Sentence-BERT encoder (Reimers and Gurevych, 2019). Critically, language models are prone to
*hallucination*, fabricating fluent but unsupported facts (Ji *et al.*, 2023); in a numerical-analytics
setting this is unacceptable, and it directly motivates the strict computation/explanation separation
and the automated faithfulness check that define this project's contribution.

**Gap.** Many works *detect* anomalies, but few combine **exact numerical querying + natural-language
explanation + an interactive interface + local, privacy-preserving deployment**, and fewer still
*guarantee* that the explanation introduces no figure absent from the computed facts. This project
targets that intersection, separating computation from explanation to make numerical trust verifiable.

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
**296 automated tests** pass (1 skipped) (`v1/tests/`): query-engine correctness vs pandas, interface
routing (incl. circular wind-direction handling and the LLM trust-gate), the Flask/CLI app layer,
the client-side chart/legend JavaScript (run in Node), synthetic evaluation with confidence
intervals, the retriever IR metrics, and the deep / ARIMA detectors. An independent
harness (`verify_results.py`) additionally recomputes results a different way and passes **31/31**
checks (plus advisory LLM-faithfulness/ablation/adversarial-probe and LLM-judge cross-check votes);
`reproduce.sh` regenerates everything in one command.

---

## 5. Results & Evaluation

### 5.1 Data characteristics
Temperature ranges from −22.7 °C (07 Jan 2009) to 37.04 °C (04 Jul 2015), mean 9.4 °C. It correlates
+0.997 with potential temperature and **−0.96 with air density** (cold air is denser) — physically
correct, confirming the pipeline handles the data soundly.

### 5.2 Query-engine correctness
On the full dataset, every exact query matched the independent `pandas` ground truth **exactly**
(values and timestamps): the `verify_results.py` harness passes **31/31** cross-checks — max, min,
mean and correlation equal pandas; scoped and threshold queries equal the corresponding pandas slice;
per-sensor availability and skew are recomputed and agree; and the retriever IR metrics are
independently reproduced — and the unit suite (**296 tests**) covers the same guarantees. This
establishes the trust guarantee the architecture depends on.

### 5.3 Anomaly-detection comparison
| Model | Type | Flags | Time | Precision* | Recall* | F1* |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Z-score (residual) | Statistical | 184 | ~4.1 s | 0.24 | 0.45 | 0.32 |
| ARIMA forecast | Forecasting | 495 | ~2.2 s | 0.20 | 1.00 | 0.34 |
| Isolation Forest | Classical ML | 699 | ~0.3 s | 0.14 | 1.00 | 0.25 |
| CNN Autoencoder | Deep (recon) | 702 | ~2.3 s | 0.14 | 1.00 | 0.25 |
| LSTM Autoencoder | Deep (recon) | 702 | ~0.4 s | 0.14 | 1.00 | 0.25 |
| LSTM Forecaster | Deep (forecast) | 702 | ~1.3 s | 0.14 | 1.00 | 0.25 |

\* From injecting 100 synthetic anomalies into the **real** series. Precision is understated because
genuine background anomalies are flagged but count as false positives against synthetic-only labels.

**Fair precision (controlled clean signal, no hidden anomalies):**
| Model | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Z-score | 0.85 | 0.82 | 0.84 |
| ARIMA | 0.98 | 1.00 | 0.99 |
| Isolation Forest | 1.00 | 1.00 | 1.00 |
| CNN Autoencoder | 0.91 | 1.00 | 0.95 |
| LSTM Autoencoder | 0.91 | 1.00 | 0.95 |
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
K-means (k=4) on daily features recovers temperature-ordered day-types (means ≈ 0.5 / 6.5 / 12.2 /
19.6 °C — cold/cool/mild/warm), aligning with the seasons — confirming the clustering is meaningful.

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
over the same protocol across **8 random seeds** (`evaluation.benchmark`, seeds 0–7), the detector's
performance is recall **0.76 (95 % seed-percentile CI 0.59–0.90)**, precision **0.89 (0.78–1.00)**,
F1 **0.81 (0.70–0.92)** — a seed-robust estimate with an explicit interval, and the wider spread
relative to the single-seed bootstrap above is itself informative: performance is more seed-sensitive
than a single draw suggests, reinforcing why this is reported as indicative, not confirmatory.
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
  paraphrase test suite (19 realistic rephrasings across all intents route correctly, including some
  plural forms); comparative phrasing ("hotter than last year") currently routes to `trend` rather
  than a dedicated `compare` intent — an acknowledged rough edge, not yet covered by a test — and
  truly novel phrasing falls through to the LLM planner — but it
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

_Author–date (Harvard) style; confirm against the department's exact style guide before submission._

Bai, S., Kolter, J.Z. and Koltun, V. (2018) *An Empirical Evaluation of Generic Convolutional and
Recurrent Networks for Sequence Modeling*. arXiv:1803.01271.

Borghesi, A., Molan, M., Milano, M. and Bartolini, A. (2023) 'M100 ExaData: a data collection
campaign on the CINECA Marconi100 Tier-0 supercomputer', *Scientific Data*, 10, 288.

Box, G.E.P. and Jenkins, G.M. (1970) *Time Series Analysis: Forecasting and Control*. San Francisco:
Holden-Day.

Breunig, M.M., Kriegel, H.-P., Ng, R.T. and Sander, J. (2000) 'LOF: Identifying Density-Based Local
Outliers', *Proceedings of the ACM SIGMOD International Conference on Management of Data*, pp. 93–104.

Cleveland, R.B., Cleveland, W.S., McRae, J.E. and Terpenning, I. (1990) 'STL: A Seasonal-Trend
Decomposition Procedure Based on Loess', *Journal of Official Statistics*, 6(1), pp. 3–73.

Deerwester, S., Dumais, S.T., Furnas, G.W., Landauer, T.K. and Harshman, R. (1990) 'Indexing by
Latent Semantic Analysis', *Journal of the American Society for Information Science*, 41(6),
pp. 391–407.

Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., Ishii, E., Bang, Y.J., Madotto, A. and Fung, P.
(2023) 'Survey of Hallucination in Natural Language Generation', *ACM Computing Surveys*, 55(12),
pp. 1–38.

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M.,
Yih, W., Rocktäschel, T., Riedel, S. and Kiela, D. (2020) 'Retrieval-Augmented Generation for
Knowledge-Intensive NLP Tasks', *Advances in Neural Information Processing Systems (NeurIPS)*, 33,
pp. 9459–9474.

Liu, F.T., Ting, K.M. and Zhou, Z.-H. (2008) 'Isolation Forest', *Proceedings of the IEEE
International Conference on Data Mining (ICDM)*, pp. 413–422.

Malhotra, P., Vig, L., Shroff, G. and Agarwal, P. (2015) 'Long Short Term Memory Networks for Anomaly
Detection in Time Series', *Proceedings of the European Symposium on Artificial Neural Networks
(ESANN)*, pp. 89–94.

Pang, G., Shen, C., Cao, L. and van den Hengel, A. (2021) 'Deep Learning for Anomaly Detection: A
Review', *ACM Computing Surveys*, 54(2), pp. 1–38.

Reimers, N. and Gurevych, I. (2019) 'Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks',
*Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP)*,
pp. 3982–3992.

Robertson, S. and Zaragoza, H. (2009) 'The Probabilistic Relevance Framework: BM25 and Beyond',
*Foundations and Trends in Information Retrieval*, 3(4), pp. 333–389.

Schölkopf, B., Platt, J.C., Shawe-Taylor, J., Smola, A.J. and Williamson, R.C. (2001) 'Estimating the
Support of a High-Dimensional Distribution', *Neural Computation*, 13(7), pp. 1443–1471.

Sparck Jones, K. (1972) 'A Statistical Interpretation of Term Specificity and its Application in
Retrieval', *Journal of Documentation*, 28(1), pp. 11–21.

Vigoya, L., Fernandez, D., Carneiro, V. and Cacheda, F. (2020) 'Annotated Dataset for Anomaly
Detection in a Data Center with IoT Sensors', *Sensors*, 20(13), 3745.

Wu, R. and Keogh, E.J. (2021) 'Current Time Series Anomaly Detection Benchmarks are Flawed and are
Creating the Illusion of Progress', *IEEE Transactions on Knowledge and Data Engineering*.

Yeh, C.-C.M., Zhu, Y., Ulanova, L., Begum, N., Ding, Y., Dau, H.A., Silva, D.F., Mueen, A. and Keogh,
E. (2016) 'Matrix Profile I: All Pairs Similarity Joins for Time Series', *Proceedings of the IEEE
International Conference on Data Mining (ICDM)*, pp. 1317–1322.

## Appendix A — How to run
```bash
cd code/v1
python -m src.main --data data/jena_climate_2009_2016.csv --sensor "T (degC)"
python -m pytest tests/          # 296 pass (1 skipped)
python verify_results.py         # 31/31 independent checks
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
