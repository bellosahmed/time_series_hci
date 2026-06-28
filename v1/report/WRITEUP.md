# Dissertation Write-up — Working Draft (v1)

> **Status: DRAFT skeleton for later.** Structure + the real measured evidence are in place;
> prose sections (esp. literature review) are marked **TODO** to expand later. Not for marking yet.
> Companion docs: `EVALUATION.md` (full results), `HCI_USER_STUDY.md` (planned study), `MODELS.md`.

**Title:** Time-Series Data Analysis Through a Natural HCI Interface
**Core objective:** a local, explainable, **trustworthy** system where a user asks questions in plain
English, **exact numbers are computed by code**, and a local LLM only *explains* them. (Novelty is
secondary; the contribution is the evaluated, verifiable, usable system.)

---

## Abstract (draft)
A local system for querying multivariate time-series in natural language. Deterministic code computes
every figure (the *truth layer*); a local LLM rephrases — never calculates — keeping answers exact and
private (no cloud). Evaluated on the Jena climate dataset: query results match pandas exactly,
forecasting beats a persistence baseline (MASE 0.83), and the LLM was measured to **invent zero
numbers**. A user-study protocol is provided to evaluate usability. *(Expand once the study runs.)*

## 1. Introduction
- **Problem:** LLMs are fluent but unreliable with numbers; spreadsheets are exact but unfriendly.
- **Aim:** combine them safely — code for truth, LLM for language.
- **Objectives:** (1) local pipeline; (2) NL interface; (3) anomaly detection + explanation;
  (4) a *verifiable trust chain*; (5) rigorous evaluation; (6) usability.
- **Contributions:** the trust architecture + its **machine verification** (`verify_results.py`),
  a flexible NL interface (year/season/day/threshold/trend/compare/forecast/weather), and an honest,
  reproducible evaluation.

## 2. Background & related work — **TODO**
- Time-series anomaly detection (statistical, ML, deep). 
- NL interfaces to data / text-to-analytics. 
- LLM hallucination and grounding; retrieval-augmented generation. 
- Trust & explainability in HCI. *(Write the literature review here.)*

## 3. Design & architecture
- **Trust principle:** computation = code; explanation = LLM. Diagram: question → intent/plan →
  query engine (truth) → reasoning/context → phrasing (template or LLM) → trust-labelled answer.
- **Why local:** privacy, reproducibility, no API cost.
- **Modules:** preprocess · analysis · anomaly · reasoning · forecasting · query_engine · interface ·
  llm · rag_system (see `CLAUDE.md` for the map).

## 4. Implementation
- Dataset: Jena Climate 2009–2016, 70,129 hourly rows × 14 sensors; `-9999` sentinels cleaned.
- Detectors: residual z-score, Isolation Forest, 1D CNN-AE, LSTM-AE, LSTM-forecaster.
- Interface: deterministic keyword/date parsing + optional LLM planner (`llm.plan`) for free phrasing;
  the LLM **fills gaps only, never computes**. Friendly dates, charts, trust labels.
- Forecasting: ARIMA on recent daily means, 14-day horizon with an 80% band.

## 5. Evaluation (measured — real numbers)
- **Query engine == pandas, exactly** (max 37.04, min −22.65, mean 9.4447, corr −0.9633; scoped
  queries match slices). Hard ground truth.
- **Anomaly detection (10-seed protocol):** strict **P 0.77 ± 0.13, R 0.66 ± 0.07, F1 0.71**;
  point-adjusted similar.
- **Parameter sensitivity:** z-threshold 4.0 is the knee (526 flags, ~0.75%); contamination 0.01.
- **Forecast backtest:** MAE 3.55 °C, RMSE 4.28 °C, **MASE 0.83** (beats persistence; MAPE avoided
  because temperature crosses 0 °C).
- **LLM faithfulness:** **0 invented numbers** (no hallucinated figures); occasional benign omission.
- **Known-event corroboration:** all-time max = July 2015 heatwave; cold extremes in winter.
- **Machine-verified:** `verify_results.py` → **17/17**. Automated tests: **48 pass**.

## 6. HCI evaluation
- See `HCI_USER_STUDY.md` for the full protocol (within-subjects: spreadsheet vs template vs AI;
  accuracy graded against ground truth; time, SUS, NASA-TLX, trust). **TODO: run + report.**

## 7. Discussion
- **Strengths:** exact + explainable + private + verified.
- **Honest limitations:** synthetic anomaly labels (no real labels in Jena); single dataset; small
  LLM; forecasting is short-horizon; reasoning is a heuristic (z-scores). 
- **Honest negative result (deep models, v2 context):** classical detectors beat the autoencoder on
  short series — documented, not hidden.

## 8. Conclusion & future work
- Conclusion: a usable, trustworthy NL analytics system with a verified truth chain. 
- Future work: run the user study; real labelled benchmarks (NAB/SKAB); larger LLM / LoRA fine-tune of
  the explainer; richer forecasting; transfer to other domains (cf. v2 trade data).

## References — **TODO**
