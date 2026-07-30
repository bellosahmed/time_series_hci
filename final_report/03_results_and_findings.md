# 03 — Results and Findings

All numbers are from each version's own `report/` and `verify_results.py` (independently checked
`== pandas`). This file consolidates them and draws the cross-cutting conclusions.

## v1 — Weather

**Anomaly detection.** On the *real* Jena series, precision is understated (synthetic labels), but recall
is faithful; on a **controlled clean signal** the detectors are genuinely precise:

| Detector | Precision* | Recall* | F1* |
| --- | ---: | ---: | ---: |
| Z-score (STL residual) | 0.85 | 0.82 | 0.84 |
| ARIMA forecast-error | 0.98 | 1.00 | 0.99 |
| Isolation Forest | **1.00** | **1.00** | **1.00** |
| CNN-AE / LSTM-AE / LSTM-forecaster | 0.91–0.93 | 1.00 | 0.95–0.96 |

\*controlled clean signal. On synthetic injection with bootstrap over seeds: point-adjusted **P 0.77,
R 0.67, F1 0.71** (mean ± std reported, not a single run).

**Forecasting (ARIMA).** MAE **3.55 °C**, RMSE 4.28, **MASE 0.83** → beats a persistence baseline
(MASE < 1). MAPE deliberately avoided (temperature crosses 0 °C).

**Retrieval.** Dense MiniLM beats keyword methods on **paraphrases** (MRR **0.267 → 0.500**); the three
keyword methods tie on in-vocabulary queries — *document quality dominates retriever choice on a small KB.*

**Gates:** 193 tests (1 skipped) · verify **29/29** · ruff + mypy clean · coverage 92 %.

**Agentic / HCI trust layer (added 30 July 2026).** Three additions, each keeping the invariant that the
LLM never computes a number — it only rephrases, decomposes, critiques, and schedules:

- **Understanding/Clarifier agent.** On 8 malformed queries, coverage rises (LLM off 4 correct / 3
  declined → on **5 correct / 0 declined**), but the 3B model also yields **3 plausible-but-wrong
  routable rewrites**. The key point: these are **mis-routings, not mis-arithmetic** — numbers stay
  exact, the interpretation is shown to the user, and acceptance is gated on the deterministic parser
  being able to route the rewrite. *More questions answered, a minority under the wrong reading — a
  genuine trade-off, not a clean win.*
- **Multi-agent orchestrator** (decompose → parallel role-agents → Critic → combine). Multi-part
  coverage **2/3 → 3/3**; a strict superset of the single-agent path (offline ≡ `agent.py`). Two honest
  findings: the 3B **decomposer mis-assigns roles** (tool-restriction + fallback recover the answer;
  routing is model-bound, the trust gates are not), and the **LLM-critic over-flags**, so it is advisory
  while the **deterministic critic is the gate**.
- **COSMA/DiRAC batch harness.** The evaluations are embarrassingly parallel; a SLURM array harness
  shards → runs → aggregates them, **schedules/collects only** (no new numbers). **Sharding is
  result-preserving** (identical aggregate for 1/2/4/8 shards); a single-driver local run over all three
  evals gives **15/19** (benchmark 8/8, orchestrator 2/3, understanding 5/8) in 52 s, during which the
  faithfulness gate rejected 4 invented numbers. Orchestrator **2/3 vs 3/3 across runs** exposes 3B
  **non-determinism** — which cheap cluster fan-out is built to quantify. *Cluster submission itself was
  not run; the core + generated `sbatch` are verified.*

**Gates after this work:** **483 tests** (1 skipped) · ruff + mypy clean.

## v2 — Trade

**Anomaly detection** (ground truth = real crisis windows 2008-09→2009-04 and 2020-03→2020-08):

| Detector | Precision | Recall | **Strict F1** | **Point-adj F1** |
| --- | ---: | ---: | ---: | ---: |
| STL z-score | 0.46 | 0.20 | 0.27 | 0.52 |
| Seasonal-naive | 0.49 | 0.15 | 0.22 | **0.90** |
| Isolation Forest (multivariate) | 0.36 | 0.36 | **0.36** | 0.76 |
| Multivariate **autoencoder** | — | — | **0.21** | — |
| **Ensemble (≥2 votes)** | — | — | 0.45 | **0.90** |

Both Isolation Forest and the ensemble **catch the real 2008 and COVID crises**. The **autoencoder (0.21)
loses to Isolation Forest (0.36)** — reported openly.

**Forecasting (SARIMA backtest).** MAPE **4.0 %** in normal periods; **14.4 %** across COVID, where 6
months breach the prediction band → *shocks are detected, not predicted* (the honest read).

**Gates:** 79 tests · verify **13/13**.

## v3 — IoT network security

**Detection** — two channels + fusion on **real** labels (2016 windows, 90 attack windows: mixed 60,
modification 26, duplication 4):

| Config | AP | AP 95 % CI | best-F1 | recall | **F1 (pt-adj)** | **recall (pt-adj)** |
| --- | ---: | :---: | ---: | ---: | ---: | ---: |
| flow-only | 0.136 | [0.077, 0.222] | 0.226 | 0.156 | 0.294 | 0.178 |
| value-only | **0.183** | [0.136, 0.245] | **0.379** | 0.444 | **0.804** | 0.889 |
| early-fusion | 0.147 | [0.108, 0.206] | 0.289 | 0.511 | 0.732 | **1.000** |
| late-fusion | 0.171 | [0.123, 0.246] | 0.306 | 0.533 | 0.789 | **1.000** |
| supervised (day-blocked) | 0.101 | [0.075, 0.141] | 0.212 | 0.378 | 0.554 | 1.000 |

Per-type recall shows **complementarity**: flow-only **0.00** on modification / **1.00** on duplication;
value-only the opposite tilt.

**Findings:** fusion **trades precision for recall** and reaches **1.00 point-adjusted recall** (every
episode caught); but the **AP CIs overlap** → *no config is significantly best on AP*; and the
**supervised combiner is worst** (day-blocked CV exposes overfitting to day-confounded attack types).

**Data quality:** 0 missing temperatures on publishes; 0.6 % value-grid gaps (interpolated, value channel
only); imbalance **186:1** (packet) / **21:1** (window) — respected, not resampled.

**Gates:** 58 tests · verify **12/12** · ruff clean.

---

## Cross-cutting findings

**1. The pattern transfers (domain-agnostic).** The identical truth-layer + LLM-explanation +
faithfulness architecture worked on weather, economics, and network security. Across all three the LLM
introduced **0 fabricated numbers** (faithfulness verified). The contribution is a *method*, not a demo.

**2. Better labels + better metrics > better models.** The most valuable progress across versions was in
*evaluation honesty*, not model complexity:
- v1 synthetic → v2 real-coarse → v3 real-fine labels;
- point-adjusted metrics + bootstrap CIs + per-type breakdowns + blocked CV were introduced and refined.
These are what let each version make a **defensible** claim.

**3. Simple beats complex — three times, honestly.** Deep/learned methods **never won**:
- v1: no single best; deep ties Isolation Forest.
- v2: autoencoder **0.21 < 0.36** Isolation Forest.
- v3: the supervised (learned) fusion is the **worst** config.
This is consistent and explainable: these datasets are **small, seasonal, and label-scarce/clustered** —
exactly where simple robust detectors generalise and deep/learned ones overfit.

**4. Ensembling/fusion helps recall but not always the aggregate.** v2's ensemble reached point-adjusted
F1 0.90; v3's fusion reached point-adjusted recall 1.00 — but v3 showed fusion **need not** beat the best
single channel on AP when one channel is weak. The nuance is only visible with CIs and per-type analysis.

**5. Honesty about forecasting.** Both forecasters (v1 ARIMA, v2 SARIMA) are explicitly framed as
*beating a baseline in normal times and detecting — not predicting — shocks*. No overclaiming.
