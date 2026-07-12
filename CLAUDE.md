# Project Context & Work Index — time_series_hci

**Topic:** Time-Series Data Analysis Through a Natural HCI Interface
A local, explainable, interactive time-series analytics system: exact numbers are computed by
deterministic Python code; a local LLM only *explains* them.

> **Core principle:** Computation = code (the source of truth). LLM = explanation only — it never
> calculates numbers.

_Last updated: 22 June 2026._

## Versions — DIFFERENT DOMAINS (do not conflate)

| Version | Domain | Dataset | Series |
| --- | --- | --- | --- |
| **v1** | **Temperature / weather** | Jena Climate 2009–2016 | 14 sensors, hourly (70k rows) |
| **v2** | **Global trade (supply chain)** | EU monthly international trade | 54 country-flows, monthly (292 rows) |
| **v3** | **Data-center IoT network (security)** | DAD (Vigoya et al. 2020, MQTT pcaps) | 7 days, ~101k packets, **real** attack labels |

**Cross-version synthesis:** `code/final_report/` — executive summary + **master v1/v2/v3 comparison
table**, techniques + why-this-not-that, consolidated results/findings, limitations + future work.

**v3 status: core pipeline built (M0–M6).** Real DAD dataset (`code/data/dad/`, gitignored; 544 labelled
attacks matching the paper). Implemented: `preprocess` (scapy pcap→labelled MESSAGES frame) · `features`
(value + flow channels) · `channels`/`fusion`/`evaluation`/`experiment` (two-channel + fusion detector,
**real-label** eval with point-adjusted metrics + bootstrap CIs) · `query_engine` (truth layer) ·
`interface`/`interface_parsing` (NL, trust-labelled) · `llm` (Ollama + faithfulness gate). Apps:
`chat.py` + Flask `web/`. **58 tests · verify 12/12 · ruff clean.** Its draw: **real anomaly labels**
(fixes v1's synthetic-label limitation). Headline finding (honest): channels are complementary; fusion
trades precision for recall and reaches 1.00 event-level recall, but AP CIs overlap so no config is
significantly best. Running record: `v3/report/EVALUATION.md`. **Optional future:** detector tuning,
supervised score-combiner.

The sections below marked **(v1)** describe the temperature system. The **(v2)** section describes the
trade system. They share the *architecture* (preprocess → analyse → detect → reason → query → explain)
but **not the data** — v1 is temperature, v2 is trade.

---

## v2 — EU global trade (current)

- **Dataset:** `code/data/eu_trade_monthly.csv` — 292 months × 54 columns (27 EU countries ×
  exports/imports), 2002–2026, € millions. Real anomalies: 2008 crisis, COVID-2020, 2022 energy.
  Fetched via `code/v2/tools/get_trade_data.py` (Eurostat API). Gitignored.
- **Pipeline (`v2/src/`):** preprocess · analysis · anomaly · reasoning (multi-country shared-shock) ·
  forecasting (SARIMA + backtest) · query_engine (truth layer) · interface · llm · rag_system
  (BM25 + LSA + BERT + chunking) · clustering (countries) · deep_anomaly (autoencoder).
- **Notebooks (`v2/notebooks/`):** 01_eda … 13_chunking_bert (documented, executed).
- **Apps:** `v2/apps/web/` (Flask browser chat). **Tests:** 79 passing. **Verify:** `v2/verify_results.py`
  (13/13). **Docs:** `v2/README.md`, `v2/report/` (REPORT, EVALUATION, VERIFICATION), `v2/GLOSSARY.md`,
  `v2/DEEP_MODELS.md`.
- **Key results:** detectors catch real 2008 & COVID; forecast backtest MAPE 4.0%; deep AE (0.21)
  honestly loses to Isolation Forest (0.36); query engine == pandas; LLM keeps numbers exact.

---

## v1 — Temperature / Jena climate (below)

---

## Status at a glance

| Area | State |
| --- | --- |
| Preprocessing, analysis, anomaly detection, clustering, query engine | ✅ done |
| NL interface (incl. **day-type** classification), retrieval, multi-sensor reasoning | ✅ done |
| Deep models: CNN-AE, LSTM-AE, LSTM-Forecaster | ✅ done |
| Forecasting: ARIMA | ✅ done |
| Evaluation: vs pandas ground truth + labeled (synthetic) precision/recall | ✅ done |
| Tests | ✅ 193 passing (1 skipped); ruff + mypy clean; coverage 92% |
| Agentic multi-step tool-use (`agent.py`), embedding-RAG option, larger-model auto-pick | ✅ done |
| Retrieval eval (P@k/R@k/MRR/MAP/NDCG) + TF-IDF/BM25/LSA/dense retrievers; Matrix Profile detector (COP509 IR) | ✅ done |
| Local LLM (Ollama) explanation layer | ⏳ deferred to write-up stage |
| Write-up / report | ⏳ to do |

---

## Dataset

**Jena Climate 2009–2016** — `code/data/jena_climate_2009_2016.csv` (gitignored).
420,551 rows @ 10 min → resampled to **70,129 hourly** rows. 14 sensors; target `T (degC)`.
Note: wind columns contain `-9999` sentinel errors, cleaned in preprocessing.

## Environment (important)

- The user's Jupyter runs on **Python 3.12** (`/usr/bin/python3.12`); deps installed there
  (numpy, pandas, scikit-learn, statsmodels, matplotlib, torch, pytest).
- There is also a **3.14 venv** at `code/.venv` (no torch — torch is 3.12 only).
- Notebooks pin to a **Python 3.12** kernel and have a self-bootstrap cell for robustness.
- Run notebooks with **Restart & Run All**.

### GPU (CUDA)

- GPU: **NVIDIA GeForce RTX 5070 Laptop (Blackwell, sm_120, 8 GB)**, driver 595.71, CUDA 13.2.
- Installed **torch 2.11.0+cu128** (CUDA 12.8 build, required for Blackwell sm_120):
  `pip install torch --index-url https://download.pytorch.org/whl/cu128`.
- All deep models are **device-aware** via `deep.get_device()` (auto: cuda → cpu).
  Pass `device="cpu"` / `device="cuda"` to override.
- **GPU memory:** inference is **batched** (`deep._squared_errors`, `deep._predict_batched`) so the
  full ~70k-window series is never resident at once; `_free_cuda()` returns memory after each model.
  This avoids CUDA OOM in notebook 04 even on the 8 GB card. If a stale kernel still hogs VRAM,
  **Restart Kernel** to free it.
- Verified: 48 tests pass (1 skipped) on GPU; CPU and GPU give equivalent results (same anomaly count,
  strong anomalies agree; borderline points differ slightly due to float/RNG differences across
  hardware). Speedup is modest (~1.7x) because the models are small.

## Source modules (`v1/src/`)

| Module | Role |
| --- | --- |
| `preprocess.py` | load, clean (`-9999` sentinels), interpolate, align to hourly |
| `analysis.py` | rolling stats, STL decomposition, correlations |
| `anomaly.py` | z-score (global + rolling), Isolation Forest, **LOF**, **One-Class SVM**, **Matrix Profile** (discords, stumpy), **Mahalanobis** (cross-sectional covariance) + **flat-line** (variance-collapse) + **`sensor_network_anomalies`** portfolio (the notebook-05 recommendation, wired into `main.py`), anomaly windows; `evaluation.py` adds **point-adjusted** metrics |
| `clustering.py` | daily features + K-means day-types (wired into the interface as the **day-type** intent) |
| `query_engine.py` | **exact** max/min/mean/std/correlation/missing — the truth layer |
| `deep.py` | deep detectors in one module: 1D CNN autoencoder + LSTM autoencoder + LSTM forecaster (per-point reconstruction / forecast error) |
| `forecasting.py` | ARIMA forecast-error anomalies (deseasonalised) + `forecast()` (14-day ahead, 80% band) |
| `reasoning.py` | multi-sensor anomaly explanation (per-sensor z-scores, co-movement) — wired into the interface's anomaly answers (a cross-sensor explanation feature, not the headline) |
| `rag_system.py` | local retrieval over sensor KB — TF-IDF, **BM25**, **LSA** (TF-IDF→SVD), **dense sentence-transformer** (all-MiniLM-L6-v2) — + trust-aware explainer |
| `retrieval_eval.py` | IR metrics (Precision@k/Recall@k/MRR/MAP/NDCG) + labelled query set to evaluate/compare retrievers (COP509 IR material) |
| `interface.py` | NL question → query engine → trust-aware answer. Intents: max/min/mean/spread/correlation/missing/summary + **trend · compare · weather(day report) · threshold · anomaly(+reasoning) · day-type(K-means)** + **forecast** (14-day). The **anomaly** intent is **network-level** with a **selectable detector** (`NETWORK_DETECTORS`: portfolio(default)/mahalanobis/isolation_forest/lof/ocsvm) — exposed as `Interface(detector=…)`, `chat.py --detector`, and a web dropdown (`/api/detectors`). Scopes to any year/month/**season**/**day**/range. Attaches a **chart** for the web UI with **day/month granularity views** (a resolution toggle) and a two-year **comparison overlay** for `compare`. **Conversational follow-ups** work **offline**: `resolve_followup` deterministically carries the prior turn's intent/sensor and swaps the new year/period/day/sensor ("what about 2014?"), with the `llm.rewrite_followup` path as the richer option when the LLM is on. With `use_llm`, `llm.plan` parses free phrasing into a structured plan (fills gaps only; never computes numbers) |
| `agent.py` | **agentic multi-step tool-use** (still numberless): LLM plans query-engine tool calls (extremum/average/correlation/threshold/anomaly_count) for multi-step questions; executor runs each == pandas; synthesis is faithfulness-gated; falls back to single-step `interface`. `chat.py --agent` |
| `main.py` | end-to-end pipeline entry point |

## Notebooks (`v1/notebooks/`)

| Notebook | Contents |
| --- | --- |
| `00_code_walkthrough` | tour of every src module with runnable examples |
| `01_data_exploration_and_visualization` | dataset documentation + full visual gallery (merged old 01+04) |
| `02_methods_and_evaluation` | methods (query engine, anomaly, clustering) + correctness vs pandas (merged old 02+03) |
| `03_interface_reasoning` | NL interface, multi-sensor reasoning, labeled evaluation |
| `04_deep_models_comparison` | CNN-AE train + 3 deep techniques + all 6 detectors compared (merged old 06+07+08) |

Notebooks consolidated 9 → 5 (28 June 2026). Run with the **`py312-tshci`** kernel.

## Models implemented

Statistical (z-score) · Forecasting (ARIMA) · Classical ML (Isolation Forest, LOF, One-Class SVM) ·
Cross-sectional/variance (Mahalanobis + flat-line, unioned as the `sensor_network_anomalies` portfolio) ·
Shape/discords (Matrix Profile) · Deep (CNN-AE, LSTM-AE, LSTM-Forecaster).
Full table: **`v1/report/MODELS.md`**.

## Tests (`v1/tests/`) — 193 passing, 1 skipped (coverage 92%)

`test_query_engine` · `test_interface` (interface + evaluation) · `test_reasoning` ·
`test_deep` · `test_forecasting` · `test_llm` · `test_agent` (agentic tool-use) ·
`test_rag` (TF-IDF/BM25/LSA/dense retrievers) · `test_retrieval_eval` (IR metrics + paraphrase gap) ·
`test_anomaly` (incl. Matrix Profile) · `test_clustering` · `test_main` · `test_preprocess` · `test_apps` · `test_generalization` · `test_web_js`.

## v1 layout (reorganised)

`src/` (code) · `tests/` · `notebooks/` (00–04 + README) · `apps/` (chat.py + web/) ·
`report/` (REPORT, EVALUATION, MODELS, DATA_QUALITY, HCI_USER_STUDY, FINETUNING; working notes in
`report/notes/`) · `docs/` (UNDERSTANDING_V1, ASSESSMENT_AND_RECOMMENDATIONS, AGENTIC_AND_POLISH,
RETRIEVAL_EVALUATION, CHAT_MEMORY_PLAN) · `pyproject.toml` (package + pytest/ruff/mypy config) · `verify_results.py`
(29/29 independent checks). Quality gates: pytest 182, verify 29/29, ruff clean, mypy clean, coverage 92%.
See `v1/README.md`.

## How to run

```bash
cd code/v1
python -m src.main --data ../data/jena_climate_2009_2016.csv --sensor "T (degC)"
python -m pytest tests/
python apps/chat.py --llm "what was the hottest temperature?"
python apps/web/app.py            # website (prints the URL)
```

## Key docs

- `v1/README.md` — v1 map + how to run
- `v1/notebooks/00_code_walkthrough.ipynb` — read the code as a notebook
- `v1/report/REPORT.md` — MSc report draft
- `v1/report/EVALUATION.md` — evaluation results
- `v1/report/MODELS.md` — model comparison + confusion matrix
- `v1/report/FINETUNING.md` — fine-tuning side note

## Conventions

- Versioned folders: current work in `v1/`; future iterations in `v2/`, ...
- Commits are authored solely by the user (no AI co-author trailer).
- **Do not push to GitHub** unless the user explicitly asks.
- Data and `.venv` are gitignored.

## Remaining work

1. **HCI user study** — the grade-gating item (SUS, task accuracy vs spreadsheet, trust); start
   ethics approval in parallel. See `report/HCI_USER_STUDY.md` + `docs/ASSESSMENT_AND_RECOMMENDATIONS.md`.
2. **Write-up** — methodology, results, limitations.
3. Cheap credibility: GitHub Actions CI (pytest + ruff + verify), coverage number, `git tag v1.0`.
4. Optional: Matrix Profile (`stumpy`), LOF / One-Class SVM, a multi-sensor deep model.

**Done (2026-07-05):** agentic multi-step tool-use (`agent.py`), embedding-RAG option, larger-model
auto-pick, `resolve_threshold` fix, mypy — see `docs/AGENTIC_AND_POLISH.md`. Retrieval evaluation
(`retrieval_eval.py`: P@k/R@k/MRR/MAP/NDCG) + four retrievers (TF-IDF/BM25/LSA/**dense** MiniLM) +
Matrix Profile detector, from COP509 IR material — see `docs/RETRIEVAL_EVALUATION.md`. Key result:
dense embeddings measurably beat keyword retrieval on paraphrases (MRR 0.267 → 0.500).
