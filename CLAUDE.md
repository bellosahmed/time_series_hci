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

The sections below marked **(v1)** describe the temperature system. The **(v2)** section describes the
trade system. They share the *architecture* (preprocess → analyse → detect → reason → query → explain)
but **not the data** — v1 is temperature, v2 is trade.

---

## v2 — EU global trade (current)

- **Dataset:** `code/data/eu_trade_monthly.csv` — 292 months × 54 columns (27 EU countries ×
  exports/imports), 2002–2026, € millions. Real anomalies: 2008 crisis, COVID-2020, 2022 energy.
  Fetched via `code/v2/get_trade_data.py` (Eurostat API). Gitignored.
- **Pipeline (`v2/src/`):** preprocess · analysis · anomaly · reasoning (multi-country shared-shock) ·
  forecasting (SARIMA + backtest) · query_engine (truth layer) · interface · llm · rag_system
  (BM25 + LSA + BERT + chunking) · clustering (countries) · deep_anomaly (autoencoder).
- **Notebooks (`v2/notebooks/`):** 01_eda … 13_chunking_bert (documented, executed).
- **Apps:** `v2/apps/web/` (Flask browser chat). **Tests:** 74 passing. **Verify:** `v2/verify_results.py`
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
| NL interface, RAG retriever (TF-IDF), multi-sensor reasoning | ✅ done |
| Deep models: CNN-AE, LSTM-AE, LSTM-Forecaster | ✅ done |
| Forecasting: ARIMA | ✅ done |
| Evaluation: vs pandas ground truth + labeled (synthetic) precision/recall | ✅ done |
| Tests | ✅ 48 passing (1 skipped) |
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
- All deep models are **device-aware** via `deep_anomaly.get_device()` (auto: cuda → cpu).
  Pass `device="cpu"` / `device="cuda"` to override.
- **GPU memory:** inference is **batched** (`_squared_errors`, `deep_lstm._predict_batched`) so the
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
| `anomaly.py` | z-score (global + rolling), Isolation Forest, anomaly windows; `evaluation.py` adds **point-adjusted** metrics |
| `clustering.py` | daily features + K-means day-types |
| `query_engine.py` | **exact** max/min/mean/std/correlation/missing — the truth layer |
| `deep_anomaly.py` | 1D CNN autoencoder (per-point reconstruction error) |
| `deep_lstm.py` | LSTM autoencoder + LSTM forecaster |
| `forecasting.py` | ARIMA forecast-error anomalies (deseasonalised) + `forecast()` (14-day ahead, 80% band) |
| `reasoning.py` | multi-sensor anomaly explanation (per-sensor z-scores, co-movement) — wired into the interface's anomaly answers (a cross-sensor explanation feature, not the headline) |
| `rag_system.py` | local TF-IDF retriever over sensor KB + trust-aware explainer (LLM-ready) |
| `interface.py` | NL question → query engine → trust-aware answer. Intents: max/min/mean/spread/correlation/missing/summary + **trend · compare · weather(day report) · threshold · anomaly(+reasoning)** + **forecast** (14-day). Scopes to any year/month/**season**/**day**/range. Attaches a downsampled **chart** for the web UI. With `use_llm`, `llm.plan` parses free phrasing into a structured plan (fills gaps only; never computes numbers) |
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

Statistical (z-score) · Forecasting (ARIMA) · Classical ML (Isolation Forest) ·
Deep (CNN-AE, LSTM-AE, LSTM-Forecaster). Full table: **`v1/report/MODELS.md`**.

## Tests (`v1/tests/`) — 48 passing, 1 skipped

`test_query_engine` · `test_interface` (interface + evaluation) · `test_reasoning` ·
`test_deep_anomaly` · `test_deep_lstm` · `test_forecasting` · `test_llm`.

## v1 layout (reorganised)

`src/` (code) · `tests/` · `notebooks/` (00–04 + README) · `apps/` (chat.py + web/) ·
`report/` (REPORT, EVALUATION, MODELS, FINETUNING) · `verify_results.py` (17/17 independent checks).
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

1. **Ollama / local LLM** — drop-in explainer (the `rag_system.explain` interface is ready).
2. **Write-up** — methodology, results, limitations.
3. Optional: Matrix Profile (`stumpy`), LOF / One-Class SVM, a multi-sensor deep model.
