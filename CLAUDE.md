# Project Context & Work Index — time_series_hci

**Topic:** Time-Series Data Analysis Through a Natural HCI Interface
A local, explainable, interactive time-series analytics system: exact numbers are computed by
deterministic Python code; a local LLM (later) only *explains* them.

> **Core principle:** Computation = code (the source of truth). LLM = explanation only — it never
> calculates numbers.

_Last updated: 21 June 2026._

---

## Status at a glance

| Area | State |
| --- | --- |
| Preprocessing, analysis, anomaly detection, clustering, query engine | ✅ done |
| NL interface, RAG retriever (TF-IDF), multi-sensor reasoning | ✅ done |
| Deep models: CNN-AE, LSTM-AE, LSTM-Forecaster | ✅ done |
| Forecasting: ARIMA | ✅ done |
| Evaluation: vs pandas ground truth + labeled (synthetic) precision/recall | ✅ done |
| Tests | ✅ 21 passing |
| Local LLM (Ollama) explanation layer | ⏳ deferred to write-up stage |
| Write-up / report | ⏳ to do |

---

## Dataset

**Jena Climate 2009–2016** — `code/data/jena_climate_2009_2016.csv` (gitignored).
420,551 rows @ 10 min → resampled to **70,129 hourly** rows. 14 sensors; target `T (degC)`.
Note: wind columns contain `-9999` sentinel errors, cleaned in preprocessing.

## Environment (important)

- The user's Jupyter runs on **Python 3.12** (`/usr/bin/python3.12`); deps installed there
  (numpy, pandas, scikit-learn, statsmodels, matplotlib, torch CPU, pytest).
- There is also a **3.14 venv** at `code/.venv` (no torch — torch is 3.12 only).
- Notebooks pin to a **Python 3.12** kernel and have a self-bootstrap cell for robustness.
- Run notebooks with **Restart & Run All**.

## Source modules (`v1/src/`)

| Module | Role |
| --- | --- |
| `preprocess.py` | load, clean (`-9999` sentinels), interpolate, align to hourly |
| `analysis.py` | rolling stats, STL decomposition, correlations |
| `anomaly.py` | z-score (global + rolling), Isolation Forest, anomaly windows |
| `clustering.py` | daily features + K-means day-types |
| `query_engine.py` | **exact** max/min/mean/std/correlation/missing — the truth layer |
| `deep_anomaly.py` | 1D CNN autoencoder (per-point reconstruction error) |
| `deep_lstm.py` | LSTM autoencoder + LSTM forecaster |
| `forecasting.py` | ARIMA forecast-error anomalies (deseasonalised) |
| `reasoning.py` | multi-sensor anomaly explanation (per-sensor z-scores, co-movement) |
| `rag_system.py` | local TF-IDF retriever over sensor KB + trust-aware explainer (LLM-ready) |
| `interface.py` | NL question → query engine → trust-aware answer (computed/retrieved/inferred) |
| `main.py` | end-to-end pipeline entry point |

## Notebooks (`v1/notebooks/`)

| Notebook | Contents |
| --- | --- |
| `01_data_exploration` | load, document, descriptive stats, seasonality |
| `02_evaluation` | query-engine correctness vs pandas, anomaly/cluster checks |
| `03_methods_demo` | query engine, anomaly detection, clustering explained |
| `04_visualizations` | 9 plots (trend, STL, seasonality, diurnal, heatmap, anomalies, etc.) |
| `05_interface_demo` | NL interface, multi-sensor reasoning, labeled evaluation |
| `06_deep_model` | CNN autoencoder train + 3-way anomaly comparison |
| `07_deep_comparison` | CNN-AE vs LSTM-AE vs LSTM-Forecaster |
| `08_model_comparison` | all 6 detectors compared (table + chart) |

## Models implemented

Statistical (z-score) · Forecasting (ARIMA) · Classical ML (Isolation Forest) ·
Deep (CNN-AE, LSTM-AE, LSTM-Forecaster). Full table: **`v1/MODELS.md`**.

## Tests (`v1/tests/`) — 21 passing

`test_query_engine` · `test_interface` (interface + evaluation) · `test_deep_anomaly` ·
`test_deep_lstm` · `test_forecasting`.

## How to run

```bash
cd code/v1
python -m src.main --data ../data/jena_climate_2009_2016.csv --sensor "T (degC)"
python -m pytest tests/
```

## Key docs

- `README.md` — overview + run instructions
- `v1/EVALUATION.md` — evaluation results (correctness, anomalies, labeled metrics)
- `v1/MODELS.md` — model comparison table

## Conventions

- Versioned folders: current work in `v1/`; future iterations in `v2/`, ...
- Commits are authored solely by the user (no AI co-author trailer).
- **Do not push to GitHub** unless the user explicitly asks.
- Data and `.venv` are gitignored.

## Remaining work

1. **Ollama / local LLM** — drop-in explainer (the `rag_system.explain` interface is ready).
2. **Write-up** — methodology, results, limitations.
3. Optional: Matrix Profile (`stumpy`), LOF / One-Class SVM, a multi-sensor deep model.
