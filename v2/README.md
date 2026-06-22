# v2 — Time-Series HCI (EU international trade)

Version 2 applies the local, explainable time-series system to a **supply-chain** dataset: EU
monthly international trade (27 countries × exports/imports, 2002–2026). Unlike v1's synthetic
anomalies, this data has **real** ones (2008 crisis, COVID, 2022 energy crisis).

> **Core principle:** code computes the exact numbers (the *truth layer*); the local LLM only
> *explains* them — it never calculates or invents a number.

## Folder map

| Path | What's in it |
| --- | --- |
| `src/` | importable logic (the pipeline modules) |
| `notebooks/` | documented walkthroughs 01–14 (each imports `src`) |
| `apps/web/` | Flask browser chat (`python3.12 apps/web/app.py`) |
| `tests/` | 79 automated tests |
| `report/` | `REPORT.md`, `EVALUATION.md` |
| `GLOSSARY.md` | dataset column meanings (`CC_EXP`/`CC_IMP`) |
| `DEEP_MODELS.md` | deep-learning options (side note, not yet built) |
| `get_trade_data.py` | fetch the main dataset from Eurostat (only step needing internet) |
| `get_bilateral_demo.py` | fetch the EU↔partner bilateral example |

## `src/` modules

| Module | Role |
| --- | --- |
| `preprocess.py` | load, validate, derive balance/total, log-transform |
| `analysis.py` | trend, STL, rolling stats, ADF/ACF, correlations, lead-lag |
| `anomaly.py` | z-score, seasonal-naive, Isolation Forest + eval helpers |
| `reasoning.py` | shared-shock vs isolated classification (multi-country) |
| `forecasting.py` | SARIMA forecast + backtest (predict-vs-memorise) |
| `query_engine.py` | **exact** stats — the truth layer (verified vs pandas) |
| `interface.py` | NL question → query engine → trust-aware answer |
| `llm.py` | optional local Ollama explainer (phrases only, never computes) |
| `rag_system.py` | hybrid retrieval (BM25 + LSA/BERT semantic, chunking), context only |
| `clustering.py` | group countries by trade behaviour (K-means) |
| `evaluation.py` | point-adjusted metrics, ensemble, PR curves |
| `deep_anomaly.py` | multivariate autoencoder (GPU; benchmarked vs baseline) |

## Notebooks (reading order)

`01_eda` · `02_preprocess` · `03_analysis` · `04_anomaly` · `05_reasoning` ·
`06_forecasting` · `07_query_engine` · `08_interface` · `09_rag` · `10_evaluation` ·
`11_clustering` · `12_deep_model` · `13_chunking_bert` · `14_evaluation_upgrade`.

Each has markdown explanations (What / Why / How to read) + commented code, and is executed with
outputs saved.

## How to run

Use **Python 3.12** (deps live there).

```bash
# from this folder (v2/)
python3.12 get_trade_data.py          # one-time: fetch the dataset (needs internet)
python3.12 -m pytest tests/           # 60 tests
# notebooks: open with the Python 3 kernel, Restart & Run All
```

## Data

`../data/eu_trade_monthly.csv` — 292 months × 54 columns (gitignored). Re-create it with
`get_trade_data.py`. See `GLOSSARY.md` for column meanings.
