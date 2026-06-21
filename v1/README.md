# v1 — Time-Series HCI

Version 1 of the local, explainable time-series analytics system. Exact numbers are computed by
deterministic code (the **query engine**); a local LLM only *explains* them.

## Folder map

| Folder | What's in it |
| --- | --- |
| `src/` | the code — importable Python modules (the logic) |
| `tests/` | automated tests (24 total) |
| `notebooks/` | Jupyter notebooks — exploration, evaluation, demos (see `notebooks/README.md`) |
| `apps/` | things you **run**: `chat.py` (CLI) and `web/` (browser app) |
| `report/` | things you **read**: REPORT, EVALUATION, MODELS, FINETUNING |

## The `src/` modules

| Module | Role |
| --- | --- |
| `preprocess.py` | load, clean (`-9999` sentinels), interpolate, resample to hourly |
| `analysis.py` | rolling stats, STL decomposition, correlations |
| `anomaly.py` | z-score (global + rolling) and Isolation Forest detectors |
| `forecasting.py` | ARIMA forecast-error anomalies |
| `deep_anomaly.py` | 1D CNN autoencoder (GPU-aware) |
| `deep_lstm.py` | LSTM autoencoder + LSTM forecaster |
| `clustering.py` | daily features + K-means day-types |
| `query_engine.py` | **exact** stats — the trusted source of numbers |
| `reasoning.py` | multi-sensor explanation of anomalies |
| `rag_system.py` | local TF-IDF context retriever + trust-aware template explainer |
| `llm.py` | optional local Ollama explainer (rephrases facts; never changes numbers) |
| `interface.py` | natural-language question → query engine → trust-aware answer |
| `evaluation.py` | synthetic-anomaly injection + precision/recall/F1/confusion |
| `main.py` | end-to-end pipeline entry point |

> Want to *understand* the code by reading and running it? Open **`notebooks/00_code_walkthrough.ipynb`**.

## How to run

Use **Python 3.12** (it has the dependencies + GPU).

```bash
# from this folder (v1/)
python3.12 -m pip install -r requirements.txt   # one-time

# run the whole pipeline once
python3.12 -m src.main --data ../data/jena_climate_2009_2016.csv --sensor "T (degC)"

# run the tests
python3.12 -m pytest tests/

# ask questions in the terminal
python3.12 apps/chat.py --llm "what was the hottest temperature?"

# launch the website (prints the URL, e.g. http://localhost:3030)
python3.12 apps/web/app.py
```

## Docs (`report/`)
- `REPORT.md` — the MSc report draft
- `EVALUATION.md` — evaluation results (correctness, anomalies, labeled metrics)
- `MODELS.md` — model comparison + confusion matrix
- `FINETUNING.md` — side note on fine-tuning the LLM
