# time_series_hci

**Time-Series Data Analysis Through a Natural HCI Interface** — a local, explainable, interactive
time-series analytics system. Exact numbers are computed by deterministic Python code; a local LLM
only *explains* them.

> **Core principle:** Computation = code (the source of truth). LLM = explanation only.

## Versioning

Each iteration of the system lives in its own version folder, so progress is snapshotted and older
versions remain runnable:

- **`v1/`** — first working pipeline (current)
- `v2/`, `v3/`, … — later iterations (added as the project evolves)

Each version folder contains:

```
vN/
├── src/         # pipeline modules
├── notebooks/   # exploration & figures
├── tests/       # unit tests (esp. query engine)
└── requirements.txt
```

## Pipeline (src/)

`preprocess` → `analysis` → `anomaly` → `clustering` → `query_engine` → `rag_system` → `main`

| Module | Role |
| --- | --- |
| `preprocess.py` | load, clean, align, interpolate |
| `analysis.py` | behaviour features & statistics |
| `anomaly.py` | z-score / Isolation Forest / 1D CNN |
| `clustering.py` | K-means / hierarchical |
| `query_engine.py` | **exact** computations — the truth layer |
| `rag_system.py` | context retrieval + local LLM explanation *(later)* |
| `main.py` | pipeline orchestration / entry point |

## What v1 implements

Built and evaluated:
- **Preprocessing** — load, clean (`-9999` sentinels), align to hourly
- **Behaviour analysis** — rolling stats, STL decomposition, correlations
- **Anomaly detection** — z-score, **ARIMA** (forecasting), Isolation Forest, and three deep models (**1D CNN AE, LSTM AE, LSTM forecaster**)
- **Clustering** — K-means day-types
- **Query engine** — exact stats (the truth layer)
- **Interface** — natural-language questions → trust-aware answers (computed/retrieved/inferred)
- **RAG retriever** — local TF-IDF over a sensor knowledge base
- **Multi-sensor reasoning** — explains anomalies via per-sensor z-scores
- **Evaluation** — vs pandas ground truth + labeled (synthetic-anomaly) precision/recall

Deferred to the write-up stage: the **local LLM (Ollama)** for phrasing explanations.

Run it:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r v1/requirements.txt    # includes torch (CPU)
cd v1
python -m src.main --data ../data/jena_climate_2009_2016.csv --sensor "T (degC)"
pytest tests/                          # 16/16 tests
```

Notebooks (`v1/notebooks/`): `01_data_exploration` · `02_evaluation` · `03_methods_demo`
· `04_visualizations` · `05_interface_demo` · `06_deep_model` · `07_deep_comparison`
· `08_model_comparison`.
Docs: [`v1/EVALUATION.md`](v1/EVALUATION.md) · [`v1/MODELS.md`](v1/MODELS.md) (model comparison)
· [`CLAUDE.md`](CLAUDE.md) (context & work index).
