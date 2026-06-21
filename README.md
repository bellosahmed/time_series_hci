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

The **computational core** is built and evaluated: preprocessing, behaviour analysis
(rolling stats, STL decomposition, correlations), anomaly detection (residual z-score +
Isolation Forest), clustering (day-types), and the exact **query engine**. RAG + LLM and
the interface are deliberately deferred.

Run it:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r v1/requirements.txt
cd v1
python -m src.main --data ../data/jena_climate_2009_2016.csv --sensor "T (degC)"
pytest tests/          # 8/8 query-engine correctness tests
```

Notebooks: `v1/notebooks/01_data_exploration.ipynb`, `v1/notebooks/02_evaluation.ipynb`.
Evaluation summary: [`v1/EVALUATION.md`](v1/EVALUATION.md) — query engine **6/6 exact** vs pandas.
