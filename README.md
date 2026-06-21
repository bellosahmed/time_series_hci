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
| `rag_system.py` | context retrieval + local LLM explanation |
| `main.py` | pipeline orchestration / entry point |
