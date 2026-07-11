# v1 — Time-Series Analysis Through a Natural Language Interface

A local, explainable system for **temperature/weather time series** (the Jena Climate dataset). You
ask questions in plain English — *"what was the hottest day in 2015?"*, *"were there any anomalies?"*
— and it answers with **exact numbers computed by code**. A local LLM is optional and only *rewords*
those numbers into friendly sentences; it never calculates anything.

> **New here? Start with [GETTING_STARTED.md](GETTING_STARTED.md).** This README is the developer's map.

> **The one idea to remember:** **computation = code (the source of truth); the LLM only explains.**
> This is the deliberate defence against language models making up figures.

---

## What it actually does

1. **Loads & cleans** the raw sensor data (fixes `-9999` error codes, aligns everything to hourly).
2. **Detects anomalies** several ways (statistical, classical ML, forecasting, deep learning) and
   **explains why** a moment looks unusual (e.g. "temperature *and* humidity both jumped").
3. **Answers questions** by routing them to an **exact query engine** (verified to match pandas), then
   optionally handing the answer to a local LLM for nicer phrasing.
4. Ships as a **CLI chat** and a **web app**.

---

## Start here (reading order)

New to the code? Read in this order — each step is small and builds on the last:

1. **This README** — the map (you're here).
2. **`notebooks/00_code_walkthrough.ipynb`** — the whole system as a runnable tour, module by module.
   Open it, *Restart & Run All*, and read top to bottom. **This is the fastest way in.**
3. **`src/query_engine.py`** — the heart: exact statistics, the "truth layer". Small and readable.
4. **`src/interface.py`** — how a plain-English question becomes an answer (intent → query engine).
5. **`src/anomaly.py`** + **`src/reasoning.py`** — how anomalies are found, then explained.
6. Everything else in `src/` as you need it (see the table below).

---

## The code, grouped into layers

The modules in `src/` fall into five layers — data flows top to bottom:

```
  preprocess → analysis            (1. get & describe the data)
        │
        ▼
  anomaly · forecasting · deep      (2. detect anomalies, several methods)
  evaluation · clustering           (   + measure them / group days)
        │
        ▼
  query_engine                      (3. EXACT numbers — the truth layer)
        │
        ▼
  reasoning · rag_system · llm       (4. explain WHY, in words)
        │
        ▼
  interface · agent  →  apps/        (5. question in, answer out;
       (chat, web)                       agent = multi-step tool-use)
```

| Layer | Module | What it does |
| --- | --- | --- |
| **1. Data** | `preprocess.py` | load, clean (`-9999` sentinels), interpolate, align to hourly |
| | `analysis.py` | rolling stats, STL decomposition, correlations |
| **2. Detect** | `anomaly.py` | z-score (global + rolling), Isolation Forest, LOF, One-Class SVM, **Matrix Profile** detectors |
| | `forecasting.py` | ARIMA forecast-error anomalies + a 14-day forecast |
| | `deep.py` | the three deep detectors: CNN-AE, LSTM-AE, LSTM forecaster (GPU-aware) |
| | `evaluation.py` | inject synthetic anomalies → precision/recall/F1/confusion |
| | `clustering.py` | daily features + K-means "day types" |
| **3. Truth** | `query_engine.py` | **exact** max/min/mean/std/correlation/missing — verified `== pandas` |
| **4. Explain** | `reasoning.py` | *why* a moment is anomalous (which sensors moved together) |
| | `rag_system.py` | local retrieval — TF-IDF, **BM25**, **LSA/SVD**, and a **dense sentence-transformer** retriever — + trust-aware template explainer |
| | `retrieval_eval.py` | IR metrics for the retriever (Precision@k, Recall@k, MRR, MAP, NDCG) over labelled + paraphrase query sets |
| | `llm.py` | optional local Ollama layer — rewords facts, never changes numbers; model auto-pick + agentic planning |
| **5. Interface** | `interface.py` | NL question → query engine → trust-aware answer (max/min/mean/trend/compare/threshold/anomaly/forecast/weather/**day-type**/summary) |
| | `agent.py` | **agentic multi-step tool-use** — LLM plans query-engine tool calls; every number still computed, faithfulness-gated |
| | `main.py` | run the whole pipeline once, end to end |

---

## How to run it

Use **Python 3.12** (it has the dependencies and the GPU build of PyTorch).

```bash
cd code/v1
python3.12 -m pip install -r requirements.txt        # one-time setup
```

**Run the whole pipeline once** (loads data, detects anomalies, prints a summary):

```bash
python3.12 -m src.main --data ../data/jena_climate_2009_2016.csv --sensor "T (degC)"
```

**Ask a question in the terminal** (add `--llm` for local-LLM phrasing if Ollama is running):

```bash
python3.12 apps/chat.py "what was the hottest temperature?"
python3.12 apps/chat.py --llm                          # interactive
python3.12 apps/chat.py --agent "compare the hottest of 2013, 2014 and 2015"   # multi-step tool-use
```

> `--agent` plans several query-engine tool calls for a multi-step question. The LLM chooses the
> tools; every number is still computed deterministically and faithfulness-checked. Pull a larger
> model (`ollama pull llama3.1:8b`, or set `TSHCI_MODEL`) and it is used automatically.

**Launch the web app** — serves the chat UI on **http://localhost:3030**:

```bash
python3.12 apps/web/app.py            # open http://localhost:3030
```

It picks the port in this order: **command-line argument → `PORT` env var → 3030**. If that port is
taken it automatically steps up to the next free one (and prints which it used), so the server always
starts:

```bash
python3.12 apps/web/app.py 3050       # force a specific port
PORT=8080 python3.12 apps/web/app.py  # or via the PORT env var
```

Stop it with `Ctrl-C` (or `pkill -f apps/web/app.py` if it's running in the background).

> The browser UI is a small **React** app (React + `htm` vendored under `apps/web/vendor/` — no build
> step, fully offline; just run `app.py`). Charts offer a **Day / Month** resolution toggle, and a
> year-vs-year question draws a **two-line comparison**. `chart.js` holds the unit-tested SVG renderer.

> The local LLM is **optional**. With no Ollama installed, every answer still works — you just get the
> exact computed figures in plain template form instead of LLM-reworded prose.

---

## How to check it works

```bash
python3.12 -m pytest tests/       # 173 passed, 1 skipped (LLM/dense/BM25/MP tests skip if deps absent)
python3.12 verify_results.py      # 29/29 independent checks (results == pandas, day-types, retriever IR metrics, LLM invents 0 numbers)
python3.12 -m ruff check src tests apps    # lint (clean)
python3.12 -m mypy                # lightweight static typing (clean)
```

`verify_results.py` is the trust gate: it re-computes headline results independently and confirms the
LLM never introduced a number that isn't in the data.

---

## Folder map

| Folder / file | What's in it |
| --- | --- |
| `src/` | the code — importable modules (the five layers above) |
| `tests/` | automated tests (run with `pytest`) |
| `notebooks/` | 5 Jupyter notebooks — walkthrough, EDA, methods, interface, deep models (see `notebooks/README.md`) |
| `apps/` | things you **run**: `chat.py` (CLI) and `web/` (browser app) |
| `report/` | things you **read**: the write-up, evaluation, model comparison (see below) |
| `docs/` | design & review notes: `UNDERSTANDING_V1.md`, `ASSESSMENT_AND_RECOMMENDATIONS.md`, `AGENTIC_AND_POLISH.md`, `RETRIEVAL_EVALUATION.md`, `CHAT_MEMORY_PLAN.md` |
| `verify_results.py` | the 29/29 independent verification of results |
| `pyproject.toml` | package metadata + pytest/ruff config (puts `src` on the path from any directory) |

### Docs in `report/`
- `REPORT.md` — the MSc report draft
- `EVALUATION.md` — evaluation results (correctness, anomalies, quantitative rigour)
- `MODELS.md` — the six detectors compared + confusion matrix
- `DATA_QUALITY.md` — data-availability / cleaning analysis (real vs imputed readings)
- `HCI_USER_STUDY.md` — planned user-study protocol (accuracy vs ground truth, SUS, trust)
- `FINETUNING.md` — a side note on fine-tuning the LLM
- `notes/` — working notes (not deliverables): `impro.md` (supervisor feedback), `IMPROVEMENTS.md`,
  `WRITEUP.md` (dissertation skeleton), `RESULTS_DATA.md`

---

## About the data

**Jena Climate 2009–2016** — 420,551 rows at 10-minute resolution → resampled to **70,129 hourly**
rows. **14 sensors** (temperature, humidity, pressure, wind, …); the default target is `T (degC)`.
The wind columns contain `-9999` error codes, which preprocessing cleans. The CSV is gitignored (too
large); download it and place it at `code/data/jena_climate_2009_2016.csv`.
