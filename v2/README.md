# v2 — Time-Series Analysis Through a Natural Language Interface (EU trade)

The same local, explainable system as v1, applied to a **very different domain**: EU monthly
**international trade** (27 countries × exports/imports, 2002–2026). You ask questions in plain
English and get **exact numbers computed by code**; a local LLM is optional and only *rewords* them.

The point of v2 is that this dataset has **real, datable anomalies** — the 2008 financial crisis,
COVID-19, the 2022 energy crisis — so the detectors can be tested against events that actually
happened, not synthetic spikes.

> **The one idea to remember:** **computation = code (the source of truth); the LLM only explains.**
> It never calculates or invents a number.

---

## What it actually does

1. **Fetches & preprocesses** the trade data (Eurostat), deriving trade balance/total and
   log-transforming.
2. **Detects anomalies** (statistical, seasonal-naive, Isolation Forest, a multivariate autoencoder)
   and **explains why** — e.g. "many countries dropped together in 2020 → a *shared shock*, not an
   isolated blip."
3. **Answers questions** via an **exact query engine** (verified to match pandas), optionally reworded
   by a local LLM.
4. Ships as a **web app**, with 14 documented notebooks as the guided tour.

---

## Start here (reading order)

New to the code? Read in this order — each step is small and builds on the last:

1. **This README** — the map (you're here).
2. **`notebooks/01_eda` → `04_anomaly`** — the dataset, then **the core of the project** (anomaly
   detection on real crises). Open each, *Restart & Run All*, read top to bottom.
3. **`src/query_engine.py`** — the heart: exact statistics, the "truth layer".
4. **`src/interface.py`** — how a plain-English question becomes an answer (intent → query engine).
5. **`src/anomaly.py`** + **`src/reasoning.py`** — how anomalies are found, then explained as
   shared shocks across countries.
6. Everything else in `src/` as you need it (see the table below).

---

## The code, grouped into layers

The 12 modules in `src/` fall into five layers — data flows top to bottom:

```
  preprocess → analysis                 (1. get & describe the data)
        │
        ▼
  anomaly · forecasting · deep_anomaly    (2. detect anomalies, several methods)
  evaluation · clustering                 (   + measure them / group countries)
        │
        ▼
  query_engine                          (3. EXACT numbers — the truth layer)
        │
        ▼
  reasoning · rag_system · llm            (4. explain WHY, in words)
        │
        ▼
  interface  →  apps/web (browser chat)  (5. question in, answer out)
```

| Layer | Module | What it does |
| --- | --- | --- |
| **1. Data** | `preprocess.py` | load, validate, derive balance/total, log-transform |
| | `analysis.py` | trend, STL, rolling stats, ADF/ACF, correlations, lead-lag |
| **2. Detect** | `anomaly.py` | z-score, seasonal-naive, Isolation Forest (+ eval helpers) |
| | `forecasting.py` | SARIMA forecast + backtest (does it predict, or just memorise?) |
| | `deep_anomaly.py` | multivariate autoencoder (GPU-aware), benchmarked vs the baselines |
| | `evaluation.py` | point-adjusted metrics, ensemble, precision-recall curves |
| | `clustering.py` | group countries by their trade behaviour (K-means) |
| **3. Truth** | `query_engine.py` | **exact** stats — the truth layer, verified `== pandas` |
| **4. Explain** | `reasoning.py` | shared-shock vs isolated classification across countries |
| | `rag_system.py` | hybrid retrieval (BM25 + LSA/BERT semantic), context only — fully offline |
| | `llm.py` | optional local Ollama layer — rewords facts, never computes |
| **5. Interface** | `interface.py` | natural-language question → query engine → trust-aware answer |

---

## How to run it

Use **Python 3.12** (it has the dependencies and the GPU build of PyTorch).

```bash
cd code/v2
python3.12 -m pip install -r requirements.txt          # one-time setup
python3.12 tools/get_trade_data.py                     # one-time: fetch the data (needs internet)
```

**Launch the web app** (prints a URL, e.g. http://localhost:3040):

```bash
python3.12 apps/web/app.py
```

**Explore the notebooks:** open any `notebooks/NN_*.ipynb` with the Python 3 kernel and *Restart &
Run All*. Start at `01_eda` and follow the numbers.

> The local LLM is **optional**. With no Ollama installed, every answer still works — you just get the
> exact computed figures in plain form instead of LLM-reworded prose.

---

## How to check it works

```bash
python3.12 -m pytest tests/       # 79 passed
python3.12 verify_results.py      # 13/13 independent checks (results == pandas, LLM invents 0 numbers)
```

`verify_results.py` is the trust gate: it re-computes headline results independently and confirms the
LLM never introduced a number that isn't in the data.

---

## Folder map

| Folder / file | What's in it |
| --- | --- |
| `src/` | the code — importable modules (the five layers above) |
| `tests/` | automated tests (run with `pytest`) |
| `notebooks/` | 14 documented walkthroughs `01_eda … 14_evaluation_upgrade` (each imports `src`) |
| `apps/web/` | Flask browser chat |
| `tools/` | developer scripts: `get_*.py` (fetch data from Eurostat) and `build_*_notebook.py` (regenerate the notebooks) |
| `report/` | `REPORT.md`, `EVALUATION.md`, `VERIFICATION.md` — things you **read** |
| `GLOSSARY.md` | dataset column meanings (`CC_EXP` / `CC_IMP`) |
| `DEEP_MODELS.md` | deep-learning design notes |
| `verify_results.py` | the 13/13 independent verification of results |
| `conftest.py` | lets `pytest` find `src` from any directory (no config needed) |

---

## About the data

`../data/eu_trade_monthly.csv` — **292 months × 54 columns** (27 EU countries × exports/imports),
2002–2026, in € millions, not seasonally adjusted (so real seasonality and the 2008/2020/2022 shocks
survive). Re-create it any time with `python3.12 tools/get_trade_data.py`. The CSV is gitignored; see
`GLOSSARY.md` for what each column means.
