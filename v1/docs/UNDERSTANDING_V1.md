# Understanding v1 — design, architecture, running, packages, methodology

A guided explanation of the v1 system (temperature / Jena climate). This is a *learning* document
for understanding the code — the formal write-up lives in `../report/`.

---

## 1. The core design idea (the "why")

Everything in v1 serves one principle:

> **Code computes the numbers. The LLM only explains them. And we automatically verify the LLM
> didn't lie.**

This answers the biggest problem with using LLMs for data analysis: **they hallucinate numbers.**
If you ask a raw LLM "what was the hottest temperature?", it might confidently invent `38.2°C`. In
v1 that can never happen, because:

1. A deterministic **query engine** computes `37.04°C` from the actual data (exact, checkable
   against pandas).
2. The LLM receives that number as a *fact it may not change*, and only rephrases it into a
   friendly sentence.
3. A **faithfulness check** scans the LLM's sentence for any number that isn't in the computed
   facts — if it finds one, the LLM output is thrown away and a template is used instead.

The report calls this pattern **"Verifiable NL Analytics"** — it is the intellectual contribution.
Every architectural decision below flows from it.

---

## 2. The architecture — five layers

The 13 modules in `src/` group into five layers. Data flows **downward** when answering a question:

```
                    ┌─────────────────────────────────────────────┐
  YOU ask a         │  INTERFACE   interface_parsing.py            │
  question  ───────▶│              interface.py  (orchestrator)    │
                    │              main.py (headless pipeline)     │
                    └───────────────────────┬─────────────────────┘
                                            │ intent + sensor + period
                    ┌───────────────────────▼─────────────────────┐
                    │  EXPLAIN     llm.py (Ollama + faithfulness)  │
                    │              rag_system.py (context+template)│
                    │              reasoning.py (multi-sensor why) │
                    └───────────────────────┬─────────────────────┘
                                            │ asks for the exact number
                    ┌───────────────────────▼─────────────────────┐
                    │  TRUTH       query_engine.py  ◀── the source │
                    │              evaluation.py       of truth    │
                    └───────────────────────┬─────────────────────┘
                                            │ operates on detected signals
                    ┌───────────────────────▼─────────────────────┐
                    │  DETECT      anomaly.py  deep.py             │
                    │              forecasting.py  clustering.py   │
                    │              analysis.py (STL, correlations) │
                    └───────────────────────┬─────────────────────┘
                                            │ runs on clean data
                    ┌───────────────────────▼─────────────────────┐
                    │  DATA        preprocess.py (load/clean/align)│
                    └─────────────────────────────────────────────┘
```

### What each file does

| Layer | File | Responsibility |
|---|---|---|
| **Data** | `preprocess.py` | Load the CSV, convert the `-9999` sentinel error codes to NaN, interpolate gaps (time-aware), resample 10-min → hourly. |
| **Detect** | `analysis.py` | STL decomposition (splits a signal into trend + daily cycle + residual), correlations, **circular statistics** for wind direction. |
| | `anomaly.py` | The detectors: z-score, rolling z-score, Isolation Forest, LOF, One-Class SVM + borderline "near-miss" flagging. |
| | `deep.py` | Three neural detectors: CNN autoencoder, LSTM autoencoder, LSTM forecaster (reconstruction/forecast error = anomaly). |
| | `forecasting.py` | ARIMA — both anomaly detection (forecast error) and a 14-day forecast with an 80% band + backtesting. |
| | `clustering.py` | K-means on daily feature vectors → "day-types" (what kind of day was this?). |
| **Truth** | `query_engine.py` | **The heart.** Exact max/min/mean/std/correlation/missing/threshold. Every number the user sees comes from here. |
| | `evaluation.py` | Scoring: precision/recall/F1, point-adjusted metrics, bootstrap confidence intervals, McNemar significance test. |
| **Explain** | `rag_system.py` | TF-IDF retriever over sensor descriptions + the template explainer with 🔢/📚/🧠 **trust labels**. |
| | `reasoning.py` | "Why was this moment weird?" — z-scores every sensor at a timestamp, reports which moved together. |
| | `llm.py` | Talks to local Ollama; `plan` (parse), `explain` (phrase), `faithfulness` (the anti-hallucination gate). |
| **Interface** | `interface_parsing.py` | Pure text→structure: which sensor, which intent, which time period — regex/keyword only, no model. |
| | `interface.py` | The orchestrator: ties parsing → query engine → context → phrasing → trust-aware `Answer`. |
| | `main.py` | Headless entry point — runs the whole pipeline and prints headline numbers. |
| **Apps** | `apps/chat.py` | Command-line chat. |
| | `apps/web/app.py` | Flask web chat (the browser UI). |

### The journey of one question

When you ask *"what was the hottest temperature in summer 2013?"*:

1. **`interface_parsing`** extracts: intent=`max`, sensor=`T (degC)`, period=`Jun–Aug 2013`.
2. **`interface`** slices the data to that window and calls...
3. **`query_engine.get_max()`** → returns `Extremum(value=33.7, when=2013-07-27 15:00)` — the
   *exact* number.
4. **`rag_system`** attaches context ("T is air temperature in °C…").
5. **`llm.explain()`** (if Ollama is on) rephrases: *"The hottest it got in summer 2013 was 33.7°C,
   on 27 July at 3pm."*
6. **`llm.faithfulness()`** checks that `33.7` and the date are really in the computed facts — if
   the model had said `34°C`, it is rejected.
7. You get an `Answer` object with the text, the trust labels, and a chart payload.

---

## 3. How to run it (and learn each file by doing)

```bash
cd code/v1

# 0. One-time: install deps + place the dataset
pip install -r requirements.txt
#    put jena_climate_2009_2016.csv in code/data/  (it's gitignored)

# 1. Headless pipeline — the simplest end-to-end run (read main.py alongside)
python3.12 -m src.main --data ../data/jena_climate_2009_2016.csv --sensor "T (degC)"

# 2. Ask questions on the command line (read apps/chat.py + interface.py)
python3.12 apps/chat.py "what was the hottest temperature?"
python3.12 apps/chat.py --llm "how warm did it get in summer 2013?"   # with Ollama phrasing

# 3. The web UI (read apps/web/app.py + web/index.html)
python3.12 apps/web/app.py        # prints http://localhost:3030

# 4. The tests — the best way to see what each module *promises* to do
python3.12 -m pytest                       # all 120
python3.12 -m pytest tests/test_query_engine.py -v   # one module

# 5. The independent proof the numbers are right
python3.12 verify_results.py       # 25/25

# 6. One-command reproduce (seeded tests + verification banner)
bash reproduce.sh
```

**Best reading order to understand the code:** `README.md` → `notebooks/00_code_walkthrough.ipynb`
(runs every module with examples) → `query_engine.py` (the truth layer, only 142 lines) →
`interface.py` (how it all connects) → then the detectors. The **tests** are the second-best
documentation: each one is a small, runnable statement of what a function should do.

---

## 4. The packages — what and *why*

| Package | Used for | Why this one |
|---|---|---|
| **pandas** | All data handling — the `DataFrame`, time-indexing, `.loc[start:end]` slicing, `.corr()`, `.resample()`. | It *is* the standard for tabular time-series in Python; the whole "truth layer" is thin wrappers over pandas so results are trivially checkable. |
| **numpy** | Numeric arrays under pandas; the circular-statistics math (`sin/cos/arctan2`), bootstrap resampling. | Foundation of the scientific stack; fast vectorised math. |
| **scikit-learn** | Isolation Forest, LOF, One-Class SVM (classical anomaly detectors), K-means (clustering), `StandardScaler`, and TF-IDF for the RAG retriever. | One consistent, well-tested API for all the classical ML — no need to hand-roll detectors. |
| **statsmodels** | **STL** decomposition (separating the daily cycle) and **ARIMA/SARIMAX** (forecasting + forecast-error anomalies). | The reference library for classical time-series statistics; STL and ARIMA are its core strengths. |
| **scipy** | The **McNemar** significance test (`binomtest`), distribution stats. | Standard statistics; needed for the rigor (is detector A *significantly* better than B?). |
| **torch** (PyTorch) | The three deep detectors (CNN-AE, LSTM-AE, LSTM-forecaster). | The mainstream deep-learning framework; imported *lazily* so the rest of the system works without it. |
| **matplotlib** | Plots in the notebooks. | Standard plotting. |
| **flask** | The web chat server. | Minimal, no-ceremony web framework — right size for a single-user local app. |
| **pytest** | The 120 automated tests. | Standard Python test runner. |
| **ruff** | Linting. | Fast, modern all-in-one linter/formatter. |
| **Ollama** | The local LLM (llama3.2:3b). | *Not a Python dependency* — `llm.py` talks to it over plain HTTP with the **standard library only**. |

Two deliberate choices worth calling out:

- **The LLM is local (Ollama) and accessed via stdlib HTTP, not an API SDK.** This means no cloud,
  no API key, no data leaving the machine, and no heavy dependency. It fits the "trust" theme —
  everything is inspectable and offline.
- **`pyproject.toml` holds loose version bounds; `requirements.txt` holds exact pins.** Standard
  best practice: the project *depends on* "pandas ≥ 2.2" abstractly, but the *reproducible tested
  environment* is pinned exactly (including the CUDA-12.8 torch build for the RTX 5070).

---

## 5. The methodology

The project is built on a few disciplined practices — these are what make it defensible in a viva:

1. **Test-Driven Development.** Behaviour is specified by a failing test *first*, then code makes it
   pass.
2. **An independent verification harness** (`verify_results.py`, separate from the tests). It
   recomputes answers with raw pandas and asserts the query engine matches — 25 checks. This is the
   evidence that "exact numbers" is a real claim, not a slogan.
3. **Honest anomaly evaluation, two ways.** Synthetic spike injection gives *exact* labels (but
   artificial); a list of *real documented weather events* (2015 heatwave, etc.) gives *real* labels
   (but coarse). Reporting both is honest about the fact that Jena has no ground-truth anomaly
   labels.
4. **Statistical rigor, not point estimates.** Every headline metric comes with a **bootstrap 95%
   confidence interval**, and detector comparisons use a **McNemar significance test** — so
   "recall = 0.78" becomes "0.78, 95% CI [0.64, 0.90]".
5. **Reproducibility.** Fixed seeds everywhere, pinned dependencies, `reproduce.sh`.
6. **Trust made visible.** Every answer is labelled 🔢 computed / 📚 context / 🧠 inferred, and the
   LLM's output is gated by the automated faithfulness check (majority-of-3 to tame the small
   model's randomness).

---

## 6. How the results change when several things change at once

Separate the changes into **two kinds** — because most reorganisation *deliberately does not change
any number*.

### A. Behaviour-preserving changes → results are byte-for-byte identical

These reorganise the project without touching computation, so **every test still passes with the
same values and `verify_results` stays 25/25**:

- Adding `pyproject.toml`, removing `conftest.py`, cleaning `__pycache__`
- Moving working-note docs into `report/notes/`
- Lint fixes (`zip(strict=True)`, import ordering, `def` vs `lambda`, `seeds=None` default)
- Docstring/comment corrections

The proof they are safe *is* that the 120 tests + 25 checks produce the same output before and after.
That is the whole point of having them.

### B. Behaviour-changing changes → results actually move

| Change | Effect on results |
|---|---|
| **LSTM seeding fix** | *Before:* two runs of the LSTM detectors gave **different** anomaly scores (unseeded weights). *After:* identical every run. The scores are the same *kind* of numbers, but now **deterministic** — the reproducibility claim holds. Test count 118 → **120** (added 2 tests). |
| **Verification harness grew** | `17/17` → **`25/25`** — not because old checks changed, but because bootstrap-CI, McNemar, and circular-wind checks were *added*. More things are now proven. |
| **Point-adjusted metrics** | Switching from strict to the standard point-adjusted metric moved recall dramatically: seasonal-naive **0.45 → 0.90**, Isolation Forest **0.36 → 0.76**. Same detectors, fairer scoring rule for event-scale anomalies. |
| **Circular wind fix** | "Maximum wind direction" went from a meaningless **359.8°** (linear max on an angle) to a correct prevailing **196.6° (S)** (circular mean). The *old* number was wrong; the new one is right. |
| **LLM faithfulness gate** | Any hallucinated figure is now discarded, so measured invented-numbers = **0**. Without it, the small model occasionally slipped a wrong number through. |

**The key mental model:** the system is layered precisely so that changes are *contained*.
Reorganising the packaging or docs (layer plumbing) cannot alter a computed number, because the
number comes from `query_engine.py` operating on the data — and the tests + harness are the tripwires
that would fire instantly if anything leaked across that boundary. The only time a *result* changes
is when you change a *detector, a metric definition, or a statistic* — and each of those changes is
isolated to one module and covered by its own test.
