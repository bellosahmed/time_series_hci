# Getting Started with v2 — EU Trade, in Plain English

Welcome. This is the friendly on-ramp. If you have never seen this project before, read this
first; the [README](README.md) is the developer's map and goes deeper once you're up and running.

## What is this?

v2 is a small, local, **explainable** analytics system for **EU monthly international trade** — 27
countries, exports and imports, from 2002 to 2026, in millions of euros. You ask questions in plain
English ("What was France's highest export month?") and get back **exact numbers computed by code**.
The reason this dataset was chosen is that it carries **real, datable shocks** — the 2008 financial
crisis, COVID-19 in 2020, the 2022 energy crisis — so the anomaly detectors can be tested against
events that genuinely happened, rather than spikes made up for a demo. The single idea to hold on to:
**the numbers always come from code; the language model, if you use one, only rewords them — it never
calculates or invents a figure.**

## What you need before you start

- **Python 3.12.** Everything is run with `python3.12`. Check it with `python3.12 --version`.
- **An internet connection — once.** You need it for the one-time data fetch from Eurostat (below).
  After that, the whole system runs fully offline.
- **A local LLM — entirely optional.** If [Ollama](https://ollama.com) is installed and running, the
  app can reword answers into fuller prose. Without it, everything still works — you simply see the
  exact computed figures in plain form. Nothing about correctness depends on it.

## Install

Install the dependencies once, from the `code/v2` directory:

```bash
cd code/v2
python3.12 -m pip install -r requirements.txt
```

This pulls in pandas, scikit-learn, statsmodels, Flask, PyTorch and a few others. The first run also
downloads a small sentence-embedding model (`all-MiniLM-L6-v2`) for retrieval.

## Get the data

The trade data is **not shipped with the project** (it's gitignored). Fetch it once — **this step
needs the internet**:

```bash
python3.12 tools/get_trade_data.py
```

This calls the Eurostat API and writes the CSV to:

```
code/data/eu_trade_monthly.csv        (292 months × 54 country-flow columns)
```

Note the path is `../data/` relative to `v2` — the data lives one level up, shared with the other
versions. **The app cannot start without this file**, so if you skip this step you'll see a
file-not-found error the moment you launch it.

## Run it — the fastest path

v2 is **web-only**: the one and only app is a browser chat. Launch it:

```bash
python3.12 apps/web/app.py
```

It prints a line like `v2 trade chat → http://localhost:3000`. Open that URL in your browser, type a
question (or click one of the suggested ones), and read the answer. That's the 60-second win.

- The app serves on **http://localhost:3000** by default. If that port is busy it automatically tries
  the next free one and prints whichever it actually used — so always trust the printed URL.
- To pin a port yourself: `python3.12 apps/web/app.py 3050` or `PORT=8080 python3.12 apps/web/app.py`.
- Stop the app with **Ctrl-C**.

## Try these questions

These all map to intents the interface actually understands. Copy any of them into the chat box:

```text
What was France's highest export month?
What was Spain's worst monthly export drop?
Which countries were abnormal in 2020-04?
How much did Poland's exports grow from 2010 to 2024?
Are Germany and France exports correlated?
Who had the biggest trade surplus in 2024?
Top exporters in 2024?
Summarise the Netherlands.
```

The system recognises countries by name (e.g. "France", "Germany") or by their two-letter code
(FR, DE). A question needs a country for most intents; if you leave it out, the app will politely ask
which country you mean.

## What you're looking at

- **The computed answer is the truth.** Every figure you see was calculated by the query engine and
  is verified to match pandas exactly.
- **The LLM only rephrases.** If you have Ollama on and enable it, you'll additionally get a
  natural-language version of the same facts. Turn it off and nothing is lost except the prose.
- **Shared-shock reasoning.** Ask about anomalies in a crisis month (e.g. `2020-04`) and the system
  doesn't just flag numbers — it reasons about whether *many* countries dropped together (a
  region-wide "shared shock", such as a global crisis) versus a single country having an isolated
  blip.
- **Trust labels.** Answers distinguish what was **computed** (the exact facts) from what was
  **inferred** (a labelled interpretation), so you always know which is which.

## Explore the notebooks

The 14 notebooks in `notebooks/` are the guided tour — documented, executed, and meant to be read top
to bottom. Open them with a **Python 3 kernel** and use **Restart & Run All**. Start at the beginning
and follow the numbers:

```
01_eda → 02_preprocess → 03_analysis → 04_anomaly → … → 14_evaluation_upgrade
```

`01_eda` introduces the dataset; `04_anomaly` is the core of the project (anomaly detection on the
real crises).

## Troubleshooting

- **"File not found" / the app won't start.** You haven't fetched the data yet, or you have no
  internet for the fetch. Run `python3.12 tools/get_trade_data.py` (needs a connection) and confirm
  `code/data/eu_trade_monthly.csv` exists.
- **Port 3000 is already taken.** Port 3000 is also [Open WebUI](https://openwebui.com)'s default, so
  a clash is common. The app handles this itself by falling back to the next free port — just read
  the URL it prints. To force a specific port: `python3.12 apps/web/app.py 3050`.
- **Answers aren't in full sentences / "Ollama: off".** That's expected when no local LLM is running.
  The exact figures are all there in plain form. Install and start Ollama (e.g. `llama3.2:3b`) if you
  want the reworded prose — but it changes nothing about the numbers.
- **`python3.12: command not found` or wrong version.** This project is pinned to Python 3.12. A
  different interpreter (including a 3.14 environment here that has no PyTorch) will fail to import
  the dependencies. Use `python3.12` explicitly.
- **"Which country?" replies.** Most questions need a country. Add one by name or two-letter code
  (e.g. "Summarise **Italy**" or "IT").

## Verify the install

Two commands confirm everything is wired up correctly:

```bash
python3.12 -m pytest tests/      # 79 passed
python3.12 verify_results.py     # 13/13 independent checks
```

`verify_results.py` is the trust gate: it independently re-computes the headline results and confirms
the LLM never introduced a number that isn't in the data.

## Where to go next

- **[README.md](README.md)** — the developer's map: the five code layers and how they fit together.
- **`notebooks/`** — the full guided tour, `01_eda` onward.
- **`GLOSSARY.md`** — what each column means (e.g. `FR_EXP`, `DE_IMP`).
- **`report/`** — `REPORT.md`, `EVALUATION.md` and `VERIFICATION.md` for the written-up methods,
  results and checks.
