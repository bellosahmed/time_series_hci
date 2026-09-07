# Getting Started (v1 — Temperature / Jena Climate)

Welcome. This is the friendly on-ramp for someone who has never seen this project. It walks you
from a blank machine to your first answer. (For the developer's map of the code, see
[README.md](README.md) once you're up and running.)

## What is this?

You ask a plain-English question about eight years of weather data — *"what was the hottest
temperature?"*, *"were there any unusual readings in 2014?"* — and the system answers with **exact
numbers computed by code**. Every figure comes from deterministic Python (verified to match pandas);
an optional local LLM only *rewords* those numbers into friendlier sentences and is checked so it
can never invent a figure. You can use it from the terminal or from a small browser chat app.

> **The one idea to remember:** numbers are always computed by code. The LLM only explains — it never
> calculates.

## What you need before you start

A short checklist:

- **Python 3.12** — the project is pinned to it (`python3.12`). Check with `python3.12 --version`.
  A different version may not have a compatible PyTorch build. See Troubleshooting if `python3.12`
  isn't found.
- **The data file** — the Jena Climate CSV. It is *not* shipped with the code (too large, and
  gitignored). You must download it and drop it in place — see **Get the data** below. Nothing loads
  without it.
- **A local LLM — OPTIONAL.** [Ollama](https://ollama.com) with a small model. You do **not** need
  it: every answer works without it, you simply get the exact figures in plain template form instead
  of reworded prose. Add it later if you want nicer phrasing.

Everything runs **locally and offline**. There are no cloud API keys or accounts to set up.

## Install

From the `code/v1` directory, install the dependencies once — pick the file that matches your
hardware, both install the same package versions:

```bash
cd code/v1
python3.12 -m pip install -r requirements.txt        # default — CPU, no GPU needed
python3.12 -m pip install -r requirements-gpu.txt     # NVIDIA GPU (CUDA 12.8 torch build) instead
```

This pulls numpy, pandas, scikit-learn, statsmodels, matplotlib, PyTorch, and Flask. The deep
models (`src/deep.py`) run fine on CPU, just slower — the core Q&A doesn't need a GPU at all.

## Get the data

The system runs on the **Jena Climate 2009–2016** dataset — the Kaggle "Jena Climate" dataset:
420,551 rows at 10-minute resolution (resampled to ~70,129 hourly rows), with 14 weather sensors
(temperature, humidity, pressure, wind, …).

Download `jena_climate_2009_2016.csv` and place it at exactly this path:

```
code/v1/data/jena_climate_2009_2016.csv
```

That is inside `v1`'s own `data/` folder. If the file is missing or in the wrong place, the app
fails to load — see Troubleshooting.

## Run it — the fastest path

Two ways to get your first answer. Pick one.

**A) Ask a single question in the terminal (fastest):**

```bash
python3.12 apps/chat.py "what was the hottest temperature?"
```

It prints the loading step, then the answer with its trust labels. That's the 60-second win.

**B) Launch the web app:**

```bash
python3.12 apps/web/app.py
```

It prints a URL — open **http://localhost:3030** in your browser and type a question. Stop the
server with `Ctrl-C`. The UI is a small vendored React app with no build step, fully offline; it
draws a chart alongside each answer.

For a chatty terminal session instead of one-shot, run interactive mode (add `--llm` to turn on LLM
phrasing when Ollama is available):

```bash
python3.12 apps/chat.py             # interactive, template phrasing
python3.12 apps/chat.py --llm       # interactive, LLM rewording if Ollama is up
```

## Try these questions

All of these map to real intents the interface understands (extremes, averages, spread,
correlation, missing data, trend, comparison, threshold counts, anomalies, weather day-reports,
day-types, and forecasts). Scope them to a year, month, season, day, or range.

```bash
python3.12 apps/chat.py "what was the hottest day in 2015?"
python3.12 apps/chat.py "were there any unusual readings in 2014?"
python3.12 apps/chat.py "was 2013 hotter than 2014?"
python3.12 apps/chat.py "how often was it above 35 degrees?"
python3.12 apps/chat.py "forecast the temperature for next week"
python3.12 apps/chat.py "tell me the weather on 26 January 2015"
python3.12 apps/chat.py "how does temperature relate to air density?"
python3.12 apps/chat.py "what kinds of day are there?"
```

For a multi-step question, use the agent (it plans several query-engine calls; every number is still
computed and faithfulness-checked, and it implies `--llm`):

```bash
python3.12 apps/chat.py --agent "compare the hottest of 2013, 2014 and 2015"
```

## What you're looking at

- **Trust labels.** Each answer tags where its information came from:
  - 🔢 **Computed** — exact, straight from the query engine (the source of truth).
  - 📚 **Context** — background retrieved from the sensor knowledge base.
  - 🧠 **Inferred** — light reasoning on top of the computed values.
- **The LLM is optional.** With no Ollama running you get the exact figures in a plain template. With
  `--llm` (and Ollama up), the same figures are reworded into prose — and a faithfulness check
  discards any wording that introduces a number not in the computed facts.
- **Charts (web app).** Answers come with a chart. There's a **Day / Month** resolution toggle, and a
  year-vs-year comparison draws a two-line overlay. Markers (red dot = anomaly, amber ring =
  borderline near-miss, orange/blue dots = max/min, shaded band = 80% forecast range) are explained
  if you ask, e.g. *"what does the red dot mean?"*.

## Troubleshooting

**"File not found" / it won't load the data.** The CSV isn't at `code/v1/data/jena_climate_2009_2016.csv`.
Confirm the exact filename and that it sits in `v1`'s own `data/` folder. You can also point
the CLI at another location with `--data /path/to/jena_climate_2009_2016.csv`.

**Port 3030 already in use.** The web app automatically steps up to the next free port and prints
which one it used, so it should still start. To force a port:

```bash
python3.12 apps/web/app.py 3050          # command-line argument
PORT=8080 python3.12 apps/web/app.py     # or the PORT env var
```

The order of preference is: command-line argument → `PORT` env var → 3030.

**"Ollama not reachable" warning.** This is harmless — the LLM is optional and the app falls back to
template phrasing automatically. To enable rewording, install [Ollama](https://ollama.com), then:

```bash
ollama serve
ollama pull llama3.2:3b            # the default small model
```

For better phrasing, pull a larger model and it's picked up automatically (or set `TSHCI_MODEL`):

```bash
ollama pull llama3.1:8b
export TSHCI_MODEL=llama3.1:8b
```

**`python3.12: command not found`.** Install Python 3.12, or substitute your interpreter — but note
the dependencies (especially PyTorch) are pinned for 3.12, so another version may not resolve. Check
with `python3.12 --version`.

**PyTorch / GPU install trouble.** `requirements.txt` (the default) installs the CPU torch wheel —
no GPU needed. If you have an NVIDIA GPU and want CUDA acceleration for the deep models, use
`requirements-gpu.txt` instead (see Install above), which pins the CUDA 12.8 build used on the
project's RTX 5070. The deep models are device-aware either way and run fine on CPU, just slower;
the core Q&A doesn't need a GPU at all.

## Verify the install

To confirm everything is wired up correctly:

```bash
python3.12 -m pytest tests/        # 520 passed, 1 skipped (skips = optional LLM/dense-retriever deps)
python3.12 verify_results.py       # 35/35 independent checks (results == pandas; LLM invents 0 numbers)
```

`verify_results.py` is the trust gate: it independently recomputes headline results and confirms the
LLM never introduced a number absent from the data. There is also a `reproduce.sh` script that runs
the checks end to end.

## Where to go next

- **[README.md](README.md)** — the developer's map: the five code layers and how to run the full
  pipeline.
- **`notebooks/00_code_walkthrough.ipynb`** — the whole system as a runnable tour, module by module.
  *Restart & Run All* and read top to bottom — the fastest way into the code.
- **`report/`** — the write-up and results: `REPORT.md`, `EVALUATION.md`, `MODELS.md` (the six
  detectors compared), `DATA_QUALITY.md`.
