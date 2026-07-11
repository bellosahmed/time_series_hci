# Getting started with v3

A friendly on-ramp for someone who has never seen this project. If you can open a terminal
and run one command, you can get this running. The [`README.md`](README.md) is the denser,
developer-facing tour — this page is the gentle first walk-through.

## What is this?

v3 lets you **ask plain-English questions about a real data-center IoT network and get exact,
trustworthy answers**. The data is the **DAD dataset** (Vigoya et al., 2020) — seven days of
MQTT network traffic from a data centre full of temperature probes, captured as raw `.pcap`
files and shipped with **real, human-labelled attack markers**. Someone injected malicious
traffic from a single host (`10.6.56.41`) that duplicates, intercepts and modifies sensor
messages; the dataset records exactly which 544 packets were the attack. You ask things like
*"how many attacks on Wednesday?"* or *"which IP is the attacker?"*, and the system computes
the answer from the data and shows it with a little trust label so you know where the number
came from. An optional local AI can reword answers into friendlier prose, but it is never
allowed to invent a number.

## What you need before you start

- **Python 3.12.** Everything here uses the `python3.12` command specifically. Check with
  `python3.12 --version`.
- **The dataset.** It is not shipped with the code (it is git-ignored, ~18 MB). You fetch it
  once with a script — see [Get the data](#get-the-data). The app will not start without it.
- **Git.** The fetch script clones the dataset from GitHub, so you need `git` installed.
- **A local LLM is optional.** If you have [Ollama](https://ollama.com) running, answers can
  be reworded into nicer prose. If you do not, *everything still works* — you just get the
  exact computed answer in template form. You never need it to try the project.

You do **not** need `sudo`, Wireshark, `tshark`, or any packet-capture tooling. The pcaps are
parsed entirely in Python by the `scapy` library (installed with the other dependencies), so
there is nothing to configure at the OS level.

## Install

From the `code/v3` directory, install the dependencies once:

```bash
python3.12 -m pip install -r requirements.txt
```

This pulls in pandas, scapy, Flask, scikit-learn and friends. It only needs doing once.

## Get the data

Fetch the real DAD dataset (~18 MB, one-time):

```bash
bash tools/get_dad_dataset.sh
```

This clones the dataset and writes it to **`code/data/dad/`** (that is one level up from
`v3/`, in `code/data/`). When it finishes you should have 22 files: 7 `.pcap` captures,
7 `_labels.csv`, 7 `_labels.xml`, and a `README.md`. If this directory is missing, the app
cannot load — every run starts by parsing these pcaps.

## Run it — the fastest path

You have two ways in. Pick either.

**Option A — one question from the command line (quickest):**

```bash
python3.12 apps/chat.py "how many attacks on wednesday?"
```

It prints the answer with its trust label. The first run pauses for a few seconds while it
parses the week of pcaps. Add `--llm` to have a local Ollama model reword the answer:

```bash
python3.12 apps/chat.py --llm "how many attacks on wednesday?"
```

Run `apps/chat.py` with no question to drop into an interactive prompt (Ctrl-D to exit).

**Option B — the web app (the 60-second win):**

```bash
python3.12 apps/web/app.py
```

It prints a line like `Open  http://localhost:3040`. Open that URL in your browser. You get a
small chat page with example chips you can click, charts, and an **"Explain with AI"** toggle.
Stop the server with `Ctrl-C`.

The web app defaults to **port 3040** and automatically hops to the next free port if 3040 is
already taken (watch the printed line for the real port). To force a port:

```bash
python3.12 apps/web/app.py 8080          # or:  PORT=8080 python3.12 apps/web/app.py
```

The page is plain HTML + JavaScript served by Flask — no build step, no internet needed.

## Try these questions

All of these are understood by the interface. Type them in the web app, or pass them to
`apps/chat.py` in quotes:

```bash
python3.12 apps/chat.py "how many attacks on wednesday?"
python3.12 apps/chat.py "which IP is the attacker?"
python3.12 apps/chat.py "show me the anomalies"
python3.12 apps/chat.py "what types of attack are in the data?"
python3.12 apps/chat.py "show all the sensors in the dataset"
python3.12 apps/chat.py "what was the highest temperature on Friday?"
python3.12 apps/chat.py "when did the attacks happen?"
python3.12 apps/chat.py "what was the busiest hour?"
python3.12 apps/chat.py "give me a summary of Thursday"
```

Things you can vary:
- **Any day of the week** — "how many attacks on Monday?", "summary of Saturday".
- **An attack type** — "how many modification attacks?" (or duplication / interception).
- **A temperature statistic** — "highest", "lowest" or "average" temperature, optionally per day.

## What you're looking at

- **Trust labels.** Every answer is tagged so you know the provenance of each part:
  🔢 *computed* (an exact figure straight from the data), 📚 *context* (fixed background about
  the domain, never a computed number), 🧠 *inferred* (a cautious interpretation). The core
  design rule is simple: **numbers are computed by code — the source of truth — and the LLM
  only ever rephrases them.**
- **The optional LLM, and the faithfulness gate.** With `--llm` (CLI) or the "Explain with AI"
  toggle (web), a local Ollama model rewrites the computed answer into friendlier prose. Before
  that prose is shown it passes a **faithfulness check** that rejects any invented number or IP;
  if the check fails, or Ollama is not running, you simply see the exact template instead. So
  the AI can make answers nicer but can never make them wrong.
- **Charts.** The web app draws a small chart alongside each answer — attack packets per day,
  readings per sensor, attacks by type, a temperature line, or messages per hour, depending on
  what you asked.

## Troubleshooting

- **"No such file" / it fails to load the dataset.** You have not fetched the data yet, or it
  landed somewhere unexpected. Run `bash tools/get_dad_dataset.sh` and confirm the files are in
  `code/data/dad/` (that is beside `v3/`, not inside it).
- **"Address already in use" / port 3040 busy.** The web app normally hops to the next free
  port automatically — check the printed `Open http://localhost:<port>` line for the real one.
  To pin a port: `python3.12 apps/web/app.py 8080`.
- **The AI toggle does nothing / answers stay in template form.** That means Ollama is not
  running (or has no model). This is expected and harmless — every answer still works without
  it. To enable rewording, install Ollama and pull a small model (e.g. `ollama pull llama3.2:3b`),
  then try again.
- **`python3.12: command not found`, or odd import errors.** You are on the wrong Python. This
  project targets **Python 3.12** specifically; check with `python3.12 --version`, and make sure
  you installed the requirements with the same `python3.12 -m pip ...` command.
- **Noisy scapy warnings on start-up.** scapy sometimes grumbles about optional protocol layers
  (TLS/Kerberos/SMB) it cannot fully load. This is deliberately silenced in the code and is
  harmless — the pcap parsing does not use those layers.

## Verify the install

To confirm everything is wired up correctly, run the one-command check from `code/v3`:

```bash
bash reproduce.sh
```

It fetches the data if missing, then runs the test suite with `pytest`, an independent
results check (`verify_results.py`, 12/12), and the `ruff` linter. You can also run those
pieces individually:

```bash
python3.12 -m pytest tests/
python3.12 verify_results.py
```

## Advanced: fully-offline Open WebUI + GPU stack

Not needed to run anything above. If you want a full on-premise chat stack — local
[Open WebUI](https://openwebui.com) + Ollama in Docker with NVIDIA GPU pass-through and a
read-only dataset mount for private RAG — the notebook
[`local_webui_dataset_spec.ipynb`](local_webui_dataset_spec.ipynb) documents the whole setup
(compose file, `.env`, GPU health checks, privacy boundaries, troubleshooting). Treat it as an
optional, advanced extra.

## Where to go next

- **[`README.md`](README.md)** — the denser project map and design notes.
- **Notebooks** (`notebooks/`, run with *Restart & Run All*) — a guided in-depth tour:
  `00_code_walkthrough`, `01_data_exploration_and_visualization`,
  `02_methods_and_evaluation`, `03_interface_and_explanation`, and the newcomer-friendly
  `04_build_along_tutorial`.
- **[`report/EVALUATION.md`](report/EVALUATION.md)** — the honest results-and-reasoning log
  (including the headline finding on detector fusion).
- **[`docs/PLAN.md`](docs/PLAN.md)** and **[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)** —
  the full plan, module map and design.
