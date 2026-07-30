# Related work — reading list for the report

A curated, grouped reading list to support the dissertation's related-work and discussion sections.
Each entry has a one-line **relation** note tying it to *this* system (deterministic truth layer +
local LLM explanation + NL/HCI interface over time series). arXiv IDs below resolve to the correct
papers; **verify exact author lists / spellings on arXiv before final submission**.

Grouped by where each slots into the report.

---

## A. Compute in code, not in the LLM — justification for the truth layer

This is the strongest citation cluster: the external-LLM benchmark
(`v1/docs/evaluation/08-external-llm-benchmark.md`) *empirically reproduces* these papers' core
argument — general LLMs make arithmetic/reading errors even with the data in front of them, so
computation should be offloaded to code.

- **PAL: Program-Aided Language Models** — Gao et al., arXiv:2211.10435.
  *Relation:* essentially our architecture — the LLM reads the question, but a Python runtime does
  the computation. Cite as the principle our system instantiates (LLM plans, `query_engine`/pandas
  computes).
- **Program of Thoughts (PoT)** — Chen et al., arXiv:2211.12588.
  *Relation:* separates numerical computation from language reasoning — the same separation our
  faithfulness gate enforces (numbers come from code, LLM only rewords).
- **ReAct: Synergizing Reasoning and Acting in LLMs** — Yao et al., arXiv:2210.03629.
  *Relation:* interleave reasoning with tool calls — the pattern behind `agent.py`'s multi-step
  tool use and the compound-question decompose→dispatch handler.
- **Toolformer** — Schick et al., arXiv:2302.04761.
  *Relation:* LLMs deciding when to call tools/APIs — background for the tool-use framing.

## B. LLMs + time series — domain positioning

- **Large Language Models for Time Series: A Survey** — Zhang et al., IJCAI 2024, arXiv:2402.01801.
  *Relation:* the anchor survey for the related-work section on LLMs applied to time series.
- **Time-LLM: Time Series Forecasting by Reprogramming LLMs** — Jin et al., ICLR 2024.
  *Relation:* contrast-with — they make the LLM *do* the forecasting; we deliberately keep the LLM
  out of computation and use classical models (ARIMA/SARIMA) for numbers.
- **A Survey of Reasoning and Agentic Systems in Time Series with LLMs** — arXiv:2509.11575.
  *Relation:* directly covers the multi-agent + time-series question; use for the discussion of
  whether an agent layer is warranted here.
- **TimeXL: Explainable Multi-modal Time Series Prediction with LLM-in-the-Loop** —
  arXiv:2503.01013.
  *Relation:* supports the explainability/trust angle (LLM explains, doesn't compute).

## C. Multi-agent frameworks — for the multi-agent discussion

- **LLM-based Multi-Agents: A Survey of Progress and Challenges** — arXiv:2402.01680.
  *Relation:* survey to frame the honest "is multi-agent warranted for this narrow question set?"
  argument (our position: helps orchestration, not correctness).
- **AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation** — Wu et al.,
  arXiv:2308.08155.
- **MetaGPT: Meta Programming for a Multi-Agent Collaborative Framework** — Hong et al.,
  arXiv:2308.00352.
- **CAMEL: Communicative Agents for "Mind" Exploration of LLM Society** — Li et al.,
  arXiv:2303.17760.
  *Relation (C group):* the three canonical role-decomposition frameworks; cite whichever's
  division-of-labour matches a proposed specialist-agent design (max / correlation / anomaly /
  forecast agents), and note the coordination/verification cost that argues against it here.

## D. Evaluation method actually used

- **Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena** — Zheng et al., NeurIPS 2023,
  arXiv:2306.05685.
  *Relation:* cite for the routing/explanation LLM-as-judge legs of the benchmark, **and** for its
  documented biases (position, verbosity, self-enhancement) — the reason we scored against an
  independent oracle and did not let the model grade itself.

---

## Infrastructure note — COSMA / DiRAC HPC

Not a method, but relevant if compute is scaled out. COSMA is the DiRAC "Memory Intensive" service
at Durham (RAC-allocated time; AMD CPU nodes with 1 TB RAM/node, some GPU nodes for AI workloads).

- **Where it helps:** embarrassingly-parallel work — running the external-LLM benchmark across many
  models at once, hyperparameter/CV sweeps, and GPU training of the deep detectors (CNN-AE /
  LSTM-AE / LSTM-forecaster).
- **Honest caveat for the report:** it adds nothing to *correctness* (the truth layer is exact and
  instant single-core), and using a shared national facility cuts against the system's
  **local/offline, privacy-preserving** framing. If used, scope it explicitly to *offline
  experimentation and training*, not to the interactive system.

Docs: https://www.durham.ac.uk/departments/academic/physics/cosma/ ·
https://dirac.ac.uk/memory-intensive-durham/ · https://cosma.readthedocs.io/en/latest/dirac.html
