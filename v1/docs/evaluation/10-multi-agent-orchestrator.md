# 10. Multi-agent orchestrator — decompose, dispatch, critique, combine

## What's being evaluated

Not computation (exact and verified elsewhere) but **coverage of genuinely multi-part questions** and
the value of a **second trust gate**. A single natural-language query engine answers one thing; a
question like *"the coldest temperature ever **and** the windiest reading"* has two independent parts.
The orchestrator (`src/orchestrator.py`) decomposes such a question into `SubTask(role, question)`,
dispatches each to a **role-restricted `agent.Agent`** (statistician / correlator / anomaly) **in
parallel**, runs a **Critic** over the assembled results, and **combines** them into one
faithfulness-gated answer. It is a strict superset of the single-agent path: offline or single-topic,
it collapses to one sub-task and behaves exactly like `agent.Agent.run`.

The trust invariant is unchanged: the LLM only **decomposes, critiques, and writes prose**; every
number is produced by `query_engine`/`analysis` via `agent.execute_step` (`facts == pandas`).

## Method

Corpus: `src/orchestrator_eval.py::MULTIPART_CASES` — three multi-part questions, each hand-labelled
(before running code) with the sensor/scope markers a *complete* answer must mention. `score()` counts
how many produce a combined answer containing **every** expected marker. This measures **coverage**
(are both parts addressed, with the right sensors/scope) — deliberately *not* the numbers, which the
query engine verifies exactly. Run twice: LLM **off** (deterministic baseline) vs **on** (Ollama
`llama3.2:3b`).

## Real evidence (30 July 2026, local `llama3.2:3b`)

```
orchestrator off: {'n': 3, 'complete': 2}
orchestrator on : {'n': 3, 'complete': 3}
```

### Per-question traces

**LLM off** (each question stays one sub-task; the interface's own compound splitter handles two):

| Question | Sub-tasks (offline) | Result |
| --- | --- | --- |
| highest temp 2015 **and** lowest humidity 2013 | 1 (`statistician`) | ✅ both parts (interface compound splitter) |
| mean pressure 2014 **and** count temp > 30 that year | 1 (`statistician`) | ✅ both parts |
| coldest temp ever **and** windiest reading | 1 (`statistician`) | ✗ misses — offline the single answer doesn't surface both sensors |

**LLM on** (the decomposer splits into independent sub-tasks dispatched in parallel):

| Question | Sub-tasks the planner produced | Result |
| --- | --- | --- |
| highest temp 2015 **and** lowest humidity 2013 | 1 (`statistician`) — kept whole | ✅ |
| mean pressure 2014 **and** count temp > 30 | `statistician`: "mean pressure in 2014" · **`anomaly`**: "how many times temperature was above 30 in 2014" | ✅ |
| coldest temp ever **and** windiest reading | `statistician`: "coldest temperature ever" · **`anomaly`**: "windiest reading" | ✅ |

## Honest reading

- **Coverage up on multi-part questions (2/3 → 3/3), and the gain is real but narrow.** The offline
  path already answers two of the three via the interface's conservative compound splitter; the
  orchestrator's win is the third case, where a genuine LLM decomposition into two parallel sub-tasks
  is what surfaces both sensors. On single-topic questions the orchestrator adds nothing (by design) —
  it *is* the single-agent path.
- **The 3B decomposer mis-assigns roles.** It routed *"how many times temperature was above 30"*
  (a threshold question, owned by `statistician`) and *"windiest reading"* (a max) to the **`anomaly`**
  role. Coverage still passed because the role's tool restriction plus the per-sub-agent interface
  fallback recovered an answer mentioning the right sensor — but the routing itself is imperfect. This
  is model-bound: the tool-restriction and fallback are what keep a mis-route from becoming a wrong
  number. A stronger model would raise decomposition fidelity; the trust gates are model-independent.
- **Two critics, very different reliability.** The **deterministic** Critic is trustworthy — it flags
  a sub-task with no usable result or an executor error, nothing else. The **LLM** Critic (3B) is
  **noisy**: on this run it raised concerns like *"No mean temperature value"* and *"wind speed data
  not specified"* for parts that were, in fact, answered. So the LLM critic over-flags. The honest
  posture: treat the deterministic critic as the gate and the LLM critic as *advisory notes* shown to
  the user under "⚠ Critic notes", never as a blocker. (A bug this surfaced: the deterministic critic
  first flagged "no usable result" whenever `facts` was empty — but a sub-agent that falls back to the
  interface produces a valid *computed* string with empty `facts`. Fixed to treat a non-empty exact
  `computed` as usable; see `test_run_offline_matches_single_agent_facts` and the Critic docstring.)
- **The scorer measures coverage, not deep semantic correctness.** Because it checks sensor/scope
  markers, a part answered by the "wrong" tool (e.g. "windiest" served as an anomaly count over `wv`)
  can still score complete. This is intentional — numbers are verified exactly elsewhere — but it
  means "3/3" is a *coverage* claim, and the traces above are the honest detail behind it.
- **Latency vs coverage, and the COSMA seam.** The multi-agent layer costs one LLM decompose call plus
  one plan+synthesis per sub-task. Sub-tasks run in a `ThreadPoolExecutor`, so independent parts
  overlap; this parallel `dispatch` is exactly the seam sub-project 3 (COSMA/DiRAC) replaces with a
  SLURM array job for large fan-outs.
- **Offline safety / regression.** With Ollama off the orchestrator equals the single-agent path
  (`test_run_offline_matches_single_agent_facts`), and `dispatch` order is deterministic despite
  parallelism (`test_dispatch_facts_equal_pandas_and_preserve_order`).

## Reproduce

```bash
cd code/v1
python3.12 - <<'PY'
from src import preprocess, orchestrator, orchestrator_eval as oe
raw = preprocess.clean(preprocess.load_data("data/jena_climate_2009_2016.csv"))
df = preprocess.align(raw, freq="1h")
print("off:", oe.score(orchestrator.Orchestrator(df, use_llm=False)))
print("on :", oe.score(orchestrator.Orchestrator(df, use_llm=True)))
PY
# interactive:
python3.12 apps/chat.py --orchestrate "coldest temperature ever and the windiest reading"
```
