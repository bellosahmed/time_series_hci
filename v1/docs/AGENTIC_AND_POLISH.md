# Agentic tool-use + polish — items 6 & 7

What was built on **2026-07-05** for the two remaining engineering items in
`ASSESSMENT_AND_RECOMMENDATIONS.md`: **(6)** agentic, still-numberless tool-use, and **(7)** the
polish tier (larger-model option, embedding-RAG option, threshold edge-case fix, mypy).

**State after this work:** 140 tests pass (1 skipped) · `verify_results.py` 25/25 · `ruff` clean ·
`mypy` clean (16 source files).

---

## 6. Agentic, still-numberless tool-use (`src/agent.py`)

### The problem it solves
The single-step pipeline answers one question with one deterministic computation
(`interface.Interface`). A **multi-step** question — *"compare the hottest of 2013, 2014 and 2015
and say which stood out"* — has no single query that answers it. It needs several computations
chained and then compared.

### The design (the LLM plans; the code computes)
The core guarantee is preserved: **the LLM never produces a number.** It only *plans*.

```
question ──▶ LLM plan            (ordered list of tool calls)
         ──▶ execute each tool   (deterministic, against pandas → exact facts)
         ──▶ LLM synthesis       (weaves the exact facts into prose)
         ──▶ faithfulness gate   (reject if any number was invented)  ──▶ answer
```

- **Plan** (`llm.plan_steps`): the model receives the tool catalogue + allowed sensors and returns
  JSON `{"steps": [{"tool", "args"}, …]}`. Shape-validated only; it chooses *which* tools and *which*
  arguments (sensors, dates, operator), never a data value.
- **Execute** (`agent.execute_step` / `execute_plan`): pure Python. Validates each step (known tool,
  known sensor, valid `kind`/`op`/`value`, coercible dates) and dispatches to the deterministic
  query engine. A bad step returns a failed `Fact`, never a crash. **Every result equals pandas.**
- **Compose** (`agent.Agent.compose`): tries an LLM synthesis (`llm.explain_synthesis`) but accepts
  it **only if `llm.faithfulness` finds no invented number**; otherwise it returns a trust-labelled
  deterministic template. When the LLM is off, it always returns the template.
- **Fallback** (`agent.Agent.run`): if the LLM is unavailable or produces no usable plan, it delegates
  to the single-step `interface.Interface`, so the user always gets an answer.

### Tools (all thin wrappers over the truth layer)
| tool | computes | delegates to |
| --- | --- | --- |
| `extremum` | max/min of a sensor in a window | `query_engine.get_max/get_min` (circular-aware) |
| `average` | mean/std/range | `query_engine.get_mean_std` (circular-aware) |
| `correlation` | Pearson (or circular–linear) between two sensors | `query_engine.get_correlation` / `analysis` |
| `threshold` | count above/below a value | `query_engine.threshold_events` |
| `anomaly_count` | anomalous readings + windows | `analysis.decompose` + `anomaly.zscore_anomalies` |

Circular sensors (`wd (deg)`) are handled specially (prevailing direction, not a meaningless linear
max), matching the interface.

### Why this is the valuable extension
It moves *"Verifiable NL Analytics"* from "the LLM explains one verified number" to "the LLM plans a
multi-step analysis, but the truth layer still owns every number and the faithfulness gate still
verifies them." That is the agent flexibility **without** reintroducing hallucination.

### Honest limits
- The only number the planner may emit is a threshold `value` echoed from the user's question (a query
  parameter, validated as a float), never a data value — the same residual as the existing `llm.plan`.
- A 3B model's tool-calling reliability is modest; the executor's validation + the single-step fallback
  keep it safe, but robust multi-step planning is a reason to pull a larger local model (see 7a).

### Try it
```bash
python3.12 apps/chat.py --agent "compare the hottest of 2013, 2014 and 2015"
```

---

## 7. Polish

### 7a — Larger local model option (`llm.py`)
- `DEFAULT_MODEL` now reads the **`TSHCI_MODEL`** env var (falls back to `llama3.2:3b`).
- New **`llm.pick_model(prefer, requested)`**: an explicit request wins; otherwise the first *pulled*
  model from `PREFERRED_MODELS` (larger first: `llama3.1:8b`, `qwen2.5:7b`, `mistral:7b`, `llama3.2:3b`)
  is used; else `DEFAULT_MODEL`. `chat.py` delegates to it, so pulling a bigger model auto-upgrades.
- **Why:** the faithfulness gate already guarantees the *numbers*; a larger model mainly reduces wrong
  *qualifiers* (the residual the gate can't catch). Made an easy opt-in, not a hard requirement.
```bash
ollama pull llama3.1:8b          # then it is picked automatically
TSHCI_MODEL=qwen2.5:7b python3.12 apps/chat.py --llm "…"
```

### 7b — Embedding-based RAG option (`rag_system.py`)
- New **`EmbeddingRetriever`**: embeds the sensor KB with a local embedding model (Ollama
  `nomic-embed-text`) and ranks by cosine similarity over unit-normalised vectors — semantic match
  instead of keyword overlap.
- **Drop-in** (same `retrieve` / `for_sensor` interface as `Retriever`) and **always safe**: if no
  embedding backend is reachable it transparently falls back to the TF-IDF `Retriever` (`semantic` is
  then `False`). TF-IDF remains the interface default; this is the recommended-but-optional upgrade.
- `embed_fn` is injectable (tests / alternative backends).
```bash
ollama pull nomic-embed-text
```

### 7c — `resolve_threshold` year edge-case fixed (`interface_parsing.py`)
The threshold is now the number **following** the comparison word:
- `"above 30 in 2013"` → `(">", 30.0)` — the scope year is ignored.
- `"above 2000"` → `(">", 2000.0)` — a genuine threshold in the 1900–2100 range is kept **when it sits
  directly after the comparison word** (the old blanket "drop 1900–2100" heuristic wrongly dropped it).
- `"above average in 2015"` → `None` — a year that is only scope (not adjacent to the word) is not a
  threshold, so no spurious query is produced.

### 7d — mypy
- Added to the `dev` optional dependencies + a **lightweight, non-strict** `[tool.mypy]` config on the
  source layer (`ignore_missing_imports`; a few dynamic error codes disabled for the deliberate
  `value: object` field and cross-type local reuse in `interface.py`).
- A rigour signal that catches obvious type mistakes without a full annotation pass.
```bash
python3.12 -m mypy        # Success: no issues found in 16 source files
```

---

## Files touched
- **New:** `src/agent.py`, `tests/test_agent.py`, `tests/test_rag.py`, this doc.
- **Changed:** `src/llm.py` (env-var default, `pick_model`, `plan_steps`, `explain_synthesis`),
  `src/rag_system.py` (`ollama_embed`, `EmbeddingRetriever`), `src/interface_parsing.py`
  (`resolve_threshold`), `apps/chat.py` (`--agent`, delegates model choice), `pyproject.toml`
  (mypy dep + config), `tests/test_llm.py`, `tests/test_interface.py`, `tests/test_apps.py`.

## Reproduce the checks
```bash
cd code/v1
python3.12 -m pytest tests/      # 140 passed, 1 skipped
python3.12 verify_results.py     # 25/25
python3.12 -m ruff check src tests apps
python3.12 -m mypy               # clean
```
