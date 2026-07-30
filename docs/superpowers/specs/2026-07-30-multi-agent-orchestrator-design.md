# Multi-agent orchestrator — design (sub-project 2)

_30 July 2026._

## Purpose

Answer hard, multi-part questions ("compare the hottest 2015 day with the wettest 2013 day and say
whether pressure and humidity move together that summer") by **decomposing** them into independent
sub-tasks, running **specialist role-agents in parallel** over the same deterministic truth layer,
**verifying** the combined facts with a Critic, then **combining** them into one answer — without ever
letting an LLM compute a number.

This is a strict superset of the existing single-planner `agent.py`: when the LLM is off or the
question is single-topic, the orchestrator collapses to exactly one sub-task and behaves identically to
`agent.Agent.run`. The parallel dispatch seam is what sub-project 3 (COSMA batch harness) later
replaces with SLURM array jobs.

## Non-goals

- No new arithmetic. Every number still comes from `query_engine`/`analysis` via the proven
  `agent.execute_step` executors (`facts == pandas`).
- No debate/ensemble/voting (novelty is out of scope; it burns tokens for no gain on a deterministic
  truth layer).
- No separate specialist agent *classes* — a "role" is a label plus a curated tool subset handed to
  the existing `agent.Agent`.
- No web-UI work in this sub-project (a `chat.py --orchestrate` flag is enough to exercise it).

## Architecture

New module `v1/src/orchestrator.py`. Two new LLM helpers in `llm.py` (`decompose`, `critique`), both
numberless and offline-safe (return a trivial/empty result when Ollama is unavailable).

```
question
  → Planner.decompose()      llm.decompose → ordered list[SubTask(role, sub_question)]
                             LLM-off / single-topic / conservative-guard → 1 SubTask = whole question
  → dispatch (parallel)      each SubTask → role-agent = agent.Agent.run over that role's tool subset
                             every number computed by query_engine (== pandas); order preserved
  → Critic.critique()        deterministic checks (+ optional LLM flag); never invents numbers
  → Combiner.combine()       faithfulness-gated prose (reuse agent.compose); template fallback offline
  → OrchestratedAnswer(question, subtasks, results, text, flags, computed, used_llm)
```

### Components

**1. Planner — `decompose(question) -> list[SubTask]`**
- Uses `llm.decompose(question, ROLES, columns)` to split into `SubTask(role, sub_question)`.
- **Conservative guard (deterministic, runs before and after the LLM):** never split a question
  containing a correlation/compare cue or "between X and Y" — reuse the same protected-intent logic as
  `interface._split_questions`. A protected question becomes a single SubTask.
- **Degenerate cases → one SubTask** holding the whole question: LLM off/unavailable, LLM returns
  empty/`NONE`, or LLM returns a single item. This guarantees the "== agent.py" regression property.
- Each returned `role` is validated against `ROLES`; an unknown role falls back to `statistician`
  (the general role) rather than being dropped.

**2. Role-agents — `dispatch(subtasks) -> list[RoleResult]`**
- `ROLES: dict[str, tuple[str, ...]]` maps a role to a subset of `agent.TOOLS` names:
  - `statistician` → `("extremum", "average", "threshold")` (the general role / fallback)
  - `correlator` → `("correlation",)`
  - `anomaly` → `("anomaly_count",)`
- Each sub-task runs through `agent.Agent(df, use_llm, model).run(sub_question)`, but with the tool
  catalogue restricted to the role's subset (so the planner prompt for that sub-agent only offers the
  relevant tools). Restriction is passed explicitly; `agent.py` gains an optional `tools` argument to
  `Agent.__init__`/`tool_specs`/`plan` — a small, backwards-compatible addition (default = all tools).
- **Parallel execution:** `concurrent.futures.ThreadPoolExecutor` over the sub-tasks (Ollama calls are
  I/O-bound and independent). Results are re-ordered to match the input order so output is
  deterministic. `max_workers` is capped (default 4) and configurable.
- Each `RoleResult` carries the role, the sub-question, the executed `Fact`s, the exact `computed`
  string, and the role-agent's own draft text.

**3. Critic — `critique(question, results) -> list[str]`** (flags, empty = clean)
- **Deterministic checks (always run):**
  - every `RoleResult` produced at least one `ok` fact, else flag `"<role>: no usable result"`.
  - no requested sub-task silently errored (surface `Fact.error`s as flags).
  - sensor/scope sanity: if two sub-tasks name contradictory scopes for the *same* comparison the
    combiner is told (flag, not a crash).
- **Optional LLM check (only when Ollama is on):** `llm.critique(question, [(role, computed)])` may
  return a short list of natural-language concerns (e.g. "sub-answer 2 does not address humidity").
  The LLM critic **only flags**; it never edits numbers. Its output is appended to the deterministic
  flags. If it errors/returns nothing, flags are just the deterministic ones.
- Flags never block the answer; they are attached to `OrchestratedAnswer.flags` and (when non-empty)
  surfaced under a "⚠ Critic notes:" line so the user sees them.

**4. Combiner — `combine(question, results, flags) -> tuple[str, bool]`**
- Reuse `agent.Agent.compose` logic: gather all `ok` facts across roles, ask the LLM for a synthesis,
  accept only if `llm.faithfulness` finds no invented number; otherwise a deterministic
  trust-labelled template (grouped by role). Returns `(text, used_llm)`.
- Critic flags, if any, are appended after the composed answer.

### Public API

```python
class Orchestrator:
    def __init__(self, df, use_llm=True, model=None, max_workers=4): ...
    def decompose(self, question) -> list[SubTask]: ...
    def dispatch(self, subtasks) -> list[RoleResult]: ...   # parallel; order preserved
    def critique(self, question, results) -> list[str]: ...
    def combine(self, question, results, flags) -> tuple[str, bool]: ...
    def run(self, question) -> OrchestratedAnswer: ...       # full pipeline

@dataclass
class SubTask:      role: str; question: str
@dataclass
class RoleResult:   role: str; question: str; facts: list; computed: str; text: str
@dataclass
class OrchestratedAnswer:
    question: str; subtasks: list; results: list; text: str
    flags: list; computed: str; used_llm: bool
```

## Data flow / error handling

- One bad sub-task never sinks the run: `agent.execute_step` already returns `Fact(ok=False, error=…)`
  instead of raising, and `dispatch` catches any executor exception per-future. If *every* sub-task is
  empty, `run` falls back to a single `agent.Agent.run(question)` (which itself falls back to
  `interface.Interface`), so the user always gets an answer.
- Offline (`use_llm=False` or Ollama down): `decompose` → 1 SubTask, `dispatch` runs the deterministic
  executor path (agent.py plans nothing, so `run` uses the interface fallback per sub-task), `critique`
  runs deterministic checks only, `combine` returns the template. Behaviour is identical to today's
  single-step path — asserted by a regression test.

## Testing (TDD, all offline)

1. `decompose` offline → exactly one SubTask equal to the whole question.
2. `decompose` protects a correlation/compare/"between X and Y" question → one SubTask (no split).
3. `dispatch` runs each SubTask through the executor; facts `== pandas` for a known window; order of
   results matches order of sub-tasks even though execution is parallel.
4. A role's tool subset is respected: a `correlator` sub-task cannot execute an `extremum` tool.
5. `critique` flags a sub-task with no usable fact, and flags an executor error.
6. `critique` flags an invented number (feed a draft with a number absent from the facts).
7. `combine` offline returns the deterministic template grouped by role (no LLM).
8. **Regression:** for a simple single-topic question, `Orchestrator(use_llm=False).run(q).text`
   matches `agent.Agent(use_llm=False).run(q)` facts (strict superset property).
9. A genuine two-part question (stubbed decompose) yields ≥2 RoleResults and a combined answer whose
   `computed` contains every sub-task's facts.
10. `llm.decompose`/`llm.critique` return the safe default when `llm.is_available()` is False (mirrors
    `understand`/`rewrite_followup`).

## Evaluation deliverable

`v1/docs/evaluation/10-multi-agent-orchestrator.md`: a small corpus of multi-part questions, showing
(a) single-agent `agent.py` vs orchestrator coverage, (b) the Critic catching a deliberately-wrong
draft, (c) the offline == single-path regression, and (d) an honest reading of what the multi-agent
layer buys and costs (latency vs coverage; the parallelism only helps when sub-tasks are independent).

## Trust invariants (unchanged)

- LLM only decomposes / critiques / writes prose. All numbers come from pandas.
- Every accepted synthesis passes the `llm.faithfulness` gate.
- Fully offline-safe and regression-guarded against the existing single-agent path.
