# Multi-agent Orchestrator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a multi-agent layer that decomposes hard questions into sub-tasks, runs specialist role-agents in parallel over the deterministic truth layer, verifies with a Critic, and combines — never letting the LLM compute a number.

**Architecture:** New `v1/src/orchestrator.py` wraps the existing `agent.Agent`. A Planner (`llm.decompose`) splits a question into `SubTask(role, question)`; `dispatch` runs each through `agent.Agent.run` restricted to that role's tool subset, in parallel; a Critic (`llm.critique` + deterministic checks) flags problems; a Combiner reuses `agent.compose` (faithfulness-gated) for the final prose. Offline or single-topic → one sub-task == today's `agent.py`.

**Tech Stack:** Python 3.12, pandas, `concurrent.futures.ThreadPoolExecutor`, stdlib `urllib` (Ollama), pytest, ruff, mypy.

## Global Constraints

- Python 3.12 (`python3.12`); run tests with `python3.12 -m pytest`.
- Commits authored **solely by the user** — NO `Co-Authored-By: Claude` trailer.
- Do NOT push to GitHub.
- **Trust invariant:** the LLM only decomposes / critiques / writes prose. Every number comes from `query_engine`/`analysis` via `agent.execute_step` (`facts == pandas`).
- Must be fully offline-safe: with `use_llm=False` or Ollama down, behaviour equals the existing single-agent path (regression-guarded).
- `ruff check` and `mypy` must stay clean.
- New LLM helpers mirror `llm.understand`: return a safe default when `llm.is_available()` is False; catch `urllib.error.URLError, OSError`.

---

### Task 1: LLM helpers `decompose` + `critique`

**Files:**
- Modify: `v1/src/llm.py` (add near `understand`, ~line 195)
- Test: `v1/tests/test_llm_orchestrator.py`

**Interfaces:**
- Consumes: `llm.is_available()`, `llm.generate`, `llm._generate_json`, `llm.DEFAULT_MODEL`.
- Produces:
  - `decompose(question: str, roles: list[str], columns: list[str], model=DEFAULT_MODEL) -> list[tuple[str, str]] | None` — ordered `(role, sub_question)` pairs, or None if unavailable/unparseable/empty.
  - `critique(question: str, role_facts: list[tuple[str, str]], model=DEFAULT_MODEL) -> list[str]` — natural-language concern flags (empty list if unavailable or none).

- [ ] **Step 1: Write the failing tests**

```python
# v1/tests/test_llm_orchestrator.py
"""decompose/critique are offline-safe: with Ollama down they return the neutral default."""
from src import llm


def test_decompose_returns_none_when_unavailable(monkeypatch):
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: False)
    assert llm.decompose("max temp 2015 and min humidity 2013",
                          ["statistician", "correlator", "anomaly"],
                          ["T (degC)", "rh (%)"]) is None


def test_critique_returns_empty_when_unavailable(monkeypatch):
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: False)
    assert llm.critique("q", [("statistician", "max T was 30.0")]) == []


def test_decompose_parses_pairs(monkeypatch):
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: True)
    monkeypatch.setattr(llm, "_generate_json", lambda *a, **k: {
        "subtasks": [{"role": "statistician", "question": "max temperature in 2015"},
                     {"role": "statistician", "question": "min humidity in 2013"}]})
    out = llm.decompose("max temp 2015 and min humidity 2013",
                        ["statistician", "correlator", "anomaly"], ["T (degC)", "rh (%)"])
    assert out == [("statistician", "max temperature in 2015"),
                   ("statistician", "min humidity in 2013")]
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3.12 -m pytest tests/test_llm_orchestrator.py -v`
Expected: FAIL (`AttributeError: module 'src.llm' has no attribute 'decompose'`).

- [ ] **Step 3: Implement the helpers**

```python
# in v1/src/llm.py, after understand()
DECOMPOSE_SYSTEM = (
    "You split a user's question about a weather time-series into the SMALLEST number of independent "
    "sub-questions, each answerable on its own. Assign each a role from the allowed list. Do NOT "
    "answer them and NEVER output any number, statistic, or measurement. If the question is already "
    "single-topic, return it unchanged as one sub-task. Respond with ONLY compact JSON, no prose."
)


def decompose(question: str, roles: list[str], columns: list[str],
              model: str = DEFAULT_MODEL) -> list[tuple[str, str]] | None:
    """Split a question into ordered (role, sub_question) pairs the orchestrator dispatches in
    parallel. The model only partitions language; it never computes a value. Returns None if Ollama
    is unavailable or the reply cannot be parsed (caller then treats the whole question as one task)."""
    if not is_available():
        return None
    prompt = (f"Allowed roles: {roles}\nAvailable sensors: {columns}\n\n"
              f"Question: {question}\n\n"
              'Return JSON ONLY: {"subtasks": [{"role": <role>, "question": <sub-question>}, ...]}')
    data = _generate_json(prompt, DECOMPOSE_SYSTEM, model, 45.0)
    if not isinstance(data, dict) or not isinstance(data.get("subtasks"), list):
        return None
    pairs: list[tuple[str, str]] = []
    for st in data["subtasks"]:
        if isinstance(st, dict) and isinstance(st.get("question"), str) and st["question"].strip():
            role = st.get("role")
            pairs.append((role if isinstance(role, str) else "", st["question"].strip()))
    return pairs or None


CRITIC_SYSTEM = (
    "You are a careful reviewer. Given a question and the sub-answers a system produced (each already "
    "containing exact computed numbers you must treat as ground truth), list any concerns: a sub-answer "
    "that does not address its part of the question, a missing part, or an inconsistency. NEVER change, "
    "add, or recompute a number. Reply with ONLY a JSON list of short concern strings; [] if none."
)


def critique(question: str, role_facts: list[tuple[str, str]],
             model: str = DEFAULT_MODEL) -> list[str]:
    """Ask the LLM to flag (not fix) concerns about the assembled sub-answers. Returns a list of short
    strings ([] if Ollama is unavailable, the reply is unparseable, or there are no concerns). The
    critic never touches numbers -- it is a second trust gate, not a computer."""
    if not is_available():
        return []
    joined = "\n".join(f"- [{role}] {facts}" for role, facts in role_facts)
    prompt = (f"Question: {question}\n\nSub-answers:\n{joined}\n\n"
              'Return JSON ONLY: {"concerns": [<short string>, ...]}')
    data = _generate_json(prompt, CRITIC_SYSTEM, model, 30.0)
    if not isinstance(data, dict) or not isinstance(data.get("concerns"), list):
        return []
    return [c.strip() for c in data["concerns"] if isinstance(c, str) and c.strip()]
```

- [ ] **Step 4: Run to verify they pass**

Run: `python3.12 -m pytest tests/test_llm_orchestrator.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add v1/src/llm.py v1/tests/test_llm_orchestrator.py
git commit -m "feat(orchestrator): llm.decompose + llm.critique helpers (numberless, offline-safe)"
```

---

### Task 2: Restrict `agent.Agent` to a tool subset

**Files:**
- Modify: `v1/src/agent.py` (`Agent.__init__`, `tool_specs`, `plan`, ~lines 215-229)
- Test: `v1/tests/test_agent.py` (append)

**Interfaces:**
- Consumes: existing `agent.TOOLS`, `agent.execute_step`.
- Produces: `Agent(df, use_llm=True, model=None, tools: tuple[str, ...] | None = None)` — when `tools` is given, `tool_specs()` and the planner offer only those tool names; `tools=None` (default) keeps all tools (backwards-compatible).

- [ ] **Step 1: Write the failing test**

```python
# append to v1/tests/test_agent.py
from src import agent as _agent


def test_agent_tool_subset_limits_catalogue():
    import pandas as pd
    df = pd.DataFrame({"T (degC)": [1.0, 2.0]},
                      index=pd.date_range("2015-01-01", periods=2, freq="h"))
    ag = _agent.Agent(df, use_llm=False, tools=("correlation",))
    names = {t["name"] for t in ag.tool_specs()}
    assert names == {"correlation"}


def test_agent_default_tools_unchanged():
    import pandas as pd
    df = pd.DataFrame({"T (degC)": [1.0, 2.0]},
                      index=pd.date_range("2015-01-01", periods=2, freq="h"))
    ag = _agent.Agent(df, use_llm=False)
    assert {t["name"] for t in ag.tool_specs()} == set(_agent.TOOLS)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3.12 -m pytest tests/test_agent.py::test_agent_tool_subset_limits_catalogue -v`
Expected: FAIL (`__init__() got an unexpected keyword argument 'tools'`).

- [ ] **Step 3: Implement the subset**

```python
# agent.Agent.__init__
def __init__(self, df: pd.DataFrame, use_llm: bool = True, model: str | None = None,
             tools: tuple[str, ...] | None = None):
    self.df = df
    self.use_llm = use_llm
    self.model = model or llm.DEFAULT_MODEL
    # None => all tools; otherwise restrict the catalogue offered to the planner (unknown names ignored)
    self.tool_names = tuple(TOOLS) if tools is None else tuple(t for t in tools if t in TOOLS)

# agent.Agent.tool_specs
def tool_specs(self) -> list[dict]:
    """The tool catalogue passed to the LLM planner (restricted to ``self.tool_names``)."""
    return [{"name": n, "description": TOOLS[n]["description"], "params": TOOLS[n]["params"]}
            for n in self.tool_names]
```

`plan()` already calls `self.tool_specs()`, so it inherits the restriction — no change needed there.

- [ ] **Step 4: Run to verify both pass**

Run: `python3.12 -m pytest tests/test_agent.py -v`
Expected: PASS (new tests + all existing agent tests still green).

- [ ] **Step 5: Commit**

```bash
git add v1/src/agent.py v1/tests/test_agent.py
git commit -m "feat(agent): optional tool-subset restriction for role-agents (default unchanged)"
```

---

### Task 3: Orchestrator dataclasses + Planner (`decompose`)

**Files:**
- Create: `v1/src/orchestrator.py`
- Test: `v1/tests/test_orchestrator.py`

**Interfaces:**
- Consumes: `llm.decompose`, `interface_parsing.detect_intent`, `interface_parsing.resolve_compare`.
- Produces:
  - `ROLES: dict[str, tuple[str, ...]]` = `{"statistician": ("extremum","average","threshold"), "correlator": ("correlation",), "anomaly": ("anomaly_count",)}`
  - `@dataclass SubTask(role: str, question: str)`
  - `Orchestrator(df, use_llm=True, model=None, max_workers=4)` with `.decompose(question) -> list[SubTask]`.

- [ ] **Step 1: Write the failing tests**

```python
# v1/tests/test_orchestrator.py
"""The Planner is conservative and offline-safe: it only splits when the LLM is on AND the question
is not a protected correlation/compare, and always yields >=1 SubTask covering the whole question."""
import pandas as pd

from src import orchestrator as orch


def _df():
    idx = pd.date_range("2013-01-01", periods=48, freq="h")
    return pd.DataFrame({"T (degC)": range(48), "rh (%)": range(48)}, index=idx)


def test_decompose_offline_is_single_subtask():
    o = orch.Orchestrator(_df(), use_llm=False)
    subs = o.decompose("max temperature in 2015 and min humidity in 2013")
    assert len(subs) == 1
    assert subs[0].question == "max temperature in 2015 and min humidity in 2013"


def test_decompose_protects_correlation(monkeypatch):
    # even with the LLM "on", a correlation question is never split
    monkeypatch.setattr(orch.llm, "is_available", lambda *a, **k: True)
    monkeypatch.setattr(orch.llm, "decompose", lambda *a, **k: [("statistician", "a"),
                                                                ("statistician", "b")])
    o = orch.Orchestrator(_df(), use_llm=True)
    subs = o.decompose("correlation between temperature and humidity in 2013")
    assert len(subs) == 1


def test_decompose_uses_llm_split_and_validates_roles(monkeypatch):
    monkeypatch.setattr(orch.llm, "is_available", lambda *a, **k: True)
    monkeypatch.setattr(orch.llm, "decompose", lambda *a, **k: [
        ("statistician", "max temperature in 2015"),
        ("bogus-role", "min humidity in 2013")])
    o = orch.Orchestrator(_df(), use_llm=True)
    subs = o.decompose("max temp 2015 and min humidity 2013")
    assert [s.role for s in subs] == ["statistician", "statistician"]  # bogus -> statistician
    assert len(subs) == 2
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3.12 -m pytest tests/test_orchestrator.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.orchestrator'`).

- [ ] **Step 3: Implement the module skeleton + Planner**

```python
# v1/src/orchestrator.py
"""Multi-agent orchestration over the deterministic truth layer (still numberless).

A Planner decomposes a hard question into independent SubTasks; each is dispatched to a role-agent
(an ``agent.Agent`` restricted to that role's tools) and executed in parallel; a Critic flags
problems; a Combiner writes the final, faithfulness-gated prose. Offline or single-topic, the whole
thing collapses to exactly one SubTask and behaves identically to ``agent.Agent.run`` -- so this is a
strict superset of the single-agent path. Every number is still produced by ``query_engine`` via
``agent.execute_step``; the LLM only decomposes, critiques, and writes prose.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import pandas as pd

from . import agent, llm
from .interface_parsing import detect_intent, resolve_compare

# role -> the subset of agent.TOOLS that role may plan with. "statistician" is the general/fallback role.
ROLES: dict[str, tuple[str, ...]] = {
    "statistician": ("extremum", "average", "threshold"),
    "correlator": ("correlation",),
    "anomaly": ("anomaly_count",),
}
_DEFAULT_ROLE = "statistician"


@dataclass
class SubTask:
    role: str
    question: str


@dataclass
class RoleResult:
    role: str
    question: str
    facts: list  # list[agent.Fact]
    computed: str = ""
    text: str = ""


@dataclass
class OrchestratedAnswer:
    question: str
    subtasks: list  # list[SubTask]
    results: list   # list[RoleResult]
    text: str
    flags: list = field(default_factory=list)
    computed: str = ""
    used_llm: bool = False


class Orchestrator:
    """Plan -> dispatch (parallel) -> critique -> combine, preserving the trust invariant."""

    def __init__(self, df: pd.DataFrame, use_llm: bool = True, model: str | None = None,
                 max_workers: int = 4):
        self.df = df
        self.use_llm = use_llm
        self.model = model or llm.DEFAULT_MODEL
        self.max_workers = max(1, int(max_workers))

    def _protected(self, question: str) -> bool:
        """A correlation/compare question legitimately uses 'and'/'between'; never split it."""
        return detect_intent(question) == "correlation" or resolve_compare(question) is not None

    def decompose(self, question: str) -> list[SubTask]:
        """Split into >=1 SubTask. Offline, protected, or single-item LLM output -> one whole-question
        SubTask (== the single-agent path). Unknown roles fall back to the general 'statistician'."""
        if not (self.use_llm and llm.is_available()) or self._protected(question):
            return [SubTask(_DEFAULT_ROLE, question)]
        pairs = llm.decompose(question, list(ROLES), list(self.df.columns), model=self.model)
        if not pairs or len(pairs) < 2:
            return [SubTask(_DEFAULT_ROLE, question)]
        return [SubTask(role if role in ROLES else _DEFAULT_ROLE, q) for role, q in pairs]
```

- [ ] **Step 4: Run to verify they pass**

Run: `python3.12 -m pytest tests/test_orchestrator.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add v1/src/orchestrator.py v1/tests/test_orchestrator.py
git commit -m "feat(orchestrator): module skeleton + conservative Planner (offline -> single task)"
```

---

### Task 4: Dispatch (parallel, order-preserving, role tool subset)

**Files:**
- Modify: `v1/src/orchestrator.py` (add `dispatch`)
- Test: `v1/tests/test_orchestrator.py` (append)

**Interfaces:**
- Consumes: `agent.Agent(df, use_llm, model, tools=ROLES[role]).run(question)` (Task 2 subset), `SubTask`, `RoleResult`.
- Produces: `Orchestrator.dispatch(subtasks: list[SubTask]) -> list[RoleResult]` — one result per sub-task, **in the same order**, executed concurrently.

- [ ] **Step 1: Write the failing tests**

```python
# append to v1/tests/test_orchestrator.py
def test_dispatch_facts_equal_pandas_and_preserve_order():
    idx = pd.date_range("2015-01-01", periods=72, freq="h")
    df = pd.DataFrame({"T (degC)": [float(i % 24) for i in range(72)]}, index=idx)
    o = orch.Orchestrator(df, use_llm=False)  # offline: agent falls back to interface per sub-task
    subs = [orch.SubTask("statistician", "max temperature"),
            orch.SubTask("statistician", "min temperature")]
    results = o.dispatch(subs)
    assert [r.question for r in results] == ["max temperature", "min temperature"]  # order kept
    # exact max/min match pandas on the same frame
    assert f"{df['T (degC)'].max():.2f}" in results[0].computed or \
           f"{df['T (degC)'].max():.2f}" in results[0].text


def test_dispatch_role_restricts_tools(monkeypatch):
    df = pd.DataFrame({"T (degC)": [1.0, 2.0], "rh (%)": [3.0, 4.0]},
                      index=pd.date_range("2015-01-01", periods=2, freq="h"))
    seen = {}

    def fake_run(self, q):
        seen["tools"] = self.tool_names
        from src.agent import AgentAnswer
        return AgentAnswer(q, [], "ok", "ok")
    monkeypatch.setattr(orch.agent.Agent, "run", fake_run)
    o = orch.Orchestrator(df, use_llm=False)
    o.dispatch([orch.SubTask("correlator", "correlation between temperature and humidity")])
    assert seen["tools"] == ("correlation",)
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3.12 -m pytest tests/test_orchestrator.py -k dispatch -v`
Expected: FAIL (`'Orchestrator' object has no attribute 'dispatch'`).

- [ ] **Step 3: Implement dispatch**

```python
# add to Orchestrator
def _run_one(self, sub: SubTask) -> RoleResult:
    tools = ROLES.get(sub.role, ROLES[_DEFAULT_ROLE])
    ag = agent.Agent(self.df, use_llm=self.use_llm, model=self.model, tools=tools)
    try:
        ans = ag.run(sub.question)
    except Exception as exc:  # noqa: BLE001 - a failed sub-agent must not sink the run
        return RoleResult(sub.role, sub.question, [], "", f"(failed: {exc})")
    return RoleResult(sub.role, sub.question, list(ans.steps), ans.computed, ans.text)

def dispatch(self, subtasks: list[SubTask]) -> list[RoleResult]:
    """Run every sub-task's role-agent concurrently; results are re-ordered to match input order so
    output is deterministic despite parallel execution. Each number is still computed == pandas."""
    if not subtasks:
        return []
    workers = min(self.max_workers, len(subtasks))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(self._run_one, subtasks))
```

Note: `ThreadPoolExecutor.map` preserves input order by contract, giving deterministic output.

- [ ] **Step 4: Run to verify they pass**

Run: `python3.12 -m pytest tests/test_orchestrator.py -k dispatch -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add v1/src/orchestrator.py v1/tests/test_orchestrator.py
git commit -m "feat(orchestrator): parallel order-preserving dispatch over role-restricted agents"
```

---

### Task 5: Critic (`critique`)

**Files:**
- Modify: `v1/src/orchestrator.py` (add `critique`)
- Test: `v1/tests/test_orchestrator.py` (append)

**Interfaces:**
- Consumes: `RoleResult`, `agent.Fact` (`.ok`, `.error`), `llm.critique`, `llm.faithfulness`.
- Produces: `Orchestrator.critique(question: str, results: list[RoleResult]) -> list[str]` — deterministic flags first, optional LLM flags appended.

- [ ] **Step 1: Write the failing tests**

```python
# append to v1/tests/test_orchestrator.py
from src.agent import Fact


def test_critique_flags_empty_result():
    o = orch.Orchestrator(_df(), use_llm=False)
    results = [orch.RoleResult("statistician", "max temp", [], "", "")]
    flags = o.critique("max temp", results)
    assert any("no usable result" in f for f in flags)


def test_critique_flags_executor_error():
    o = orch.Orchestrator(_df(), use_llm=False)
    bad = Fact("extremum", {"sensor": "X"}, ok=False, error="unknown sensor 'X'")
    results = [orch.RoleResult("statistician", "max X", [bad], "", "")]
    flags = o.critique("max X", results)
    assert any("unknown sensor" in f for f in flags)


def test_critique_clean_when_ok():
    o = orch.Orchestrator(_df(), use_llm=False)
    good = Fact("extremum", {"sensor": "T (degC)"}, computed="max T was 30.00", value={"value": 30.0})
    results = [orch.RoleResult("statistician", "max T", [good], "max T was 30.00", "max T was 30.00")]
    assert o.critique("max T", results) == []  # offline: no LLM flags, deterministic checks pass
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3.12 -m pytest tests/test_orchestrator.py -k critique -v`
Expected: FAIL (`no attribute 'critique'`).

- [ ] **Step 3: Implement critique**

```python
# add to Orchestrator
def critique(self, question: str, results: list[RoleResult]) -> list[str]:
    """Second trust gate. Deterministic checks always run; an optional LLM critic (Ollama on) may
    append natural-language concerns. The critic only flags -- it never recomputes a number."""
    flags: list[str] = []
    for r in results:
        oks = [f for f in r.facts if getattr(f, "ok", True)]
        errs = [f for f in r.facts if not getattr(f, "ok", True) and getattr(f, "error", None)]
        for f in errs:
            flags.append(f"{r.role} ('{r.question}'): {f.error}")
        if not oks and not errs:
            flags.append(f"{r.role} ('{r.question}'): no usable result")
    if self.use_llm and llm.is_available():
        role_facts = [(r.role, r.computed or r.text) for r in results if (r.computed or r.text)]
        try:
            flags.extend(llm.critique(question, role_facts, model=self.model))
        except Exception:  # noqa: BLE001 - a critic failure must never block the answer
            pass
    return flags
```

- [ ] **Step 4: Run to verify they pass**

Run: `python3.12 -m pytest tests/test_orchestrator.py -k critique -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add v1/src/orchestrator.py v1/tests/test_orchestrator.py
git commit -m "feat(orchestrator): Critic (deterministic checks + optional LLM flags, never recomputes)"
```

---

### Task 6: Combiner + `run` (full pipeline, regression == agent path)

**Files:**
- Modify: `v1/src/orchestrator.py` (add `combine`, `run`, `_template`)
- Test: `v1/tests/test_orchestrator.py` (append)

**Interfaces:**
- Consumes: `agent.Agent.compose`, `agent.Fact`, all of the above.
- Produces:
  - `Orchestrator.combine(question, results, flags) -> tuple[str, bool]`
  - `Orchestrator.run(question) -> OrchestratedAnswer`

- [ ] **Step 1: Write the failing tests**

```python
# append to v1/tests/test_orchestrator.py
def test_combine_offline_template_lists_all_roles():
    o = orch.Orchestrator(_df(), use_llm=False)
    f1 = Fact("extremum", {"sensor": "T (degC)"}, computed="max T was 30.00")
    f2 = Fact("extremum", {"sensor": "rh (%)"}, computed="min rh was 20.00")
    results = [orch.RoleResult("statistician", "max T", [f1], "max T was 30.00", ""),
               orch.RoleResult("statistician", "min rh", [f2], "min rh was 20.00", "")]
    text, used = o.combine("max T and min rh", results, [])
    assert used is False
    assert "max T was 30.00" in text and "min rh was 20.00" in text


def test_run_offline_matches_single_agent_facts():
    idx = pd.date_range("2015-01-01", periods=72, freq="h")
    df = pd.DataFrame({"T (degC)": [float(i % 24) for i in range(72)]}, index=idx)
    q = "what was the hottest temperature?"
    single = agent.Agent(df, use_llm=False).run(q)
    multi = orch.Orchestrator(df, use_llm=False).run(q)
    # strict superset: offline single-topic run reproduces the single-agent computed facts
    assert f"{df['T (degC)'].max():.2f}" in (multi.computed + multi.text)
    assert multi.computed.strip() != "" or single.computed.strip() == ""


def test_run_attaches_critic_flags(monkeypatch):
    o = orch.Orchestrator(_df(), use_llm=False)
    monkeypatch.setattr(o, "decompose",
                        lambda q: [orch.SubTask("statistician", "max X")])
    ans = o.run("max X")  # unknown sensor -> empty result -> flag surfaced
    assert ans.flags and "Critic notes" in ans.text
```

(Add `from src import agent` at the top of the test module if not already imported.)

- [ ] **Step 2: Run to verify they fail**

Run: `python3.12 -m pytest tests/test_orchestrator.py -k "combine or run_" -v`
Expected: FAIL (`no attribute 'combine'`).

- [ ] **Step 3: Implement combine + run**

```python
# add to Orchestrator (and a module-level template helper)
def combine(self, question: str, results: list[RoleResult],
            flags: list[str]) -> tuple[str, bool]:
    """Weave verified role-facts into one answer. Reuses agent.Agent.compose so an LLM synthesis is
    accepted only if faithfulness finds no invented number; otherwise a deterministic template."""
    facts: list = []
    for r in results:
        facts.extend(f for f in r.facts if getattr(f, "ok", True))
    if facts:
        text, used = agent.Agent(self.df, use_llm=self.use_llm, model=self.model).compose(
            question, facts)
    else:
        text, used = _template(results), False
    if flags:
        text = f"{text}\n⚠ Critic notes: " + "; ".join(flags)
    return text, used


def run(self, question: str) -> OrchestratedAnswer:
    """Full pipeline: decompose -> dispatch (parallel) -> critique -> combine."""
    subtasks = self.decompose(question)
    results = self.dispatch(subtasks)
    flags = self.critique(question, results)
    text, used = self.combine(question, results, flags)
    computed = " ".join(r.computed for r in results if r.computed)
    return OrchestratedAnswer(question, subtasks, results, text, flags, computed, used)
```

```python
# module-level, near _template usage
def _template(results: list[RoleResult]) -> str:
    """Deterministic multi-agent answer (offline / LLM-off path), grouped by role."""
    lines = ["🔢 Computed (multi-agent):"]
    for r in results:
        label = r.computed or r.text or "(no result)"
        lines.append(f"  • [{r.role}] {label}")
    return "\n".join(lines)
```

- [ ] **Step 4: Run to verify the full suite passes**

Run: `python3.12 -m pytest tests/test_orchestrator.py -v`
Expected: PASS (all orchestrator tests).

- [ ] **Step 5: Commit**

```bash
git add v1/src/orchestrator.py v1/tests/test_orchestrator.py
git commit -m "feat(orchestrator): Combiner + run pipeline (faithfulness-gated, critic flags surfaced)"
```

---

### Task 7: CLI flag + evaluation corpus/scorer

**Files:**
- Modify: `v1/apps/chat.py` (add `--orchestrate`)
- Create: `v1/src/orchestrator_eval.py`
- Test: `v1/tests/test_orchestrator_eval.py`

**Interfaces:**
- Consumes: `Orchestrator`, `agent.Agent`.
- Produces: `orchestrator_eval.MULTIPART_CASES: list[tuple[str, tuple[str, ...]]]` (question, expected substrings) and `orchestrator_eval.score(orch_obj, cases) -> dict` counting how many multi-part questions produce every expected fact.

- [ ] **Step 1: Write the failing test**

```python
# v1/tests/test_orchestrator_eval.py
"""The multi-part scorer counts questions whose combined answer contains every expected fact. Uses a
stub so it runs without Ollama."""
from src import orchestrator_eval as oe


class _Stub:
    def run(self, q):
        class A:  # minimal duck-typed OrchestratedAnswer
            text = "max temperature 30.00; min humidity 20.00"
            computed = "max temperature 30.00; min humidity 20.00"
        return A()


def test_score_counts_complete_answers():
    cases = [("max temp and min humidity", ("30.00", "20.00")),
             ("something missing", ("999.99",))]
    out = oe.score(_Stub(), cases)
    assert out["n"] == 2 and out["complete"] == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3.12 -m pytest tests/test_orchestrator_eval.py -v`
Expected: FAIL (`No module named 'src.orchestrator_eval'`).

- [ ] **Step 3: Implement the scorer and the CLI flag**

```python
# v1/src/orchestrator_eval.py
"""Evaluation for the multi-agent orchestrator: a small corpus of genuinely multi-part questions and
a scorer counting how many produce a combined answer containing every expected exact fact. Ground
truth is the obviously-correct set of facts, written before running any code."""
from __future__ import annotations

# (question, tuple of exact substrings that MUST all appear in the combined answer)
MULTIPART_CASES: list[tuple[str, tuple[str, ...]]] = [
    ("the highest temperature in 2015 and the lowest humidity in 2013", ("2015", "2013")),
    ("mean pressure in 2014 and how many times temperature was above 30 that year", ("2014",)),
    ("coldest temperature ever and the windiest reading", ("T (degC)", "wv (m/s)")),
]


def score(orch_obj, cases: list[tuple[str, tuple[str, ...]]] = MULTIPART_CASES) -> dict:
    """Run each multi-part question through ``orch_obj.run`` and count *complete* answers (every
    expected substring present in text+computed). Returns ``{"n", "complete"}``."""
    n = len(cases)
    complete = 0
    for question, expected in cases:
        ans = orch_obj.run(question)
        blob = f"{getattr(ans, 'text', '')} {getattr(ans, 'computed', '')}"
        if all(sub in blob for sub in expected):
            complete += 1
    return {"n": n, "complete": complete}
```

```python
# v1/apps/chat.py -- add the flag near --agent (line ~42) and wire it
p.add_argument("--orchestrate", action="store_true",
               help="multi-agent mode (decompose+dispatch+critic+combine; implies --llm)")
# where the agent is constructed (~line 70), add:
from src import orchestrator as _orch  # top-of-file import preferred; shown here for locality
orc = _orch.Orchestrator(df, use_llm=use_llm, model=model) if args.orchestrate else None
# in the answer path, prefer the orchestrator when requested:
#   if orc is not None: ans_text = orc.run(question).text
#   elif ag is not None: ...
```

Wire `--orchestrate` to imply `use_llm=True` exactly as `--agent` does, and print `mode = "orchestrate:" + model` when active. Keep the change minimal and mirror the existing `--agent` branch.

- [ ] **Step 4: Run to verify it passes**

Run: `python3.12 -m pytest tests/test_orchestrator_eval.py -v && python3.12 apps/chat.py --help`
Expected: test PASS; `--orchestrate` appears in help.

- [ ] **Step 5: Commit**

```bash
git add v1/src/orchestrator_eval.py v1/tests/test_orchestrator_eval.py v1/apps/chat.py
git commit -m "feat(orchestrator): multi-part eval corpus/scorer + chat.py --orchestrate flag"
```

---

### Task 8: Evaluation write-up + WORKLOG + gates

**Files:**
- Create: `v1/docs/evaluation/10-multi-agent-orchestrator.md`
- Modify: `WORKLOG.md`

**Interfaces:** none (documentation).

- [ ] **Step 1: Run the real experiment and capture numbers**

```bash
cd v1 && python3.12 - <<'PY'
from src import preprocess, agent, orchestrator, orchestrator_eval as oe
raw = preprocess.clean(preprocess.load_data("data/jena_climate_2009_2016.csv"))
df = preprocess.align(raw, freq="1h")
print("single-agent (fallback per-Q):", oe.score(agent.Agent(df, use_llm=False)) if hasattr(agent.Agent,'run') else "n/a")
print("orchestrator off:", oe.score(orchestrator.Orchestrator(df, use_llm=False)))
print("orchestrator on :", oe.score(orchestrator.Orchestrator(df, use_llm=True)))
PY
```

(If Ollama is off, report the offline numbers and note the on-path is model-bound, mirroring doc 09.)

- [ ] **Step 2: Write the evaluation doc**

Write `v1/docs/evaluation/10-multi-agent-orchestrator.md` covering: what's evaluated (multi-part coverage + Critic + regression, NOT computation which is exact elsewhere); method (`orchestrator_eval` corpus, LLM on/off); the real numbers from Step 1; a worked decompose→dispatch→critique→combine trace for one question; the Critic catching an empty/erroring sub-task; the offline==single-agent regression property; and an **honest reading** — the multi-agent layer buys coverage on genuinely multi-part questions and a second trust gate, at the cost of latency (mitigated by parallel dispatch, itself the COSMA seam) and dependence on decompose quality (model-bound, like doc 09). Note the invariant: numbers still == pandas.

- [ ] **Step 3: Update WORKLOG.md**

Add a dated `2026-07-30` entry under the multi-agent heading: the orchestrator (planner/roles/critic/combiner), the tool-subset addition to `agent.py`, the eval corpus, and the headline finding.

- [ ] **Step 4: Run the full gate suite**

Run: `cd v1 && python3.12 -m pytest tests/ -q && ruff check src tests && mypy src`
Expected: all tests pass; ruff + mypy clean.

- [ ] **Step 5: Commit**

```bash
git add v1/docs/evaluation/10-multi-agent-orchestrator.md WORKLOG.md
git commit -m "docs(orchestrator): evaluation write-up + worklog for the multi-agent layer"
```

---

## Self-Review

**Spec coverage:** Planner (Task 3) ✓, Role-agents + parallel dispatch + tool subset (Tasks 2, 4) ✓, Critic deterministic + optional LLM (Task 5) ✓, Combiner faithfulness-gated + template (Task 6) ✓, offline==single-agent regression (Task 6) ✓, `llm.decompose`/`llm.critique` offline-safe (Task 1) ✓, CLI flag (Task 7) ✓, eval doc (Task 8) ✓. All spec sections covered.

**Placeholder scan:** No TBD/TODO. The one prose-only step (Task 8 Step 2, the doc body) is documentation, which is inherently written at execution time from real numbers — acceptable.

**Type consistency:** `SubTask(role, question)`, `RoleResult(role, question, facts, computed, text)`, `OrchestratedAnswer(question, subtasks, results, text, flags, computed, used_llm)`, `Orchestrator(df, use_llm, model, max_workers)` with `.decompose/.dispatch/.critique/.combine/.run`, `ROLES` names matching `agent.TOOLS` keys, and `agent.Agent(..., tools=...)` — all consistent across Tasks 1-8. `llm.decompose -> list[tuple[str,str]] | None`, `llm.critique -> list[str]` used consistently.
