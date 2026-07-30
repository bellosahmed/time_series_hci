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

# role -> the subset of agent.TOOLS that role may plan with. "statistician" is the general/fallback
# role (max/min/mean/threshold); "correlator" owns correlation; "anomaly" owns anomaly counting.
ROLES: dict[str, tuple[str, ...]] = {
    "statistician": ("extremum", "average", "threshold"),
    "correlator": ("correlation",),
    "anomaly": ("anomaly_count",),
}
_DEFAULT_ROLE = "statistician"


def _template(results: list[RoleResult]) -> str:
    """Deterministic multi-agent answer (offline / LLM-off path), grouped by role."""
    lines = ["🔢 Computed (multi-agent):"]
    for r in results:
        label = r.computed or r.text or "(no result)"
        lines.append(f"  • [{r.role}] {label}")
    return "\n".join(lines)


@dataclass
class SubTask:
    """One independent piece of a decomposed question, tagged with the role that should answer it."""

    role: str
    question: str


@dataclass
class RoleResult:
    """The output of one role-agent: its exact facts (== pandas) plus its draft text."""

    role: str
    question: str
    facts: list  # list[agent.Fact]
    computed: str = ""
    text: str = ""


@dataclass
class OrchestratedAnswer:
    """The full multi-agent result: sub-tasks, per-role results, the combined answer, and any flags."""

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
        """Split into >=1 SubTask. Offline, protected, or single-item LLM output -> one
        whole-question SubTask (== the single-agent path). Unknown roles fall back to the general
        'statistician' role rather than being dropped."""
        if not (self.use_llm and llm.is_available()) or self._protected(question):
            return [SubTask(_DEFAULT_ROLE, question)]
        pairs = llm.decompose(question, list(ROLES), list(self.df.columns), model=self.model)
        if not pairs or len(pairs) < 2:
            return [SubTask(_DEFAULT_ROLE, question)]
        return [SubTask(role if role in ROLES else _DEFAULT_ROLE, q) for role, q in pairs]

    def _run_one(self, sub: SubTask) -> RoleResult:
        """Run one sub-task through a role-restricted agent; a failure becomes a flagged result,
        never an exception that sinks the whole run."""
        tools = ROLES.get(sub.role, ROLES[_DEFAULT_ROLE])
        ag = agent.Agent(self.df, use_llm=self.use_llm, model=self.model, tools=tools)
        try:
            ans = ag.run(sub.question)
        except Exception as exc:  # noqa: BLE001 - a failed sub-agent must not sink the run
            return RoleResult(sub.role, sub.question, [], "", f"(failed: {exc})")
        return RoleResult(sub.role, sub.question, list(ans.steps), ans.computed, ans.text)

    def dispatch(self, subtasks: list[SubTask]) -> list[RoleResult]:
        """Run every sub-task's role-agent concurrently; results are re-ordered to match input order
        so output is deterministic despite parallel execution. Each number is still == pandas."""
        if not subtasks:
            return []
        workers = min(self.max_workers, len(subtasks))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(self._run_one, subtasks))

    def critique(self, question: str, results: list[RoleResult]) -> list[str]:
        """Second trust gate. Deterministic checks always run; an optional LLM critic (Ollama on) may
        append natural-language concerns. The critic only flags -- it never recomputes a number."""
        flags: list[str] = []
        for r in results:
            oks = [f for f in r.facts if getattr(f, "ok", True)]
            errs = [f for f in r.facts if not getattr(f, "ok", True) and getattr(f, "error", None)]
            for f in errs:
                flags.append(f"{r.role} ('{r.question}'): {f.error}")
            # A sub-task is usable if it produced ok facts OR an exact computed string -- the latter
            # covers a sub-agent that fell back to the interface (empty `facts`, but a real, exact
            # answer, e.g. the interface's own compound splitter). Only a genuinely empty result
            # (no facts, no computed number) is flagged.
            if not oks and not errs and not (r.computed or "").strip():
                flags.append(f"{r.role} ('{r.question}'): no usable result")
        if self.use_llm and llm.is_available():
            role_facts = [(r.role, r.computed or r.text) for r in results if (r.computed or r.text)]
            try:
                flags.extend(llm.critique(question, role_facts, model=self.model))
            except Exception:  # noqa: BLE001 - a critic failure must never block the answer
                pass
        return flags

    def combine(self, question: str, results: list[RoleResult],
                flags: list[str]) -> tuple[str, bool]:
        """Weave verified role-facts into one answer. Reuses agent.Agent.compose so an LLM synthesis
        is accepted only if faithfulness finds no invented number; otherwise a deterministic
        template. Any Critic flags are appended so a wrong reading is visible, not silent."""
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
