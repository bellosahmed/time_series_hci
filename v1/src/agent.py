"""Agentic, still-numberless tool-use over the truth layer.

Single-step questions ("hottest in 2013?") are answered directly by `interface.Interface`.
This module handles *multi-step* questions — "compare the three hottest summers and explain
why one stood out" — that no single query needs.

The design keeps the project's core guarantee. The LLM only **plans**: it chooses which
deterministic query-engine tools to call, in what order, with which arguments (sensors,
dates, operator). Every number in the answer is produced by **executing** a tool against
pandas here; the LLM never computes a value. A final synthesis step weaves the exact facts
into prose and is checked by the same `llm.faithfulness` gate as the single-step path, so an
agent's flexibility is gained without reintroducing hallucination.

Flow:  question -> LLM plan (list of tool calls)
                -> execute each tool deterministically (exact facts, == pandas)
                -> LLM synthesis (faithfulness-gated) or template fallback

The whole executor is pure Python and fully testable offline: `execute_step` / `execute_plan`
run without any LLM, and `compose` falls back to a trust-labelled template when Ollama is off.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import analysis, llm
from . import query_engine as qe
from .interface_parsing import _coerce_date


@dataclass
class Fact:
    """One executed tool call: the exact computed fact plus its structured value."""

    tool: str
    args: dict
    computed: str = ""
    value: object = None
    ok: bool = True
    error: str | None = None


@dataclass
class AgentAnswer:
    """The result of an agentic run: the per-step facts and the composed answer."""

    question: str
    steps: list  # list[Fact]
    text: str
    computed: str = ""
    used_llm: bool = False
    plan: list = field(default_factory=list)


def _scope(df: pd.DataFrame, start, end) -> tuple[pd.DataFrame | None, str]:
    """Slice to a date window, returning (data, human label); (None, label) if it has no rows."""
    return qe.slice_window(df, start, end)


# --- Tool executors -----------------------------------------------------------------------
# Each takes the DataFrame plus already-validated args and returns (computed_str, value).
# They delegate every number to `query_engine` / `analysis`, so tool output == pandas.

def _run_extremum(df, sensor, kind, start=None, end=None, **_):
    data, where = _scope(df, start, end)
    if data is None:
        return f"I have no {sensor} data for {where}.", None
    if sensor in analysis.CIRCULAR_SENSORS:  # a plain max/min on a compass bearing is meaningless
        cs = analysis.circular_stats_deg(data[sensor])
        return (f"{sensor}{where} is a compass bearing; prevailing direction {cs['mean']:.1f}° "
                f"({analysis.compass_point(cs['mean'])}), circular std {cs['std']:.1f}°."), cs
    r = qe.get_max(data, sensor) if kind == "max" else qe.get_min(data, sensor)
    return f"{kind} {sensor}{where} was {r['value']:.2f} on {qe.fmt_when(r['when'])}.", r


def _run_average(df, sensor, start=None, end=None, **_):
    data, where = _scope(df, start, end)
    if data is None:
        return f"I have no {sensor} data for {where}.", None
    if sensor in analysis.CIRCULAR_SENSORS:
        cs = analysis.circular_stats_deg(data[sensor])
        return (f"{sensor}{where}: prevailing (circular mean) {cs['mean']:.1f}° "
                f"({analysis.compass_point(cs['mean'])}), circular std {cs['std']:.1f}°."), cs
    r = qe.get_mean_std(data, sensor)
    return (f"{sensor}{where}: mean={r['mean']:.2f}, std={r['std']:.2f}, "
            f"range={r['range']:.2f}."), r


def _run_correlation(df, a, b, start=None, end=None, **_):
    data, where = _scope(df, start, end)
    if data is None:
        return f"I have no {a}/{b} data for {where}.", None
    circ = [s for s in (a, b) if s in analysis.CIRCULAR_SENSORS]
    if len(circ) == 1:  # one compass angle + one linear sensor -> circular-linear association
        deg = circ[0]
        lin = b if deg == a else a
        c = analysis.circular_linear_corr(data[deg], data[lin])
        return (f"circular-linear correlation({deg}, {lin}){where} = {c:.3f} "
                f"(0-1, unsigned)."), c
    c = qe.get_correlation(data, a, b)
    return f"correlation({a}, {b}){where} = {c:.3f}.", c


def _run_threshold(df, sensor, op, value, start=None, end=None, **_):
    data, where = _scope(df, start, end)
    if data is None:
        return f"I have no {sensor} data for {where}.", None
    r = qe.threshold_events(data, sensor, op, value)
    sign = "above" if op == ">" else "below"
    when = f", first on {qe.fmt_when(r['first'])}" if r["first"] is not None else ""
    return (f"{sensor} was {sign} {value:g}{where} {r['count']} times "
            f"({r['pct']:.2f}% of readings){when}."), r


def _run_anomaly_count(df, sensor, start=None, end=None, **_):
    data, where = _scope(df, start, end)
    if data is None:
        return f"I have no {sensor} data for {where}.", None
    decomp = analysis.decompose(data[sensor], period=24)
    resid = decomp["resid"].dropna()
    from . import anomaly
    flags = anomaly.zscore_anomalies(resid, threshold=4.0)
    windows = qe.get_anomaly_windows(flags)
    return (f"{int(flags.sum())} anomalous {sensor} readings{where} in "
            f"{len(windows)} windows."), windows


# name -> {description, params (for the planner prompt), run (executor), needs (required args)}
TOOLS: dict[str, dict] = {
    "extremum": {
        "description": "Exact max or min of one sensor over an optional date window.",
        "params": ("sensor", "kind='max'|'min'", "start=YYYY-MM-DD|null", "end=YYYY-MM-DD|null"),
        "run": _run_extremum, "needs": ("sensor", "kind"),
    },
    "average": {
        "description": "Mean, std and range of one sensor over an optional date window.",
        "params": ("sensor", "start=YYYY-MM-DD|null", "end=YYYY-MM-DD|null"),
        "run": _run_average, "needs": ("sensor",),
    },
    "correlation": {
        "description": "Pearson correlation between two sensors over an optional date window.",
        "params": ("a", "b", "start=YYYY-MM-DD|null", "end=YYYY-MM-DD|null"),
        "run": _run_correlation, "needs": ("a", "b"),
    },
    "threshold": {
        "description": "How often a sensor was above ('>') or below ('<') a value in a window.",
        "params": ("sensor", "op='>'|'<'", "value", "start=YYYY-MM-DD|null", "end=YYYY-MM-DD|null"),
        "run": _run_threshold, "needs": ("sensor", "op", "value"),
    },
    "anomaly_count": {
        "description": "Count anomalous readings (and windows) for one sensor in a date window.",
        "params": ("sensor", "start=YYYY-MM-DD|null", "end=YYYY-MM-DD|null"),
        "run": _run_anomaly_count, "needs": ("sensor",),
    },
}


def execute_step(df: pd.DataFrame, step: dict) -> Fact:
    """Validate and run a single planned tool call against `df` (deterministic, no LLM).

    Rejects unknown tools, unknown/missing sensors, and bad operator/kind/value — returning a
    ``Fact`` with ``ok=False`` and an ``error`` rather than raising, so one bad step never
    sinks the whole plan. On success ``computed`` is the exact fact and ``value`` its structure.
    """
    name = step.get("tool")
    args = dict(step.get("args") or {})
    spec = TOOLS.get(name)
    if spec is None:
        return Fact(str(name), args, ok=False, error=f"unknown tool {name!r}")

    cols = list(df.columns)
    for key in ("sensor", "a", "b"):  # every named column must exist in the data
        if key in args and args[key] not in cols:
            return Fact(name, args, ok=False, error=f"unknown sensor {args[key]!r}")
    for key in spec["needs"]:
        if args.get(key) in (None, ""):
            return Fact(name, args, ok=False, error=f"missing argument {key!r}")
    if "kind" in args and args["kind"] not in ("max", "min"):
        return Fact(name, args, ok=False, error=f"bad kind {args['kind']!r}")
    if "op" in args and args["op"] not in (">", "<"):
        return Fact(name, args, ok=False, error=f"bad op {args['op']!r}")
    if "value" in args:
        try:
            args["value"] = float(args["value"])
        except (TypeError, ValueError):
            return Fact(name, args, ok=False, error=f"bad value {args.get('value')!r}")
    # Coerce/validate any supplied dates (drop unparseable ones -> whole record).
    for key in ("start", "end"):
        if key in args:
            args[key] = _coerce_date(args[key])

    try:
        computed, value = spec["run"](df, **args)
    except Exception as exc:  # any pandas/key error becomes a failed step, not a crash
        return Fact(name, args, ok=False, error=str(exc))
    return Fact(name, args, computed=computed, value=value)


def _template(facts: list[Fact]) -> str:
    """Deterministic multi-step answer (the offline fallback / LLM-off path)."""
    lines = ["🔢 Computed (multi-step):"]
    lines += [f"  • {f.computed}" for f in facts]
    return "\n".join(lines)


class Agent:
    """Plan → execute → synthesize an answer to a multi-step question.

    ``use_llm=False`` (or Ollama unavailable) still works: pass an explicit plan to
    ``execute_plan`` and get exact facts, and ``compose`` returns the trust-labelled template.
    """

    def __init__(self, df: pd.DataFrame, use_llm: bool = True, model: str | None = None):
        self.df = df
        self.use_llm = use_llm
        self.model = model or llm.DEFAULT_MODEL

    def tool_specs(self) -> list[dict]:
        """The tool catalogue passed to the LLM planner."""
        return [{"name": n, "description": t["description"], "params": t["params"]}
                for n, t in TOOLS.items()]

    def plan(self, question: str) -> list[dict] | None:
        """Ask the LLM for an ordered list of tool calls (None if LLM is off/unavailable)."""
        if not (self.use_llm and llm.is_available()):
            return None
        return llm.plan_steps(question, self.tool_specs(), list(self.df.columns), model=self.model)

    def execute_plan(self, steps: list[dict]) -> list[Fact]:
        """Run every planned step deterministically; each result is exact (== pandas)."""
        return [execute_step(self.df, st) for st in steps]

    def compose(self, question: str, facts: list[Fact]) -> tuple[str, bool]:
        """Turn executed facts into an answer; returns (text, used_llm).

        Tries an LLM synthesis first, but only accepts it if the faithfulness gate finds no
        invented number; otherwise (and always when the LLM is off) returns the template.
        """
        good = [f for f in facts if f.ok]
        computed = " ".join(f.computed for f in good)
        if self.use_llm and llm.is_available() and good:
            try:
                narrative = llm.explain_synthesis(question, [f.computed for f in good],
                                                  model=self.model)
                if not llm.faithfulness(computed, narrative)["invented"]:
                    return narrative, True
            except Exception:
                pass  # fall through to the deterministic template
        return _template(good), False

    def run(self, question: str) -> AgentAnswer:
        """Full agentic answer: plan, execute, compose.

        If the LLM produces no usable plan (off, unavailable, or every step invalid), falls
        back to the single-step `interface.Interface` so the user always gets an answer.
        """
        plan = self.plan(question) or []
        facts = self.execute_plan(plan) if plan else []
        if not any(f.ok for f in facts):
            from .interface import Interface  # lazy import avoids a module cycle
            ans = Interface(self.df, use_llm=self.use_llm, model=self.model).ask(question)
            return AgentAnswer(question, [], ans.text, ans.computed,
                               used_llm=bool(self.use_llm and llm.is_available()), plan=plan)
        text, used = self.compose(question, facts)
        computed = " ".join(f.computed for f in facts if f.ok)
        return AgentAnswer(question, facts, text, computed, used_llm=used, plan=plan)
