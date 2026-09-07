"""Comparative benchmark: our trust-layer app vs. general-purpose LLMs on the *same* questions.

Motivation
----------
The truth layer proves our numbers equal pandas (see ``docs/evaluation/01-truth-layer.md``). That
answers "are we self-consistent?" but not "are we *better than just asking an LLM*?" — the obvious
thing a user would otherwise do. This module runs a like-for-like contest:

    same question  +  same data in the prompt  ->  { our app , local LLMs , (manually) ChatGPT/… }

and scores every contestant against one independent ground truth (the oracle: the exact answer
recomputed from the raw CSV, with the same de-duplication + sentinel handling the app uses).

Fairness
--------
Each gold question is scoped to a **small window** whose raw 10-min readings are embedded verbatim
in the prompt. Nobody guesses blind — the data is right there; the only question is whether the
contestant computes over it correctly. This isolates *computation/reading* skill from *data access*,
which is the fair comparison (our app's advantage is a deterministic engine, not secret data).

The identical prompt string is reused for the local models here, for the manual pack that a human
pastes into ChatGPT/Kimi/GLM, and for Claude answering inline — so all contestants face the exact
same task.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from . import config

DATA = config.DATA_PATH


@dataclass(frozen=True)
class Case:
    question: str          # natural-language question given to our app (and, reworded, to LLMs)
    sensor: str            # dataset column the question is about
    kind: str              # "max" | "min" | "mean"
    window: str            # pandas .loc slice for the scoped window (small enough to embed)
    label: str             # human label for the window, e.g. "4 July 2015"


# Small windows (a single day = 144 ten-min rows) keep the embedded table compact while still being
# a real, non-trivial computation. Spread across sensors so no contestant can special-case one.
GOLD: list[Case] = [
    Case("what was the highest temperature on 4 July 2015?",   "T (degC)",    "max",  "2015-07-04", "4 July 2015"),
    Case("what was the lowest temperature on 4 July 2015?",    "T (degC)",    "min",  "2015-07-04", "4 July 2015"),
    Case("what was the average temperature on 4 July 2015?",   "T (degC)",    "mean", "2015-07-04", "4 July 2015"),
    Case("what was the maximum humidity on 2 October 2012?",   "rh (%)",      "max",  "2012-10-02", "2 October 2012"),
    Case("what was the minimum pressure on 15 January 2014?",  "p (mbar)",    "min",  "2014-01-15", "15 January 2014"),
    Case("what was the highest wind speed on 10 February 2016?", "wv (m/s)",  "max",  "2016-02-10", "10 February 2016"),
    Case("what was the highest dew point on 20 August 2011?",  "Tdew (degC)", "max",  "2011-08-20", "20 August 2011"),
    Case("what was the average humidity on 2 October 2012?",   "rh (%)",      "mean", "2012-10-02", "2 October 2012"),
]


def load_raw() -> pd.DataFrame:
    """Independent raw frame straight from the CSV, matching the app's cleaning decisions
    (drop duplicate timestamps keep-first; treat -9999 sentinels as missing). No interpolation:
    these are point/aggregate stats over the real readings only."""
    df = pd.read_csv(DATA)
    df["Date Time"] = pd.to_datetime(df["Date Time"], format="%d.%m.%Y %H:%M:%S")
    df = df.set_index("Date Time").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df.mask(df <= -9999.0)


def ground_truth(raw: pd.DataFrame, case: Case) -> float:
    s = raw.loc[case.window, case.sensor].dropna()
    if case.kind == "max":
        return float(s.max())
    if case.kind == "min":
        return float(s.min())
    return float(s.mean())


def window_table(raw: pd.DataFrame, case: Case) -> str:
    """Compact `HH:MM,value` table of the window's readings for one sensor — the data every
    contestant is given."""
    s = raw.loc[case.window, case.sensor].dropna()
    lines = [f"{ts:%Y-%m-%d %H:%M},{v:.2f}" for ts, v in s.items()]
    return "\n".join(lines)


_STAT_WORD = {"max": "highest (maximum)", "min": "lowest (minimum)", "mean": "average (mean)"}


def build_prompt(raw: pd.DataFrame, case: Case) -> str:
    """The exact task string handed to every LLM contestant (local, manual, and inline)."""
    table = window_table(raw, case)
    return (
        f"Below are the '{case.sensor}' readings for {case.label} (one row per 10 minutes, "
        f"as `timestamp,value`).\n\n"
        f"{table}\n\n"
        f"Question: what was the {_STAT_WORD[case.kind]} '{case.sensor}' value that day?\n"
        f"Answer with ONLY the number (no units, no words, rounded to 2 decimals)."
    )


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_number(text: str) -> float | None:
    """Pull a single numeric answer out of a free-text reply. Prefers the last number on the
    final non-empty line (models often end with the answer), else the first number seen."""
    if not text:
        return None
    for line in reversed([ln for ln in text.splitlines() if ln.strip()]):
        nums = _NUM_RE.findall(line)
        if nums:
            return float(nums[-1])
    nums = _NUM_RE.findall(text)
    return float(nums[0]) if nums else None


def ask_app(iface, case: Case) -> float | None:
    """Our application's answer via the full pipeline."""
    ans = iface.ask(case.question)
    v = ans.value
    if isinstance(v, dict):
        if "value" in v:
            return float(v["value"])
        if case.kind == "mean" and "mean" in v:
            return float(v["mean"])
    return None


def ask_local(model: str, raw: pd.DataFrame, case: Case, timeout: float = 120.0) -> tuple[float | None, str]:
    """A local Ollama model as a raw contestant: give it the data + question, parse its number."""
    from src import llm
    reply = llm.generate(build_prompt(raw, case), model=model, system="", timeout=timeout)
    return parse_number(reply), reply


def is_correct(pred: float | None, truth: float, tol: float = 0.01) -> bool:
    return pred is not None and abs(pred - truth) <= tol
