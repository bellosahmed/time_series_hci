# Understanding/Clarifier Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When the deterministic parser can't route a query, let the local LLM rewrite it into a routable standalone question — accepted only if the deterministic parser can route the rewrite — so the system answers typo-heavy/paraphrased queries it currently rejects, with numbers still computed by code.

**Architecture:** A new `llm.understand()` proposes a rewrite (never a number). `Interface.ask` calls it **only** on queries the deterministic parser fails to route, and accepts the rewrite **only if `_routable`** passes. The interpretation is surfaced to the user. Offline (no Ollama) → today's behaviour unchanged.

**Tech Stack:** Python 3.12, stdlib `urllib` → local Ollama (via existing `src/llm.py`), pandas, pytest, ruff, mypy. No new dependencies. **No LangChain** (see `docs/related-work.md` / project decision).

## Global Constraints

- Run everything with `python3.12`.
- Numbers are computed by the deterministic query engine ONLY; the LLM never emits a value.
- Offline degradation: if `llm.is_available()` is False, behaviour must be byte-identical to the current deterministic path (regression guard required).
- Commits authored **solely by the user** — **NO** `Co-Authored-By: Claude` trailer. **Do not push.**
- Quality gate after each task: `python3.12 -m pytest -q`, `ruff check`, `python3.12 -m mypy src/` all clean.
- Keep `code/WORKLOG.md` updated (final task).
- Reproducibility: LLM calls use low temperature + fixed seed (already the default in `llm.generate`).

## File Structure

- `v1/src/llm.py` — **modify**: add `UNDERSTAND_SYSTEM` constant + `understand()` function (mirrors `rewrite_followup`).
- `v1/src/interface.py` — **modify**: add the uncertainty-fallback hook in `ask()`; surface the interpretation note. Reuses `_routable`, `INTENTS`, `logger`.
- `v1/tests/test_interface_understanding.py` — **create**: unit tests (LLM mocked).
- `v1/src/understanding_eval.py` — **create**: malformed-query corpus + LLM-on/off scorer.
- `v1/tests/test_understanding_eval.py` — **create**: scorer unit test (mocked interface).
- `v1/docs/evaluation/09-understanding-agent.md` — **create**: evaluation write-up (real numbers, run with Ollama).
- `code/WORKLOG.md` — **modify**: dated entry.

---

### Task 1: `llm.understand()` — the rewrite proposer

**Files:**
- Modify: `v1/src/llm.py` (add constant near `PLAN_SYSTEM`; add function near `rewrite_followup`)
- Test: `v1/tests/test_interface_understanding.py`

**Interfaces:**
- Produces: `understand(question: str, columns: list[str], intents: list[str], model: str = DEFAULT_MODEL) -> str | None` — returns a cleaned one-line rewritten question, or `None` (LLM unavailable / reply empty / reply is `NONE` / network error).

- [ ] **Step 1: Write the failing tests**

```python
# v1/tests/test_interface_understanding.py
"""Understanding/Clarifier agent: llm.understand proposes a routable rewrite of a malformed query
(fixes spelling/paraphrase); the interface accepts it only if the deterministic parser can route it,
and never lets the LLM produce a number. LLM is mocked; no Ollama needed."""

from src import llm


def test_understand_returns_cleaned_rewrite(monkeypatch):
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: True)
    monkeypatch.setattr(llm, "generate", lambda *a, **k: '"humidity in October 2012"\n')
    out = llm.understand("umidty tocber 2012", ["rh (%)"], ["mean", "max"])
    assert out == "humidity in October 2012"


def test_understand_none_reply_is_none(monkeypatch):
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: True)
    monkeypatch.setattr(llm, "generate", lambda *a, **k: "NONE")
    assert llm.understand("what is the meaning of life", ["rh (%)"], ["mean"]) is None


def test_understand_offline_is_none_without_calling_generate(monkeypatch):
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: False)
    def boom(*a, **k):
        raise AssertionError("generate must not be called when offline")
    monkeypatch.setattr(llm, "generate", boom)
    assert llm.understand("anything", ["rh (%)"], ["mean"]) is None


def test_understand_network_error_is_none(monkeypatch):
    import urllib.error
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: True)
    def boom(*a, **k):
        raise urllib.error.URLError("down")
    monkeypatch.setattr(llm, "generate", boom)
    assert llm.understand("anything", ["rh (%)"], ["mean"]) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3.12 -m pytest tests/test_interface_understanding.py -q`
Expected: FAIL (`AttributeError: module 'src.llm' has no attribute 'understand'`)

- [ ] **Step 3: Implement `UNDERSTAND_SYSTEM` + `understand()`**

Add near `PLAN_SYSTEM` in `v1/src/llm.py`:

```python
UNDERSTAND_SYSTEM = (
    "You rewrite a user's messy question about a weather time-series into ONE clear, standalone "
    "question. Fix spelling and word order. Use only the sensors and operations the user is given. "
    "You DO NOT answer the question and must NEVER output any data value, number, statistic, or "
    "measurement. If the message is not a question about this weather data, reply with exactly: NONE."
)
```

Add near `rewrite_followup` in `v1/src/llm.py`:

```python
def understand(question: str, columns: list[str], intents: list[str],
               model: str = DEFAULT_MODEL) -> str | None:
    """Rewrite a malformed/paraphrased question into one clear standalone question the deterministic
    parser can route. Fixes spelling and phrasing only — never emits a number; the caller must still
    verify the rewrite is routable before using it. Returns None if Ollama is unavailable, the reply
    is empty, or the model judges the message off-topic (``NONE``)."""
    if not is_available():
        return None
    prompt = (f"Available sensors: {columns}\nAvailable operations: {intents}\n\n"
              f"User's message: {question}\n\n"
              "Rewrite it as one clear standalone question, or reply NONE.")
    try:
        out = generate(prompt, model=model, system=UNDERSTAND_SYSTEM, timeout=30.0)
    except (urllib.error.URLError, OSError):
        return None
    out = (out or "").strip().strip('"').strip()
    if not out or out.strip().upper() == "NONE":
        return None
    return out.splitlines()[0].strip()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3.12 -m pytest tests/test_interface_understanding.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Lint/type then commit**

```bash
cd /home/saizer/Documents/Msc_project/code
ruff check v1/src/llm.py v1/tests/test_interface_understanding.py && python3.12 -m mypy v1/src/llm.py
cd v1 && git add src/llm.py tests/test_interface_understanding.py
git commit -m "feat(understanding): llm.understand rewrite proposer (numberless, offline-safe)"
```

---

### Task 2: Wire the uncertainty-fallback hook into `Interface.ask`

**Files:**
- Modify: `v1/src/interface.py` (in `ask()`, after the follow-up-resolution block, before `ans = self._answer(resolved)`; and where the spelling note is applied)
- Test: `v1/tests/test_interface_understanding.py` (append)

**Interfaces:**
- Consumes: `llm.understand(...)` (Task 1); `self._routable(question) -> bool`; `INTENTS` (already imported in `interface.py`).
- Produces: on an unroutable query with the LLM on, `ask()` answers via the accepted rewrite and prepends `"(I read this as: '<rewrite>'.) "` to `ans.text`, setting `ans.resolved` to the rewrite.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_interface_understanding.py`)

```python
import numpy as np
import pandas as pd
from src import interface


def _df():
    idx = pd.date_range("2015-01-01", "2016-01-01", freq="h")
    rng = np.random.default_rng(0)
    return pd.DataFrame({"T (degC)": rng.normal(10, 8, len(idx)),
                         "rh (%)": rng.normal(70, 12, len(idx))}, index=idx)


def test_unroutable_query_uses_llm_rewrite_and_answers(monkeypatch):
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: True)
    monkeypatch.setattr(llm, "understand", lambda *a, **k: "maximum temperature in 2015")
    iface = interface.Interface(_df(), use_llm=True)
    ans = iface.ask("zzzq the hot stuff blah")          # deterministically unroutable
    assert ans.intent == "max" and ans.value is not None
    assert "I read this as" in ans.text
    assert ans.resolved == "maximum temperature in 2015"


def test_llm_rewrite_that_is_not_routable_is_rejected(monkeypatch):
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: True)
    monkeypatch.setattr(llm, "understand", lambda *a, **k: "still complete nonsense")
    iface = interface.Interface(_df(), use_llm=True)
    ans = iface.ask("zzzq the hot stuff blah")
    assert ans.value is None                            # no guess; fell back to clarify/rephrase


def test_offline_matches_deterministic_and_does_not_call_understand(monkeypatch):
    called = {"n": 0}
    def spy(*a, **k):
        called["n"] += 1
        return "maximum temperature in 2015"
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: False)
    monkeypatch.setattr(llm, "understand", spy)
    iface = interface.Interface(_df(), use_llm=True)
    ans = iface.ask("zzzq the hot stuff blah")
    assert ans.value is None and called["n"] == 0


def test_already_routable_query_does_not_call_understand(monkeypatch):
    called = {"n": 0}
    def spy(*a, **k):
        called["n"] += 1
        return "maximum temperature in 2015"
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: True)
    monkeypatch.setattr(llm, "understand", spy)
    iface = interface.Interface(_df(), use_llm=True)
    ans = iface.ask("what was the max temperature in 2015")
    assert ans.intent == "max" and called["n"] == 0


def test_understand_exception_is_caught_and_falls_back(monkeypatch, caplog):
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: True)
    def boom(*a, **k):
        raise RuntimeError("ollama exploded")
    monkeypatch.setattr(llm, "understand", boom)
    iface = interface.Interface(_df(), use_llm=True)
    with caplog.at_level("WARNING"):
        ans = iface.ask("zzzq the hot stuff blah")
    assert ans.value is None
    assert any("understanding" in r.getMessage().lower() for r in caplog.records)
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3.12 -m pytest tests/test_interface_understanding.py -q`
Expected: FAIL (rewrite not applied; `ans.value is None` for the first test, no "I read this as")

- [ ] **Step 3: Implement the hook in `ask()`**

In `v1/src/interface.py`, immediately AFTER the follow-up-resolution `if (...):` block and BEFORE the `clarify_words = clarifications_needed(...)` guardrail line, insert:

```python
        # Understanding agent: if the deterministic parser still can't route this query, ask the LLM
        # to rewrite it into something routable (heavy typos, paraphrase, word-order). Accept ONLY a
        # rewrite the parser can route -- the LLM never gains routing authority and never emits a
        # number. Offline / on failure -> unchanged deterministic behaviour below.
        interpretation: str | None = None
        if self.use_llm and llm.is_available() and not self._routable(resolved):
            try:
                rw = llm.understand(resolved, list(self.df.columns), list(INTENTS.keys()),
                                    model=self.model)
            except Exception as exc:  # noqa: BLE001 - any LLM failure must degrade, not crash
                logger.warning("understanding rewrite failed (%s); using deterministic fallback.", exc)
                rw = None
            if rw and self._routable(rw) and rw.strip().lower() != resolved.strip().lower():
                interpretation = rw.strip()
                resolved = interpretation
```

Then, where the spelling note is applied after `ans = self._answer(resolved)` (just after the
`if spell_fixes:` block), add:

```python
        if interpretation is not None:
            ans.text = f"(I read this as: '{interpretation}'.) {ans.text}"
            ans.resolved = interpretation
```

- [ ] **Step 4: Run to verify they pass**

Run: `python3.12 -m pytest tests/test_interface_understanding.py -q`
Expected: PASS (all)

- [ ] **Step 5: Full suite + lint/type, then commit**

```bash
cd /home/saizer/Documents/Msc_project/code
python3.12 -m pytest v1/ -q && ruff check v1/src/interface.py v1/tests/test_interface_understanding.py && python3.12 -m mypy v1/src/interface.py
cd v1 && git add src/interface.py tests/test_interface_understanding.py
git commit -m "feat(understanding): route unroutable queries via LLM rewrite, gated by _routable"
```

---

### Task 3: Guardrail upgrade — try understanding before blocking on a gray-zone word

**Files:**
- Modify: `v1/src/interface.py` (the `clarify_words = clarifications_needed(...)` block in `ask()`)
- Test: `v1/tests/test_interface_understanding.py` (append)

**Interfaces:**
- Consumes: `llm.understand`, `self._routable`, existing `clarifications_needed`, `self._clarify`.
- Produces: when a gray-zone word would block AND the LLM yields a routable rewrite, `ask()` answers-with-interpretation instead of returning `_clarify`; otherwise it asks as today.

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_grayzone_word_recovered_by_understanding(monkeypatch):
    # A near-miss word that the deterministic guardrail would block; the LLM recovers it.
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: True)
    monkeypatch.setattr(llm, "understand", lambda *a, **k: "average humidity in 2015")
    iface = interface.Interface(_df(), use_llm=True)
    ans = iface.ask("avrage humdity in 2015")          # gray-zone spellings
    assert ans.intent == "mean" and ans.value is not None
    assert "I read this as" in ans.text


def test_grayzone_without_llm_still_asks(monkeypatch):
    monkeypatch.setattr(llm, "is_available", lambda *a, **k: False)
    iface = interface.Interface(_df(), use_llm=True)
    ans = iface.ask("avrage humdity in 2015")
    assert ans.value is None                            # asks to clarify, as today
```

- [ ] **Step 2: Run to verify the first fails**

Run: `python3.12 -m pytest tests/test_interface_understanding.py -k grayzone -q`
Expected: FAIL on `test_grayzone_word_recovered_by_understanding` (currently returns `_clarify`, value None)

- [ ] **Step 3: Implement the guardrail upgrade**

Replace the guardrail block in `ask()`:

```python
        clarify_words = clarifications_needed(resolved, tuple(self.df.columns))
        if clarify_words:
            clar = self._clarify(original, spell_fixes, clarify_words)
            self.history.append({"question": original, "resolved": resolved, "computed": ""})
            self.history = self.history[-6:]
            return clar
```

with:

```python
        clarify_words = clarifications_needed(resolved, tuple(self.df.columns))
        if clarify_words:
            # Before blocking on a near-miss word, let the LLM try to recover the intended query.
            recovered = None
            if self.use_llm and llm.is_available():
                try:
                    rw = llm.understand(resolved, list(self.df.columns), list(INTENTS.keys()),
                                        model=self.model)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("guardrail rewrite failed (%s); asking to clarify.", exc)
                    rw = None
                if rw and self._routable(rw) and rw.strip().lower() != resolved.strip().lower():
                    recovered = rw.strip()
            if recovered is None:
                clar = self._clarify(original, spell_fixes, clarify_words)
                self.history.append({"question": original, "resolved": resolved, "computed": ""})
                self.history = self.history[-6:]
                return clar
            interpretation = recovered
            resolved = interpretation
```

Note: `interpretation` is already declared just below in Task 2's block; move its declaration
(`interpretation: str | None = None`) to just ABOVE this guardrail block so both paths can set it,
and delete the duplicate declaration inside the Task 2 hook.

- [ ] **Step 4: Run to verify they pass**

Run: `python3.12 -m pytest tests/test_interface_understanding.py -q`
Expected: PASS (all)

- [ ] **Step 5: Full suite + lint/type, then commit**

```bash
cd /home/saizer/Documents/Msc_project/code
python3.12 -m pytest v1/ -q && ruff check v1/src/interface.py && python3.12 -m mypy v1/src/interface.py
cd v1 && git add src/interface.py tests/test_interface_understanding.py
git commit -m "feat(understanding): recover gray-zone guardrail words via LLM before asking"
```

---

### Task 4: Malformed-query evaluation corpus + LLM-on/off scorer

**Files:**
- Create: `v1/src/understanding_eval.py`
- Test: `v1/tests/test_understanding_eval.py`

**Interfaces:**
- Produces: `MALFORMED_CASES: list[Case]` (reuses `query_coverage.Case`) and
  `score(iface, cases=MALFORMED_CASES) -> dict` returning
  `{"n", "answered", "correct", "declined"}` where a case is *correct* when the routed intent matches
  and the expected sensors are a subset of the routed sensors.

- [ ] **Step 1: Write the failing test**

```python
# v1/tests/test_understanding_eval.py
"""The malformed-query scorer counts how many messy queries an interface answers correctly. Uses a
stub interface so it runs without Ollama."""

from src import understanding_eval as ue
from src.query_coverage import Case


class _StubAnswer:
    def __init__(self, intent, sensors, value):
        self.intent, self.sensors, self.value = intent, sensors, value


class _StubIface:
    def __init__(self, mapping):
        self.mapping = mapping
        self.history = []
    def ask(self, q):
        intent, sensors, value = self.mapping.get(q, (None, [], None))
        return _StubAnswer(intent, sensors, value)


def test_score_counts_correct_and_declined():
    cases = [Case("q1", "max", frozenset({"T (degC)"}), True),
             Case("q2", "mean", frozenset({"rh (%)"}), False)]
    iface = _StubIface({"q1": ("max", ["T (degC)"], {"value": 1.0}),
                        "q2": (None, [], None)})
    out = ue.score(iface, cases)
    assert out["n"] == 2 and out["correct"] == 1 and out["declined"] == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3.12 -m pytest tests/test_understanding_eval.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.understanding_eval'`)

- [ ] **Step 3: Implement the corpus + scorer**

```python
# v1/src/understanding_eval.py
"""Evaluation for the Understanding agent: a corpus of MALFORMED / paraphrased queries and a scorer
that reports how many an interface answers correctly. Run it with the LLM on vs off to quantify the
capability the Understanding agent adds. Ground truth is the obviously-correct human reading, written
before running any code (same principle as query_coverage.GROUND_TRUTH)."""
from __future__ import annotations

from .query_coverage import Case

MALFORMED_CASES: list[Case] = [
    Case("higist tempreture 2015", "max", frozenset({"T (degC)"}), True),
    Case("umidty tocber 2012", "summary", frozenset({"rh (%)"}), True),
    Case("how muggy did it get in summer 2013", "summary", frozenset({"rh (%)"}), True),
    Case("avrage windspeed 2016", "mean", frozenset({"wv (m/s)"}), True),
    Case("coldist it evar got", "min", frozenset({"T (degC)"}), False),
    Case("presure in jan 2014 lowset", "min", frozenset({"p (mbar)"}), True),
    Case("how breezy in june", "summary", frozenset({"wv (m/s)"}), True),
    Case("dew pont standard deviaton 2014", "spread", frozenset({"Tdew (degC)"}), True),
]


def score(iface, cases: list[Case] = MALFORMED_CASES) -> dict:
    n = len(cases)
    answered = correct = declined = 0
    for case in cases:
        iface.history = []
        ans = iface.ask(case.question)
        if ans.value is None:
            declined += 1
            continue
        answered += 1
        if ans.intent == case.intent and case.sensors.issubset(set(ans.sensors)):
            correct += 1
    return {"n": n, "answered": answered, "correct": correct, "declined": declined}
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3.12 -m pytest tests/test_understanding_eval.py -q`
Expected: PASS

- [ ] **Step 5: Lint/type then commit**

```bash
cd /home/saizer/Documents/Msc_project/code
ruff check v1/src/understanding_eval.py v1/tests/test_understanding_eval.py && python3.12 -m mypy v1/src/understanding_eval.py
cd v1 && git add src/understanding_eval.py tests/test_understanding_eval.py
git commit -m "feat(understanding): malformed-query eval corpus + LLM-on/off scorer"
```

---

### Task 5: Live evaluation run + write-up + WORKLOG

**Files:**
- Create: `v1/docs/evaluation/09-understanding-agent.md`
- Modify: `code/WORKLOG.md`

**Interfaces:**
- Consumes: `understanding_eval.score`, `interface.Interface`, `preprocess`.

- [ ] **Step 1: Run the live comparison (needs Ollama running)**

```bash
cd /home/saizer/Documents/Msc_project/code/v1
python3.12 - <<'PY'
from src import preprocess, interface, understanding_eval as ue
raw = preprocess.clean(preprocess.load_data("data/jena_climate_2009_2016.csv"))
df = preprocess.align(raw, freq="1h")
off = ue.score(interface.Interface(df, raw=raw, use_llm=False))
on  = ue.score(interface.Interface(df, raw=raw, use_llm=True))
print("LLM off:", off)
print("LLM on :", on)
PY
```
Record the two dicts. Expected shape: `on["correct"] > off["correct"]` (the capability gain); both
should keep `declined` for anything genuinely ambiguous (no wrong guesses).

- [ ] **Step 2: Write `v1/docs/evaluation/09-understanding-agent.md`**

Follow the numbered-eval-doc pattern (see `08-external-llm-benchmark.md`): *What's being evaluated*
(understanding, not computation) · *Method* (MALFORMED_CASES, LLM-on vs off, correct = intent match
+ expected sensors ⊆ routed sensors) · *Real evidence* (paste the two dicts from Step 1 as a table) ·
*Honest reading* (the LLM widens understanding; numbers stay deterministic; anything still ambiguous
is declined, not guessed) · *Reproduce* (the snippet above).

- [ ] **Step 3: Update `code/WORKLOG.md`**

Add a dated `## 2026-07-30` sub-entry: the Understanding agent (LLM rewrite-and-reroute, `_routable`
-gated, offline-safe), the trust rationale, the on/off numbers, and the honest note that it improves
*coverage of correctly-answered queries*, not numeric accuracy.

- [ ] **Step 4: Commit**

```bash
cd /home/saizer/Documents/Msc_project/code
git add v1/docs/evaluation/09-understanding-agent.md WORKLOG.md
git commit -m "docs(understanding): evaluation write-up + worklog for the Understanding agent"
```

---

## Self-Review

**Spec coverage:** rewrite proposer (Task 1) · uncertainty-fallback hook + transparency + offline guard (Task 2) · guardrail upgrade (Task 3) · testing (Tasks 1–4) · evaluation corpus + write-up (Tasks 4–5). All spec sections map to a task.

**Placeholder scan:** every code step contains real code; no TBD/TODO; commit messages carry no Claude trailer (per Global Constraints).

**Type consistency:** `understand(question, columns, intents, model)` defined in Task 1 is called with the same argument order/types in Tasks 2 and 3; `_routable(str) -> bool` and `INTENTS` are existing symbols; `score(iface, cases) -> dict` keys (`n/answered/correct/declined`) are consistent between Task 4's implementation, its test, and Task 5's run.
