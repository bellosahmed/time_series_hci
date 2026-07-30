# Understanding/Clarifier Agent — Design Spec

*Date: 2026-07-30 · Component of the "multi-agent + COSMA" work (sub-project 1 of 3). Sub-projects
2 (multi-agent orchestrator) and 3 (COSMA batch harness) are separate specs.*

## Problem

The v1 interface understands questions with a **deterministic** parser: `spell_normalize` fixes
high-confidence typos (`difflib`, ratio ≥ 0.85), `clarifications_needed` blocks-and-asks on
gray-zone words (0.75–0.85), and `_semantic_fallback` asks the user to rephrase anything it can't
route. This is exact and fully offline, but **literal**: it cannot recover a query that is heavily
misspelled beyond the difflib threshold, paraphrased ("how muggy was it" → humidity), or
word-order-mangled ("umidty tocber 2012"). Those queries are currently rejected even though a human
reads them instantly.

## Goal

When — and only when — the deterministic path cannot confidently route a query, use the local LLM to
**rewrite the raw query into a clean standalone question**, accept the rewrite **only if the
deterministic parser can route it**, compute the answer with the existing exact engine, and surface
the interpretation to the user. This widens *understanding* without giving the LLM any authority over
*routing* or *computation*.

### Non-goals
- No structured `{intent, sensors, scope}` JSON routing (that is approach "B" — deliberately
  rejected here to keep the parser the sole router).
- No numeric computation by the LLM (unchanged project invariant).
- No conversation-memory changes, no new dependencies, no change to the offline behaviour.
- Not aimed at improving correctness of numbers — those are already exact and verified
  (`verify_benchmark.py`). This improves *coverage of correctly-answered queries*, not accuracy.

## Design

### Data flow (in `Interface.ask`)
Order is unchanged up to routing; the LLM engages as a **fallback on uncertainty only**:

1. Deterministic `spell_normalize` + follow-up resolution (as today).
2. Deterministic guardrail `clarifications_needed` (as today) — see the guardrail note below.
3. Attempt deterministic routing.
   - **Routable** → answer exactly as today. **The LLM is never called** on the common path
     (no latency, no risk).
   - **Not routable** (would reach `_semantic_fallback`) **and** `use_llm and llm.is_available()`
     → call the Understanding agent (`llm.understand`), obtain a rewrite `rw`.
     - If `rw` is non-empty, differs from the input, and **`self._routable(rw)` is True** →
       route `rw`, answer, and set the interpretation note ("I read this as: '…'").
     - Otherwise → existing `_semantic_fallback` / `_clarify` ("could you rephrase?"). **No guess.**
4. **Guardrail upgrade (optional, same hook):** when `clarifications_needed` would block on a
   gray-zone word and the LLM is available, first try `llm.understand`; if it yields a confidently
   routable rewrite, answer-with-interpretation-note instead of blocking; else ask as today.

### Component: `llm.understand`
```
understand(question: str, columns: list[str], intents: list[str],
           model: str = DEFAULT_MODEL) -> str | None
```
Mirrors the existing `llm.rewrite_followup` in signature, error handling, determinism (low
temperature + fixed seed), and return contract (a plain rewritten question string, or `None`).
System prompt (vocab-constrained): *"Rewrite the user's weather-data question into one clear
standalone question, using only these sensors {columns} and these operations {intents} and ordinary
date/time expressions. Fix spelling and word order. Do NOT answer the question. If it is not a
question about this weather data, reply exactly NONE."* Returns `None` for `NONE`/empty/error.

### Trust gates (the core of the design)
- **Router stays deterministic.** A rewrite is accepted **only if `self._routable(rw)`** — the LLM
  can only steer toward interpretations the deterministic parser already handles; it cannot invent a
  sensor, intent, or scope outside the vocabulary.
- **No numeric-hallucination path.** Every number is computed by the query engine over the accepted
  rewrite; the LLM output is a *question*, never a value. The existing faithfulness gate on any
  downstream synthesis still applies.
- **Transparency.** The accepted rewrite is always shown (reuse `ans.resolved` + a visible
  "I read this as: '<rw>'" note, analogous to the existing spelling note) so the user can catch a
  misread.
- **Ask-when-unsure preserved.** If no routable rewrite is produced, the system asks to rephrase; it
  never answers on a guessed interpretation.

### Error handling / offline
- `llm.is_available()` is False → the Understanding step is skipped entirely → **exactly today's
  deterministic behaviour, zero regression** (core to the local-first design).
- Any exception in the LLM call → log a warning and fall back to `_semantic_fallback` / `_clarify`
  (the same defensive pattern the follow-up-rewrite path already uses).

## Testing

Unit tests (`tests/test_interface_understanding.py`), LLM mocked as in
`tests/test_interface_llm_gate.py`:
1. Malformed query the parser rejects + mocked routable rewrite → answered, number correct,
   interpretation surfaced in `ans.text` / `ans.resolved`.
2. Mocked rewrite that is **not** routable → falls back to clarify/rephrase, `ans.value is None`
   (no guess).
3. LLM **off** (`is_available` False) → behaviour identical to the deterministic baseline
   (regression guard).
4. Mocked `understand` returns `None` (off-topic) → `_semantic_fallback`.
5. An **already-routable** query → assert `llm.understand` is **not called** (common path untouched).
6. LLM raises → caught, deterministic fallback, failure logged.

## Evaluation (eval-depth goal)

A small labelled corpus of **malformed / paraphrased** queries (typo-heavy, paraphrase,
word-order), each with expected `(intent, sensors, has_scope)`, scored with the existing
`query_coverage` machinery. Report **LLM-on vs LLM-off** "correctly answered" counts — the citable
capability gain — plus how many previously-rejected queries are now answered, and how many the
system still (correctly) declines to guess. Write-up in `docs/evaluation/` following the numbered
pattern.

## Files touched
- `v1/src/llm.py` — add `understand()` + system-prompt constant.
- `v1/src/interface.py` — add the uncertainty-fallback hook in `ask()`; surface the interpretation;
  reuse `_routable`, `_semantic_fallback`, `_clarify`.
- `v1/tests/test_interface_understanding.py` — new.
- `v1/src/query_coverage.py` (or a sibling) — malformed-query corpus + LLM-on/off scoring helper.
- `v1/docs/evaluation/09-understanding-agent.md` — evaluation write-up.

## Risks & mitigations
- *LLM proposes a plausible-but-wrong reading that happens to be routable.* Mitigated by always
  surfacing the interpretation (user sees "I read this as…") and by only engaging on queries the
  deterministic path already failed — the alternative was rejecting the query outright.
- *Latency on hard queries.* Only the already-failing minority pay the LLM cost; the common path is
  untouched.
- *Non-determinism across runs.* Low temperature + fixed seed, as elsewhere in `llm.py`.

## Future extensions (out of scope here)
- Approach B (structured interpretation) as a second-tier fallback if evaluation shows A leaves a
  real gap.
- Feed the Understanding agent's output into sub-project 2 (multi-agent orchestrator) as the
  shared "understanding" front-end.
