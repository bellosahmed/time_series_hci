# Chat interface — conversational memory: fix plan (to implement later)

Drafted **2026-07-05**. A plan to fix the chat interface's conversational memory. **Not yet
implemented** — this is the design record to pick up later. No code changed when this was written.

---

## Diagnosis — how "memory" works today

- **CLI (`apps/chat.py`):** one `Interface` in a loop, so `self.history` persists in-process. ✅ works.
- **Web (`apps/web/app.py` + `apps/web/index.html`):** the server is **stateless** — the browser keeps
  `convo = [{question, computed}]`, resends it as `history` on each POST, and the server seeds
  `iface.history` from it (`app.py:91`). Wired end-to-end.
- **The catch:** follow-up resolution runs **only** through `llm.rewrite_followup`, which is gated on
  `self.use_llm and llm.is_available()` (in `Interface.ask`).

## The real issues (ranked)

1. **🔴 Memory only works with the LLM on.** With Ollama off (the default offline mode), *"what about
   2014?"* is **not** resolved — `_understand` sees only the current words, gets `intent=None`, and
   returns the "couldn't map that" fallback. The chat silently forgets context whenever the LLM isn't
   running. **This is the main bug.**
2. **🟡 Chained follow-ups degrade.** The web history stores the user's **raw wording**, not the
   *resolved* standalone question, so a follow-up-of-a-follow-up ("…and 2015?") loses the thread.
3. **🟡 Client history grows unbounded.** `convo` is never trimmed (`index.html:169`); the payload grows
   every turn (the server only uses the last 6).
4. **🟡 Per-request `Interface` recreation** rebuilds the retriever and drops the `_daytypes` K-means
   cache, so every day-type follow-up recomputes clustering.
5. **⚪ Empty-`computed` turns** (legend/error answers) pollute the context window.

## Plan to fix

### Phase 1 — deterministic follow-up memory (the big win, LLM-free)
Add `resolve_followup(question, history)` in `interface_parsing.py` that carries forward the **last
turn's intent + sensor + period** when the new message is elliptical (no intent/sensor/period of its
own but names a new scope, e.g. a bare year). Wire it into `Interface.ask` *before* the LLM path so
memory works with Ollama off; keep `llm.rewrite_followup` as the richer fallback when available.
- *Example:* history "hottest in 2013" + "what about 2014?" → carry intent=`max`, sensor=`T`, swap
  period → 2014.
- **Tests:** multi-turn follow-up with `use_llm=False`; chained follow-ups (new `test_interface` cases).

### Phase 2 — fix chaining + trim (web)
In `index.html`: push the **resolved** question when present
(`convo.push({question: data.resolved || data.question, computed})`) and **cap `convo` to the last 6**.
The server already returns `resolved`. This makes chained follow-ups resolve against normalised
questions.

### Phase 3 — stop recomputing per request (server memory/perf)
Build the derived state **once** at app level — cache the `Retriever` and the `_daytypes` clustering on
the shared `df` in `create_app` — and have each request's `Interface` reuse them (inject via the
constructor or a module-level cache). Keeps the stateless, multi-user design but removes the
per-request rebuild.

### Phase 4 — hygiene
Skip pushing turns with empty `computed`; document the memory model (client-side stateless, 6-turn
window) in the web app docstring and README.

## Verification
- New tests: `test_interface` (deterministic follow-up without LLM, chained follow-ups), `test_apps`
  (server seeds + caps history). Keep the full suite green + `verify_results.py` 29/29.

## Design decision
**Keep the stateless client-side history** (the phases above) rather than moving to server-side Flask
sessions — it stays multi-user / multi-tab safe with no session-cleanup burden, and the only real gap
(LLM-off memory) is closed by Phase 1. Server-side sessions would simplify the client but add shared
state and lifecycle management for little gain at this scope.

## Priority when picked up
**Phase 1 first** — it's the one that actually fixes "it forgets" (works without the LLM). Phases 2–4
are refinements. Estimated effort: Phase 1 ≈ half a day incl. tests; Phases 2–4 small.

## Open question to confirm before implementing
Confirm "memory issue" means **conversational context** (this plan). If it turns out to mean **RAM/GPU
growth** or a specific observed behaviour, re-scope — the web app loads `df` once and recreates a
lightweight `Interface` per request, so no obvious leak was found, but that path wasn't the focus here.
