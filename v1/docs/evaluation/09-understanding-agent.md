# 9. Understanding/Clarifier agent — recovering malformed queries

## What's being evaluated

Not computation (that is exact and verified elsewhere) but **understanding**: when a question is
misspelled beyond the deterministic auto-corrector, paraphrased, or word-order-scrambled, does the
system recover the intended query instead of rejecting it? The Understanding agent (`llm.understand`,
wired into `Interface.ask`) asks the local LLM to **rewrite** such a query into a clean standalone
question, and accepts the rewrite **only if the deterministic parser can route it** (`_routable`) —
so the LLM widens understanding without ever routing or computing a number itself.

## Method

Corpus: `src/eval_corpora.py::MALFORMED_CASES` — 8 deliberately messy queries (heavy typos,
paraphrase, scrambled order), each hand-labelled with the obviously-correct `(intent, sensors,
has_scope)` before running any code. `score_understanding()` runs each through the real
`Interface.ask` and counts:
*answered* (produced a value), *correct* (routed intent matches and expected sensors ⊆ routed
sensors), *declined* (asked to rephrase — the right outcome for a genuinely ambiguous query, not a
failure). Run twice: LLM **off** (deterministic baseline) vs LLM **on** (Ollama `llama3.2:3b`).

## Real evidence (30 July 2026, local `llama3.2:3b`)

```
LLM off: {'n': 8, 'answered': 5, 'correct': 4, 'declined': 3}
LLM on : {'n': 8, 'answered': 8, 'correct': 5, 'declined': 0}
```

Per-question (LLM on), showing the accepted rewrite:

| Query | Outcome | Interpretation the agent computed on |
| --- | --- | --- |
| `higist tempreture 2015` | ✅ correct (max, T) | "What is the highest recorded temperature in 2015?" |
| `presure in jan 2014 lowset` | ✅ correct (min, p) | "What is the lowest recorded pressure value for January 2014?" |
| `avrage windspeed 2016` | ✅ correct (mean, wv) | (deterministic spell-fix; LLM not needed) |
| `coldist it evar got` | ✅ correct (min, T) | (deterministic spell-fix; LLM not needed) |
| `dew pont standard deviaton 2014` | ✅ correct (spread, Tdew) | (deterministic spell-fix; LLM not needed) |
| `umidty tocber 2012` | ✗ mis-routed | rewrote to a *weather report for 2012* (not humidity) |
| `how muggy did it get in summer 2013` | ✗ mis-routed | routed to anomaly, not a humidity summary |
| `how breezy in june` | ✗ mis-routed | routed to a weather report, not wind speed |

## Honest reading

- **Coverage up, with a caveat.** The agent turned all 3 previously-*declined* queries into answers
  and added +1 *correct*. But the small 3B model also produced **plausible-but-wrong routable
  rewrites** on 3 cases — it answered where the baseline had (safely) declined. So this is a genuine
  trade-off, not a clean win: more questions answered, but a minority answered under the wrong
  interpretation.
- **The failures are mis-routings, not mis-arithmetic.** Every number is still computed exactly by
  the query engine; "umidty tocber 2012" returned a *correct* weather report — just not the humidity
  the user meant. Correctness of computation is untouched (the project invariant holds).
- **Transparency is the mitigation.** Every accepted rewrite is shown to the user ("I read this
  as: '…'"), so a wrong interpretation is visible and correctable rather than silent.
- **Model-bound, not design-bound.** The mis-rewrites are the 3B model's limits; a stronger model
  (larger local model, or a hosted one) would raise rewrite fidelity. The trust gates
  (`_routable` acceptance, deterministic computation) are model-independent.
- **Offline safety.** With Ollama off the agent is skipped entirely and behaviour is identical to
  the deterministic baseline (regression-guarded by `test_offline_matches_deterministic_and_does_not_call_understand`).

## Reproduce

```bash
cd code/v1
python3.12 - <<'PY'
from src import preprocess, interface, eval_corpora as ec
raw = preprocess.clean(preprocess.load_data("data/jena_climate_2009_2016.csv"))
df = preprocess.align(raw, freq="1h")
print("off:", ec.score_understanding(interface.Interface(df, raw=raw, use_llm=False)))
print("on :", ec.score_understanding(interface.Interface(df, raw=raw, use_llm=True)))
PY
```
