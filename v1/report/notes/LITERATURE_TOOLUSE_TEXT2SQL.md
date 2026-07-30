# Literature positioning: tool-use and text-to-SQL

> **Status: write-up material. No code required.** Nothing in this document asks for an
> implementation. `agent.py` already *is* the tool-use pattern; text-to-SQL is deliberately
> **not** built — it is cited as the competing paradigm in order to sharpen this project's
> own contribution claim. Companion docs: `report/REPORT.md` (§2 literature review, §6
> discussion), `docs/evaluation/` (the evidence these arguments rest on).
>
> **Written 17 July 2026** in response to an assessment which found the literature review to be the
> weakest written component precisely because the project's *contribution area* — verifiable
> natural-language analytics — carries only a single citation (Ji et al. 2023), while its *methods*
> area is richly and correctly cited.
>
> **On marking criteria**: deliberately not quoted anywhere in this document. The assessment behind
> it used a published UK MSc rubric that was *not Loughborough's*, and criteria change year to
> year — a quoted weighting would go stale and mislead. Pull the real criteria off Learn when you
> need them. Everything below is argued on its own merits and holds regardless.

---

## 1. Why this document exists

The assessment finding, stated plainly:

The literature review is deep and correct everywhere the project does something standard
(Box & Jenkins for ARIMA, Cleveland for STL, Liu for Isolation Forest, Sparck Jones for TF-IDF —
all primary sources, all appropriate, all should stay). It is one-citation-thin at exactly the
point where §1.6 claims a contribution. An examiner reading "Verifiable NL Analytics" as a
contribution, then checking the references to see who else has attacked this problem, finds one
hallucination survey.

This creates a specific, avoidable problem. A literature review's academic function is to
**establish the gap** — to prove the thing you built needed building. §2's "Gap" paragraph
asserts that few works combine exact querying + NL explanation + verification. With one citation
in that space, an examiner cannot distinguish between two possibilities:

1. The gap is real, and you demonstrated it.
2. You did not search hard enough to find the work that fills it.

Those are indistinguishable from the outside, and the burden of proof sits with the author.

Two bodies of work sit directly on top of what this project built. Engaging with them
**strengthens** the contribution claim rather than weakening it — which is the counter-intuitive
part worth internalising. Both are cheap: prose and citations, no code.

---

## 2. Part one — tool-use / program-aided language models

### 2.1 What the pattern is

The core idea: **do not let the language model do the arithmetic**. Instead, have it decide
*which* deterministic operation to invoke and with *what* arguments, execute that operation in
real code, and feed the exact result back. The model contributes language and control flow; a
non-stochastic runtime contributes every number.

This is a recognised, heavily-published pattern with several named variants:

- **Program-aided / program-of-thought**: the model emits code (typically Python) which is
  executed; the model never evaluates the expression itself.
- **Tool-augmented / function-calling**: the model selects from a catalogue of typed tools and
  emits a structured call; a runtime validates and executes it.
- **Reason-and-act loops**: the model interleaves reasoning steps with tool invocations,
  observing each result before deciding the next action.

All three rest on the same premise this project independently arrived at: **LLM arithmetic is
unreliable, so remove arithmetic from the LLM's job description.**

### 2.2 What this project already built

`src/agent.py` is a genuine implementation of this pattern, and a more precise fit to the
literature than a casual reading suggests. The mapping is close to exact:

| Literature concept | `agent.py` implementation |
| --- | --- |
| Tool catalogue / function schema | `TOOLS` dict — five tools, each with `description`, `params`, `run`, `needs` |
| Tool schema passed to the planner | `Agent.tool_specs()` → `llm.plan_steps(question, specs, columns)` |
| Model emits a structured call, not an answer | `Agent.plan()` returns an ordered `list[dict]` of `{tool, args}` |
| Deterministic execution | `execute_step()` → `spec["run"](df, **args)` → delegates to `query_engine` / `analysis` |
| Argument validation before execution | `execute_step()` rejects unknown tools, unknown sensors, bad `kind`/`op`, unparseable `value` |
| Graceful degradation | failed step → `Fact(ok=False, error=…)` rather than raising; one bad step never sinks the plan |
| Grounded synthesis | `Agent.compose()` → `llm.explain_synthesis()`, accepted **only** if `llm.faithfulness()` finds no invented number |
| Fallback | no usable plan → single-step `interface.Interface`; LLM off → deterministic `_template()` |

The five tools (`extremum`, `average`, `correlation`, `threshold`, `anomaly_count`) each return a
`(computed_str, value)` pair where the number came from `query_engine` — which is itself verified
`== pandas` by `verify_results.py`. The chain from user question to number never passes through
the model.

**The code is complete, tested (`tests/test_agent.py`), and works. What is missing is telling the
examiner that the thing you built has a name and a literature.** Currently it reads as a personal
hunch. Cited, it reads as the correct application of an established principle — which is a
materially different impression.

### 2.3 The argument to make in the write-up

Two moves, both honest:

**Move one — claim the principle, cite its origin.** The design decision "computation = code,
LLM = explanation only" is not idiosyncratic; it is the same insight that motivates program-aided
LMs and tool-augmented LMs. Citing that literature converts the project's founding principle from
an assertion into a position with support behind it.

**Move two — name what this project adds on top.** The tool-use literature removes arithmetic
from the model. It does **not**, in general, verify that the model's *prose* about the results is
faithful to them. That is what `llm.faithfulness()` does, and it is genuinely a layer the standard
pattern lacks:

> Tool-use guarantees the *number* is computed correctly. It does not guarantee the *sentence
> wrapped around that number* reports it correctly. This project adds an automated faithfulness
> gate at the synthesis step (`Agent.compose`, `Interface._phrase`) which extracts every figure
> from the model's prose, checks each against the computed facts, and discards the phrasing
> entirely — falling back to a deterministic template — if any figure traces to nothing.

This is a defensible, specific, *small* contribution claim, and it is supported by real evidence
already collected: the adversarial probe in `docs/evaluation/05-llm-faithfulness.md`, where the
model fabricated "3.11 degrees higher than 2012" from facts containing neither 3.11 nor 2012, and
the production gate caught and discarded it.

**Do not overclaim.** The ablation in the same document found a 0% raw invention rate on
ordinary routed questions — meaning the gate has never been *observed* rescuing a real user
question, only an adversarially constructed one. That honesty is already in the evaluation docs
and must survive into the dissertation. The claim is "the gate demonstrably works when tempted,"
not "the gate is constantly saving users."

---

## 3. Part two — text-to-SQL and semantic parsing

### 3.1 What it is, and why it matters here

Text-to-SQL (a subfield of semantic parsing) is the dominant paradigm for natural-language
querying of structured data: the user asks a question, a model generates a database query, the
database executes it, and the answer is exact *by construction* because a database computed it.

**This project's architecture is a semantic parser.** Natural language in → structured
deterministic query → exact answer out. That is the definition. The interface's
`detect_intent` / `resolve_sensors` / `resolve_period` chain is a hand-built rule-based semantic
parser; `query_engine` is its execution backend.

Right now `report/REPORT.md` does not acknowledge that this field exists. The risk is concrete
and it is not about missing a citation for politeness — an examiner who knows the field will ask:
**does the author realise they have reinvented a well-studied paradigm?** The honest answer is
that the *verification layer* genuinely is distinct, but that argument can only be made after
demonstrating you know what you are distinguishing yourself from.

### 3.2 The failure mode that makes this project defensible

Here is the crux, and the single most valuable paragraph in this document.

Text-to-SQL's correctness guarantee is narrower than it first appears. The database will execute
the query correctly — that part is genuinely exact. But **nothing in the pipeline verifies that
the generated query answers the question the user actually asked.** The model can emit SQL that
is syntactically valid, executes without error, returns a plausible number, and answers the wrong
question. Wrong join, wrong aggregate, wrong filter, wrong date range — all silent. The user
receives a confident, exact, wrong answer, with no signal that anything went wrong.

This is a well-documented problem: it is precisely why the field measures *execution accuracy*
against gold queries rather than simply asserting correctness, and why benchmark scores on
realistic schemas sit well below 100%.

**Contrast with what this project does**, and note that the contrast is backed by real code in
`agent.py` rather than by argument alone:

| | Text-to-SQL | This project |
| --- | --- | --- |
| Query space | Unconstrained — any SQL the model emits | Fixed catalogue of five pre-verified tools (`TOOLS`) |
| Argument validation | None; malformed intent still executes | `execute_step()` validates tool name, sensor existence, `kind`/`op` domain, `value` type — rejects before execution |
| Failure mode | Silent wrong answer | Explicit rejection (`Fact(ok=False, error=…)`) or honest decline |
| Numerical correctness | Guaranteed by the database | Guaranteed by `query_engine`, verified `== pandas` |
| Prose faithfulness | Not addressed | `llm.faithfulness()` gate; invented figure → template fallback |
| Expressiveness | Arbitrary questions | Defined intent set only |

The design decision to state explicitly:

> Text-to-SQL trades verifiability for expressiveness: it will attempt any question, and can be
> silently wrong. This project takes the opposite trade — the space of expressible queries is
> constrained to exactly the space of pre-verified operations, so a question the system cannot
> answer correctly is one it declines rather than one it answers wrongly. Expressiveness is
> deliberately sacrificed for the guarantee that a returned number is both correct and correctly
> described.

### 3.3 Why this reframes an existing weakness into a design decision

`report/REPORT.md` §7.1 currently contains this admission:

> *"Interface routing is keyword-based with an optional LLM planner… it still handles a defined
> set of intents, not fully arbitrary language."*
>
> *"The question vocabulary is likewise a defined set… not open-ended analytics."*

As written, that reads as an apology — a limitation the author regrets. Positioned against
text-to-SQL, **the same fact becomes the point**: the constrained vocabulary is not a
shortcoming of the implementation, it is the mechanism that delivers the guarantee. A system that
accepted arbitrary language would forfeit exactly the property the dissertation is claiming.

This is the highest-leverage single change available to the write-up. It costs one paragraph and
converts a confessed weakness into a defended architectural choice.

### 3.4 The honest caveat that must accompany this — do not skip

**Do not compare this project's routing coverage against published text-to-SQL benchmark scores.**
It would be a category error, an examiner would spot it immediately, and it would undermine the
credibility the rest of the evaluation has earned.

The reasons, all of which must be stated openly:

- The 100% figure in `docs/evaluation/06-query-routing-coverage.md` is measured on **72
  hand-written questions over a fixed 14-sensor schema with ~13 intents**. Text-to-SQL benchmarks
  measure thousands of questions across hundreds of unseen database schemas requiring joins,
  nesting, and aggregation the model has never encountered.
- **These are not the same problem, and this project's is substantially easier.** 100% on a
  constrained, known vocabulary is not a superior result to ~80% on arbitrary schemas — it is a
  different measurement entirely.
- The honest framing is a **trade-off, not a victory**: this project answers fewer question types
  and guarantees them; text-to-SQL answers more question types without that guarantee. Both are
  legitimate engineering positions. Claiming to have beaten text-to-SQL would be false.
- `06-query-routing-coverage.md` already models the right tone in its "Honest caveats" section
  ("It is not a claim that *any* plausible English question is handled"). Carry that discipline
  into the comparison.

---

## 4. Where this goes in the report

| Section | What to add | Approx. length |
| --- | --- | --- |
| §2 Literature Review | New subsection: *Natural-language interfaces to structured data* — semantic parsing → text-to-SQL → the valid-but-wrong-query failure mode. Establishes the paradigm this project sits in and departs from. | ~2 paragraphs |
| §2 Literature Review | New subsection: *Grounding LLM computation* — program-aided LMs, tool-augmented LMs, reason-act loops. Establishes "don't let the model compute" as an established principle, not a hunch. | ~1–2 paragraphs |
| §2 "Gap" paragraph | Rewrite. Currently asserts few works combine the pieces. Should now say: text-to-SQL solves exact execution but not intent verification; tool-use solves computation grounding but not prose faithfulness; **neither verifies the explanation**. That is the gap, and it is now specific rather than assumed. | rewrite existing |
| §4 Implementation | One sentence in §4.2 naming `agent.py` as an implementation of the tool-use pattern, with citation. | 1 sentence |
| §6 Discussion | The trade-off argument from §3.2 above — expressiveness vs verifiability — plus the §3.4 honesty caveat. | ~2 paragraphs |
| §7.1 Limitations | Reframe the "defined set of intents" admission as a design decision, cross-referencing §6. Keep the limitation honest; change its framing from apology to rationale. | rewrite existing |

---

## 5. Citations to obtain

> ### ⚠️ VERIFY EVERY ONE OF THESE BEFORE CITING
>
> The references below are recalled from training, not looked up. Author lists, years, venues and
> exact titles **may be wrong**, and some may misattribute. Do not paste any of these into a
> dissertation on my say-so. Look each one up on Google Scholar / ACL Anthology / arXiv, confirm
> the details, read at least the abstract, and cite only what you have actually seen and can
> defend in a viva. A fabricated or misattributed citation in a dissertation is an academic
> integrity issue, and it is *your* name on the submission.
>
> Treat this as a **search list** — the right terms to look for — not a bibliography.

### 5.0 Prioritised shortlist — if you add only a few, add these

The full lists below are candidates, not a to-cite-all. The contribution area is *one-citation-thin*,
and the fix is **~4 papers**, not twelve — an examiner reads padding as padding. Ranked by how
directly each maps to this project's own code or its central contrast:

**Essential (3) — each directly describes the code or the paradigm being contrasted:**
- **Gao et al. (2023) — PAL.** "The model writes the plan, the interpreter computes" = `agent.py`.
- **Chen et al. (2022) — Program of Thoughts.** Disentangling computation from reasoning for
  *numeric* tasks = this project's founding principle, stated by someone else.
- **One** text-to-SQL anchor — **Yu et al. (2018) Spider** (canonical, unseen-schema generalisation)
  *or* **Li et al. (2023) BIRD** (recent; quantifies valid-but-wrong SQL, the exact failure the
  truth layer avoids). Pick one, not both.

**High value (1) — punches above its weight:**
- **Tan et al. (2024) — "Are Language Models Actually Useful for Time Series Forecasting?"** A
  *skeptical* result that **defends this project's design choice** (keep the LLM away from the
  numbers) and adds post-2023 currency. See §5.5 below.

**Optional (≤1) — only if the section still feels thin:**
- One of **Yao et al. (2023) ReAct** *or* **Schick et al. (2023) Toolformer** to round out the
  tool-use family. One — they are redundant with PAL/PoT for the point being made here.

**Deliberately skip (looked relevant, are not, for *this* project):**
- LLM-*for-forecasting* papers (Time-LLM, Gruver et al., LLM-for-TS surveys) — this project uses
  ARIMA and deliberately does **not** forecast with an LLM; citing them prominently would muddy
  the stance. Landscape-awareness at most, one sentence, probably skip. (The Tan et al. counter-
  argument above is the exception worth keeping.)
- FinQA / extra faithfulness papers beyond Maynez — the faithfulness ground is already covered by
  Ji et al. (2023) + Maynez et al. (2020).

### Semantic parsing / text-to-SQL

Search terms: *semantic parsing natural language database*, *text-to-SQL*, *NL interface to
databases*, *execution accuracy*, *Spider benchmark*, *BIRD benchmark*.

- Zelle & Mooney (1996) — inductive logic programming for parsing database queries (GEOQUERY).
  The classical origin point of the field.
- Zhong, Xiong & Socher (2017) — Seq2SQL / WikiSQL. Neural generation of SQL from language.
- Yu et al. (2018) — Spider. The cross-domain, multi-table benchmark; important because it
  measures generalisation to *unseen schemas*.
- Rajkumar, Li & Bahdanau (2022) — evaluation of LLM text-to-SQL capability.
- Li et al. (2023) — BIRD. Large-scale, realistic database-grounded text-to-SQL; useful for the
  "scores sit well below 100% on realistic schemas" point.
- Pourreza & Rafiei (2023) — DIN-SQL. Decomposed in-context learning with self-correction;
  relevant because self-correction is an *acknowledgement* that generated queries are often wrong.

### Tool-use / program-aided LMs

Search terms: *program-aided language models*, *tool-augmented LM*, *function calling*,
*program of thought*, *augmented language models survey*.

- Gao et al. (2023) — PAL: program-aided language models. The clearest statement of "the model
  writes the code, the interpreter runs it."
- Chen et al. (2022) — program-of-thought prompting; disentangling computation from reasoning for
  numerical tasks. Directly on-point for this project's founding principle.
- Schick et al. (2023) — Toolformer; models learning to call APIs.
- Parisi, Zhao & Fiedel (2022) — TALM: tool-augmented language models.
- Yao et al. (2022/2023) — ReAct; interleaved reasoning and acting. Closest to `agent.py`'s
  plan → execute → observe structure.
- Mialon et al. (2023) — augmented language models survey. Useful as a single anchoring citation
  covering the whole family.

### Faithfulness / hallucination (extending the existing single citation)

Search terms: *faithfulness abstractive summarization*, *hallucination detection*, *LLM-as-judge*,
*RAG evaluation faithfulness*.

- Ji et al. (2023) — already cited in §2. Keep.
- Maynez et al. (2020) — faithfulness and factuality in abstractive summarisation. Establishes
  the faithfulness-vs-factuality distinction, which maps directly onto this project's
  invented-vs-dropped number distinction in `llm.faithfulness()`.
- Min et al. (2023) — FActScore; atomic-fact-level verification of generated text.
- Es et al. (2023) — RAGAS; automated RAG evaluation including a faithfulness metric.
- Zheng et al. (2023) — judging LLM-as-a-judge (MT-Bench). Directly relevant to
  `query_coverage.score_against_llm_judge` — cite this to justify the LLM-judge method *and* its
  known biases, which `06-query-routing-coverage.md` already documents empirically.

### 5.5 LLMs and time series (currency + a useful counter-argument)

Search terms: *large language models time series forecasting*, *LLM time series survey*, *are LLMs
useful for time series*, *reprogramming LLM forecasting*.

- **Tan et al. (2024)** — "Are Language Models Actually Useful for Time Series Forecasting?" A
  skeptical, empirical result: LLM-based forecasters often fail to beat simple baselines. Cite it
  **for** this project, not against — it independently supports the core stance that the LLM should
  *explain*, not *compute* or *forecast*. Citing a paper that argues against one's own tool is good
  academic form and shows the stance was chosen, not assumed. (Recalled from training — **verify**.)
- Gruver et al. (2023), Jin et al. (2024) Time-LLM, and LLM-for-TS surveys exist for landscape
  context, but are **lower priority** here (see §5.0 "skip"): this project forecasts with ARIMA, not
  an LLM, so these describe a road not taken. One sentence of acknowledgement at most.

### 5.6 Verification of NL-to-data interfaces (related work + future work)

- **Schwabe et al. (2025) — "Q-NL Verifier: Leveraging Synthetic Data for Robust Knowledge Graph
  Question Answering" (arXiv:2503.01385).** *Confirmed by reading the paper (abstract + method
  page), not recalled — details below are reliable.* A learned **ModernBERT cross-encoder** scores
  0–1 semantic equivalence between a SPARQL query and its NL paraphrase (threshold τ=0.6), used to
  filter LLM-generated synthetic training pairs and as a quality metric (rank-correlates with human
  judgement at ρ=1.0, beating BLEU/BERTScore). **Relevance:** it is the *learned, supervised* cousin
  of this project's verification idea — and its τ=0.6 similarity threshold is conceptually the same
  move as the `DOMAIN_THRESHOLD` used in `interface.py`'s semantic fallback router.
  - **Related-work framing:** cite as prior art for "add a verifier to an NL-to-structured-data
    system." Distinguish clearly: Q-NL verifies *NL↔query equivalence* with a trained model; this
    project verifies *numbers-in-prose trace to computed facts* with a deterministic check — a
    different object verified, and exact rather than probabilistic.
  - **Future-work framing:** a learned routing verifier that scores *question ↔ plan* agreement
    (does the parser's intent/sensor/period actually match what was asked?) — the one verification
    layer this pipeline lacks. Deliberately **not** implemented: it would need a synthetic labelled
    dataset (re-introducing the self-authored-ground-truth problem), add a trained-model dependency,
    and trade away the deterministic router's auditability. Named as future work, cf. Schwabe et al.

### Also fix while you are in there

§2 currently has **no search-strategy description** — no databases, no search terms, no
inclusion/exclusion criteria. This matters independently of any marking scheme: without it, a
reader cannot distinguish "the gap is real and was established by searching" from "the gap is an
assumption nobody checked." Since §2's whole job is to establish the gap, that ambiguity
undermines the section's purpose no matter how good the citations are. Add a short paragraph:
which databases, which search terms, what inclusion/exclusion criteria, roughly how many papers
screened versus included.

Also note: the newest citation currently in §2 is from 2023, and the report is dated 2026. A
three-year gap in the fastest-moving area of the topic is conspicuous. Most of the search terms
above will surface 2024–2026 work.

---

## 6. The optional experiment — and the argument against doing it

A text-to-SQL **baseline comparison** is technically available and would be a genuinely strong
piece of evidence: load the dataframe into SQLite, prompt the local model to generate SQL for the
existing 72-question corpus, execute, and measure how often it produces a *confidently wrong*
answer versus how often this project's router declines or misroutes. That would convert the §3.2
trade-off from an argument into a measurement.

**Recommendation: do not do this**, at least not before the write-up is finished. The reasoning is
about where the marginal effort actually pays:

- The evaluation is **already this project's strongest component** — McNemar significance test,
  bootstrap CIs, 8-seed benchmarking, point-adjusted metrics, ablation, adversarial probe, honest
  negative results. It does not need more evidence. Adding a sixth kind of rigour to a section
  that already has five has sharply diminishing returns.
- The literature review and the conclusions are the two genuinely weak written components, and
  both are fixable with prose alone — no new code, no new experiments, using evidence that already
  exists.
- The baseline costs 1–2 days and strengthens the part that is already strong. The rewrites cost
  comparable time and fix the parts that are actually weak.

If time remains after the write-up is genuinely finished, revisit. Not before.

---

## 7. Summary

- **Both items are write-up. Neither requires code.**
- **Tool-use**: already built (`agent.py`). Needs naming and citing. The addition on top of the
  standard pattern — the faithfulness gate on the synthesis step — is a real, small, defensible
  contribution supported by the adversarial-probe evidence already collected.
- **Text-to-SQL**: deliberately not built. Cited as the competing paradigm. Its valid-but-wrong-
  query failure mode is what makes this project's constrained-tool-space design defensible, and
  the comparison reframes §7.1's confessed vocabulary limitation as the mechanism delivering the
  guarantee.
- **The honesty constraint is not optional**: this project solves an easier problem than
  text-to-SQL does. Frame as a trade-off, never as a win. The evaluation docs already hold this
  line; the dissertation must too.
- **Every citation above must be independently verified before use.**
