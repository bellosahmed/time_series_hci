# Project Assessment — Grading v1, v2, v3

*An honest, examiner's-eye grading of the three systems and the project as a whole. These are **indicative,
formative** grades on the delivered technical artefact, its evaluation, and its documentation — not a mark
for the final dissertation write-up (still a draft). They are deliberately critical: the point is to show
where marks are won and where they are at risk, not to flatter.*

## How this is graded

Against a UK MSc classification — **Distinction ≥ 70 · Merit 60–69 · Pass 50–59** — across eight criteria a
Data-Science / Computer-Science examiner would weigh for a systems-plus-evaluation dissertation:

1. Problem framing & motivation
2. Data handling & rigour
3. Methods — breadth and *appropriateness*
4. Evaluation quality & **honesty**
5. System engineering & reproducibility
6. HCI / interface & explainability
7. Documentation & communication
8. Contribution / originality

**Headline grades (indicative):**
- **v1 — Weather:** ~**72** (low Distinction)
- **v2 — Trade:** ~**74** (mid Distinction)
- **v3 — Network security:** ~**76** (mid–high Distinction)
- **Project as a whole:** ~**76** (Distinction) — with one scope risk that caps the very top band (below).

---

## v1 — Weather (Jena) · ~72 / Distinction (low)

The **reference implementation** and the broadest system. It is where the architecture, the widest method
set, and the supporting machinery were built and proven.

**Where it earns marks.**
- **Methods breadth (excellent):** six anomaly families (z-score/STL, Isolation Forest, LOF, One-Class SVM,
  Matrix Profile, ARIMA-residual) *plus* three deep detectors, *plus* a four-retriever RAG comparison with
  real IR metrics (P@k/MRR/MAP/NDCG), *plus* agentic multi-step tool-use. Very strong for an MSc.
- **Engineering & reproducibility (excellent):** **183 tests**, `verify_results.py` **29/29**, ruff + mypy
  clean, ~92% coverage, one-command reproduce. This is above typical MSc standard.
- **Explainability:** multi-sensor co-movement reasoning + trust labels + faithfulness-gated LLM.

**Where it loses marks.**
- **Evaluation validity (the ceiling):** Jena has **no real labels**, so detection is measured on
  *injected* anomalies. Honest, but it structurally caps the strength of the detection claim — the single
  biggest reason this sits below v2/v3.
- **No human evaluation** of the "interface" (a descoped item, but an examiner still notices).

**What would raise it:** it can't easily fix the label problem (that's *why v2/v3 exist*), so its remaining
lever is the write-up — framing v1 explicitly as the *architecture-and-breadth* contribution.

## v2 — Trade (EU) · ~74 / Distinction (mid)

The **external-validity** system: the same architecture on real, datable economic anomalies.

**Where it earns marks.**
- **Real anomalies (2008, COVID, 2022)** — detectors tested against events that actually happened, a clear
  step up in evaluation validity.
- **Evaluation honesty (strong):** point-adjusted metrics, an out-of-sample **SARIMA backtest**, and the
  openly-reported result that the **autoencoder (F1 0.21) loses to Isolation Forest (0.36)** — reporting a
  negative deep-learning result is exactly the maturity examiners reward.
- **Best-documented version:** **14 executed notebooks** — the strongest guided narrative of the three.
- **Domain-appropriate reasoning:** multi-country shared-shock vs isolated classification.

**Where it loses marks.**
- **Coarse labels & small data:** ground truth is crisis *months* (~14) over only **292** points, so
  strict per-point precision is necessarily low and deep models are starved.
- **Narrower method set** than v1, and **web-only** (no CLI, one fewer interface surface).

**What would raise it:** a higher-resolution companion series (e.g. daily/weekly trade or an hourly proxy)
would let the deep models and point metrics breathe — the honest "future work" it already names.

## v3 — Network security (DAD) · ~76 / Distinction (mid–high)

The **most rigorous evaluation** in the project, on the best labels, with the most original method design.

**Where it earns marks.**
- **Best labels:** **real, per-packet** attack ground truth (544 labelled packets) — directly fixes v1's
  core weakness.
- **Evaluation rigour (the standout):** average precision + **bootstrap confidence intervals**,
  **per-attack-type recall**, **point-adjusted** metrics, and **day-blocked cross-validation**. The CIs and
  per-type breakdown yield a genuinely sophisticated, honest finding: *no fusion config is significantly
  best on AP, and the learned combiner is worst.* This is PhD-flavoured evaluation discipline.
- **Original design:** a two-channel (value + flow) detector with early/late/supervised fusion, and the
  disciplined **exclusion of the attacker-IP feature** to avoid label leakage.
- **Honesty:** the weak flow channel is *measured and reported*, not hidden.

**Where it loses marks.**
- **Modest absolute performance** (AP 0.10–0.18): honest given no leakage and untuned detectors, but an
  examiner still reads the raw numbers.
- **Dataset scope:** "semi-synthetic" traffic, one data centre, seven days, one attacker; 71% of attacks
  only typeable as "mixed."
- **Less polish than v1** (untuned detectors; the two-channel work is newer).

**What would raise it:** a tuned/seasonal value model and a stronger flow channel to lift the absolute
scores, plus more captures to enlarge the tiny per-type strata — again, already named as future work.

---

## The project as a whole · ~76 / Distinction — with one capping risk

Graded as *one body of work*, the three versions are considerably stronger than the sum of their parts,
because they turn a single claim into a tested, three-domain argument.

**The distinction-level strengths.**
- **A domain-agnostic architecture, demonstrated** on weather, economics, and network security — the same
  truth-layer + explanation-only LLM + faithfulness gate transferring with only the data layer changing.
  Across all three, the LLM introduced **0 fabricated numbers**. This is a *reusable method*, not a demo.
- **A coherent research thesis** carried by the version order: a deliberate **ladder of label quality**
  (synthetic → real-coarse → real-fine), where each version fixes the previous one's main weakness.
- **Evaluation honesty as the through-line** — real labels, event-level metrics, confidence intervals,
  per-type breakdowns, blocked CV, and negative results reported openly. This is the single most
  distinction-defining quality of the work.
- **Reproducibility & verification** — independent `verify_results.py` trust gates (29/29, 13/13, 12/12) on
  top of ~318 tests total, one-command reproduce scripts, and newcomer guides.

**The one risk that caps the very top band.** The dissertation title is *"…Through a Natural HCI
Interface,"* yet there is **no human evaluation** of the interface (usability, task accuracy vs a
spreadsheet, trust/SUS). It is a deliberately descoped item, and the *explainability engineering* is
genuinely strong — but a strict examiner marking against the title will see the absence of user evidence as
the main thing standing between "solid Distinction" (~76) and "high Distinction" (80+). The **highest-leverage
single improvement to the grade** is therefore either (a) a small user study, or (b) a write-up that
re-frames the contribution precisely as *verifiable NL analytics + honest evaluation*, so the title and the
delivered evidence line up.

**Second-order levers (smaller, cheaper):**
- Reconcile the stale counts across docs (v1 README says "173" vs the real **183** tests; CLAUDE.md "182").
  Trivial to fix, but inconsistencies read badly to an examiner.
- Lift v3's absolute detection scores (tuned/seasonal value model, stronger flow channel).
- A short "threats to validity" paragraph per version in the final write-up (much of it already exists in
  `04_limitations_and_future_work.md`).

## Bottom line

A **Distinction-level** project (~76). Its durable strength is a discipline, not a model: *honest
evaluation carried across three domains on a single, verifiable architecture.* Its ceiling is set less by
the engineering — which is already above MSc standard — than by aligning the final write-up (and, if brought
back into scope, a modest user study) with the "HCI interface" framing in the title.
