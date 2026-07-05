# v1 — Assessment & Recommendations

A review-later snapshot: what to add/improve, how a supervisor is likely to grade/criticize/praise
the work, and notes to keep. Dated **2026-07-04**. Companion to `UNDERSTANDING_V1.md`.

**Current state at time of writing:** 120 tests pass (1 skipped), `verify_results.py` 25/25, ruff
clean, pushed to the private repo `bellosahmed/time_series_hci` (`main`).

> **Update 2026-07-05:** items **4 (agentic tool-use)** and **7 (all polish: larger-model option,
> embedding-RAG option, `resolve_threshold` fix, mypy)** below are now **done** — see
> `AGENTIC_AND_POLISH.md`. Quality gates are now **140 tests** pass (1 skipped), verify 25/25, ruff
> clean, mypy clean. Remaining critical path is unchanged: **HCI user study → ethics → CI/tag.**

---

## 1. Recommendations — what to add & improve

### Highest leverage (do these)
1. **Run the HCI user study.** The #1 thing. The title is *"…Through a Natural HCI Interface,"* and
   right now there is an interface but no human evaluation. Even 8–12 participants measuring **SUS
   score, task accuracy vs a spreadsheet, time-on-task, and a trust rating** would close the single
   biggest gap. Everything else is polish by comparison.
2. **Start ethics approval now** — usually the bottleneck for the study above, not the study itself.
3. **Add CI (GitHub Actions)** running `pytest + ruff + verify_results.py` on every push. Cheap, and
   it visibly proves the "25/25" claim holds continuously. Works fine on a private repo.

### Strengthens the contribution
4. **Agentic, still-numberless tool-use** — let the LLM orchestrate multiple query-engine calls for
   multi-step questions ("compare the three hottest summers and explain why") via function-calling,
   while never computing a number. Pushes "Verifiable NL Analytics" toward publishable.
5. **A short demo (GIF/video + screenshots) in the README** — examiners like seeing it work.
6. **Coverage number** (`pytest-cov`) and a `git tag v1.0` frozen at the exact numbers in the report.

### Smaller improvements
7. Offer a **larger local model option** — the 3B model is the weak link; the faithfulness gate
   catches invented *numbers* but wrong *qualifiers* rely on prompting.
8. Optional: swap TF-IDF RAG for a local embedding model; fix the documented `resolve_threshold`
   year-edge-case; add `mypy`.

---

## 2. From the supervisor's point of view

### Likely grade
**Distinction-capable (≈68–75)** on engineering and intellectual contribution — but currently
**gated to high-Merit / borderline-Distinction by the missing human study.** The earlier
supervisor-style feedback said "close three things → comfortable Distinction"; two of the three
(statistical rigor, report results) are now **done**. Run the study and it's a clean Distinction.

### What they'll be happy about (praise)
- The **core idea is genuinely novel and timely** — deterministic truth layer + LLM explanation +
  *automated faithfulness verification* directly tackles LLM hallucination in analytics.
  Distinction-level thinking.
- **Serious engineering:** clean layered architecture, 120 tests, an *independent* 25/25 harness
  proving numbers == pandas, reproducibility (seeds, pinned deps, `reproduce.sh`), `pyproject` +
  ruff-clean.
- **Unusually honest evaluation:** bootstrap CIs, McNemar significance, point-adjusted metrics,
  class-imbalance caveats, *and* both synthetic and real-event labels. Rare maturity at MSc level.
- Domain care that shows real understanding: the **circular wind-direction fix** and the
  **faithfulness gate**.
- The **v2 generalization appendix** — evidence it's a reusable *method*, not a one-off weather demo.

### What they'll criticize
- **"Where is the human evaluation?"** — the defining critique. HCI as a discipline evaluates
  interaction *with people*; there is a protocol, not results.
- **Single core dataset** (Jena) — external validity. Partly mitigated by v2, but they'll press it.
- **Weak LLM** — a 3B model; they may probe robustness (wrong qualifiers, edge phrasing).
- **Synthetic anomaly labels** — honest about it, but it caps how strong the detection evaluation
  can be.
- **Retrospective, not operational** — batch analysis, not real-time; limits "practical impact"
  claims.

---

## 3. Notes to keep

- **Keep the repo private until after marking** — a public mirror can trip Turnitin/originality
  checks against the submission.
- **Keep `WORKLOG.md` going** — the methodology/discussion goldmine, and evidences process for the
  viva.
- **Before final submission:** re-run `verify_results.py` + tests, then `git tag` a release so the
  examiner can reproduce the exact numbers in the report.
- **Sync discipline:** whenever a number changes in code, update the report (and vice-versa) — the
  harness is the tripwire.
- **Document the faithfulness check's honest limit** in the report: it is majority-of-3 advisory and
  catches numbers, not all wrong context — stating this pre-empts a criticism.

---

## Priority order (one-line summary)
1. HCI user study (unlocks title + grade) → 2. Ethics paperwork (do in parallel) →
3. CI + tagged release (cheap credibility) → 4. One ambitious extension (agentic tool-use) →
5. Polish (demo video, larger model option).
