# HCI User Study — Protocol (v1)

A plan to **evaluate the natural-language interface with real users** — the one examiner item
that needs people, not code. Designed so results are objective: because the system computes exact
answers, **task correctness can be graded against ground truth**.

> Status: protocol ready to run (not yet executed). Target: 10–15 participants.

## 1. Research questions

- **RQ1 (Effectiveness).** Can non-experts get **correct** answers about a time-series dataset using
  the natural-language interface, compared with a spreadsheet?
- **RQ2 (Efficiency).** Is it **faster**?
- **RQ3 (Trust/Confidence).** Do users **trust** the answers, and does the trust labelling
  (🔢 computed / 📚 background / 🧠 reasoning) affect confidence?
- **RQ4 (AI vs template).** Does "Explain with AI" wording improve understanding over the plain
  template, without reducing trust?

## 2. Hypotheses

- H1: task **accuracy** is higher with the interface than the spreadsheet (we expect a large effect,
  since exact answers remove arithmetic mistakes).
- H2: **time-on-task** is lower with the interface.
- H3: **SUS** (usability) and **trust** ratings are high (> 70 SUS).
- H4: AI-phrased answers score higher on **comprehension** with no drop in trust.

## 3. Design

- **Within-subjects** (each participant uses every condition; counterbalanced order to cancel
  learning effects).
- **Conditions:**
  1. **Spreadsheet** (the raw CSV in Excel/LibreOffice) — control/baseline.
  2. **Interface, template mode** (AI off).
  3. **Interface, AI mode** (local LLM phrasing).
- **Dataset:** the Jena climate data the system already serves.

## 4. Participants

- **10–15** participants; mix of **non-experts** (no data-analysis background) and a few
  **semi-experts** (some Excel/stats). Recruited from the department / course peers.
- Inclusion: comfortable using a computer; English-speaking. No special skills required.

## 5. Tasks (8, each with a known correct answer)

Graded automatically against the query engine / pandas ground truth.

| # | Task | Tests |
| --- | --- | --- |
| 1 | What was the hottest temperature ever, and when? | extremum + timestamp |
| 2 | What was the coldest it got in 2010? | year scoping |
| 3 | How many times did temperature exceed 35 °C? | threshold |
| 4 | What was the weather like on 5 July 2014? | day report |
| 5 | Was 2013 hotter than 2014? | comparison |
| 6 | Were there any unusual readings in 2014, and what happened? | anomaly + reasoning |
| 7 | Is temperature rising over the years? | trend |
| 8 | Roughly what will the temperature be next week? | forecast |

## 6. Measures

- **Task success** (0/1, graded vs ground truth) — primary, objective.
- **Time on task** (seconds) — screen-recorded.
- **Errors / retries** per task.
- **Confidence** (1–7 Likert, per task): "How confident are you this answer is correct?"
- **Trust** (1–7): "How much do you trust the system's numbers?" + a question on whether the
  🔢/📚/🧠 labels helped.
- **Comprehension** (AI vs template): did they correctly restate the finding in their own words?
- **Usability: SUS** (10-item System Usability Scale) at the end.
- **Workload: NASA-TLX** (raw, 6 items) after each condition.
- **Qualitative:** short semi-structured interview (what was confusing / liked / wanted).

## 7. Procedure (~35–40 min/participant)

1. Consent + 2-min demographics (no identifying data).
2. 3-min tutorial of each condition (one warm-up task).
3. Counterbalanced blocks: do the 8 tasks in each condition (different but matched task phrasings to
   avoid memorising answers).
4. After each condition: NASA-TLX + trust items.
5. End: SUS + interview.

## 8. Analysis

- Accuracy & success: per-condition proportion; **McNemar / Cochran's Q** (within-subjects, binary).
- Time, confidence, TLX: **repeated-measures ANOVA** or **Friedman** (non-parametric) + post-hoc.
- SUS: report mean (benchmark 68 = average).
- Qualitative: lightweight thematic coding.
- Report **effect sizes** and **95% CIs**, not just p-values; n is small so frame as **indicative**.

## 9. Ethics & data

- Lightweight ethics approval (departmental). **Informed consent**; participants can stop anytime.
- **No personal data** stored; recordings anonymised and deleted after analysis. Fully **offline**
  system — no participant data leaves the machine (a study strength to highlight).

## 10. Pilot & validity

- **Pilot with 2 people** first to fix wording/timing.
- Threats: small n (→ frame as indicative, not confirmatory); learning effects (→ counterbalancing +
  matched task variants); novelty effect; experimenter bias (→ scripted instructions).

## 11. What "good" looks like

Interface accuracy near 100% (exact answers) vs spreadsheet error rate; lower time-on-task; SUS > 70;
high trust with the labels rated helpful. Any of these would be a strong, defensible HCI result for
the dissertation.
