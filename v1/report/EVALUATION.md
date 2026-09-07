# v1 Evaluation Summary

Evaluation of the **core (computational) pipeline** on the Jena Climate dataset
(2009–2016, 70,129 hourly rows after preprocessing). RAG + LLM and the interface
are intentionally out of scope for v1. Full reproducible workings: `notebooks/02_methods_and_evaluation.ipynb`.
> **Superseded (15 July 2026, updated 16 July 2026): current, actively-maintained evaluation methodology
> and numbers now live in [`v1/docs/evaluation/`](../docs/evaluation/README.md)** — one page per method
> (truth layer, anomaly detection, forecasting, retrieval, LLM faithfulness, query-routing coverage,
> `recommend`/correlation explainability), each with the exact command that reproduces its numbers,
> regenerated on 15–16 July 2026 directly from the code. This file is kept as a historical narrative
> snapshot; where it disagrees with `docs/evaluation/`, the latter is correct. Known stale points below:
> the check counts ("17/17", "29 independent checks") are outdated — `verify_results.py` currently runs
> **35/35** independent checks (added: query-routing coverage, a held-out generalisation check, then
> four checks covering the `recommend` intent and correlation explainability added 16 July 2026).
>
> **Note (this file predates the 9→5 notebook consolidation and a later detector/reporting refresh):**
> the exact figures below in "Labeled evaluation" and "Quantitative rigor" were produced by an earlier
> evaluation protocol and could not be reproduced byte-for-byte against the current codebase when this
> file was last audited. §5.3/§5.7 of `report/REPORT.md` supersede them with figures freshly verified
> against the current code (`reproduce.sh`); treat this file as a historical snapshot for those two
> sections. Everything else in this file (query-engine correctness, clustering, retrieval IR metrics,
> forecast backtest MAE/RMSE/MASE, offline operation) has been independently re-verified and is current.

## 1. Query engine correctness (the trust guarantee)

Every exact query was compared against pandas computed independently.

| Query | Result |
| --- | --- |
| max / min / mean / std `T (degC)` | ✅ exact |
| correlation(T, air density) | ✅ exact |
| missing-value count | ✅ exact |
| timestamps of hottest/coldest | ✅ match `idxmax` / `idxmin` |

**Pass rate: 6/6 numeric checks exact** (rtol/atol 1e-9), plus matching timestamps.
Unit tests (`tests/test_query_engine.py`) — **8/8 passing**.

> Sanity of values: mean 9.4 °C, range −22.7 °C (07 Jan 2009) to 37.0 °C (04 Jul 2015);
> T correlates +0.997 with potential temperature and −0.96 with air density — physically correct.

## 2. Anomaly detection (three methods)

| Method | Type | Flagged | % of points |
| --- | --- | --- | --- |
| Residual z-score (4σ) | statistical | 526 | ~0.75% |
| Isolation Forest (multi-sensor) | classical ML | 702 | ~1.00% |
| 1D CNN autoencoder (top 1% error) | deep | 702 | ~1.00% |

Flagged residual-anomaly points have a **mean \|residual\| of 8.09 °C vs 1.00 °C overall** —
genuinely extreme, not noise. The methods agree only partially (z-score & CNN-AE share 75),
which is expected: z-score targets large residuals, Isolation Forest unusual *multi-sensor*
combinations, and the autoencoder unusual *shapes*.

The **CNN autoencoder** is unsupervised, trains on CPU in ~2.4 s on 70k points, and produces
a smooth per-timestamp reconstruction-error score.

### Labeled evaluation (synthetic anomaly injection)

Because Jena has no labels, we injected 100 known spikes and measured recovery:

| Detector | Precision | Recall | F1 |
| --- | --- | --- | --- |
| Residual z-score | 0.17 | 0.95 | 0.29 |
| CNN autoencoder | 0.14 | 1.00 | 0.25 |
| LSTM autoencoder | 0.14 | 1.00 | 0.25 |
| LSTM forecaster | 0.14 | 1.00 | 0.25 |

All achieve **very high recall** (they catch injected anomalies). Apparent precision is low
because the data *also* contains genuine background anomalies that get flagged but count as
"false positives" against synthetic-only labels — a useful illustration of why labels matter.
On a controlled clean signal (see `report/MODELS.md`), precision is high too.

### Deep techniques compared (notebook 07)

Three unsupervised deep detectors, all trained on CPU in 1–3 s on 70k points:

| Technique | Idea | Reacts to |
| --- | --- | --- |
| **CNN autoencoder** | convolutional reconstruction | unusual window *shapes* |
| **LSTM autoencoder** | recurrent reconstruction | unusual *ordered* sequences |
| **LSTM forecaster** | predict next step, error = anomaly | sudden *jumps* (one step late) |

They overlap substantially but not completely. The CNN-AE is the simplest/fastest default;
the LSTM variants serve as comparison baselines.

## 3. Clustering (day-types)

K-means (k=4) on per-day feature vectors yields temperature-ordered clusters with
mean daily temperatures of **0.5, 6.5, 12.2, 19.6 °C** — a clean cold→warm gradient
that maps onto winter → spring/autumn → summer. Clusters are meaningful.

**Choice of k (justified, not assumed).** `clustering.kmeans_selection` scores inertia (elbow) +
silhouette across k (plotted in notebook 02). Honestly, the **silhouette peaks at k=2** (0.287, a coarse
warm/cold split); **k=4** (0.173) is a *deliberate interpretability trade-off* — four seasonal day-types
at a modest silhouette cost, with the inertia elbow bending around k≈3–4. Stating this pre-empts the
"why k=4?" question.

## 3b. Retrieval quality (RAG retriever — IR metrics)

The truth layer is verified `== pandas`; the RAG **retriever** that grounds explanations is
evaluated separately with standard IR measures (Precision@k, Recall@k, MRR, MAP, NDCG) over a
10-query standard set + a 5-query paraphrase set (query → relevant sensor doc). Retrievers compared on
the same labels — the table is with the **original one-line docs** (the finding below shows what
enriching them did):

| retriever | standard R@3 | standard MRR | paraphrase R@3 | paraphrase MRR |
| --- | --- | --- | --- | --- |
| TF-IDF (VSM baseline) | 0.900 | 0.900 | 0.400 | 0.267 |
| BM25 (Okapi) | 0.900 | 0.900 | 0.200 | 0.067 |
| LSA (TF-IDF → truncated SVD) | 0.900 | 0.900 | 0.400 | 0.267 |
| **Dense** (sentence-transformers `all-MiniLM-L6-v2`) | **1.000** | **0.933** | **0.600** | **0.500** |

**Finding (honest):** with the *original* one-line docs the three keyword methods tie on standard
queries (MRR 0.900) and the dense encoder leads on paraphrases (0.500 vs 0.267). But **enriching the
knowledge base** with everyday synonyms (a cheap ~14-string edit, made after this evaluation exposed the
gap) closed it for everyone — paraphrase MRR rose to **1.000 for TF-IDF/BM25/LSA** and 0.867 for dense.
So on a small curated KB, **document quality dominates the retriever**: better descriptions beat a neural
model, and cheap TF-IDF becomes the sensible default. Full before/after + per-query evidence:
`docs/archive/RETRIEVAL_EVALUATION.md`.

**Held-out generalisation (not teaching to the test).** Because enrichment adds the words the paraphrase
set probes, we also test a **held-out set** (n=6) whose every descriptive word is *verified absent* from
the KB. There the keyword methods collapse to **MRR 0.250** (no word to match) while the **dense encoder
generalises to 0.556** — real but imperfect. The fair claim: *enriched docs win on the everyday words
users use; the dense retriever is what still helps on words the KB has never seen.* Two `verify_results.py`
checks guard the baselines. Together with the LLM **faithfulness** gate (the RAG "faithfulness" measure),
both halves of the RAG system — retrieve well, generate faithfully — are now evaluated.

## 4. Offline operation

The entire pipeline uses only local libraries (pandas, numpy, scikit-learn, statsmodels,
matplotlib) on a local CSV. **No network calls, no external API** — the privacy-preserving
requirement holds.

## Limitations

- Jena has **no anomaly labels**, so detector *accuracy* (precision/recall) cannot be measured
  directly — only agreement and extremeness sanity checks.
- z-score assumes roughly symmetric residuals; heavy-tailed sensors can inflate false positives.
- Isolation Forest `contamination` is a chosen prior, not learned.
- Deep model (1D CNN), RAG + local LLM explanation, and the user interface are not yet built.

## Ground truth & honesty (how we know results are true)

Because Jena has **no official anomaly labels**, we are explicit about what *is* and *isn't*
hard ground truth:

| Result type | How it's verified | Strength |
| --- | --- | --- |
| Exact stats (max/min/mean/correlation, scoped queries, thresholds) | recomputed independently with **pandas** — must match exactly | **hard ground truth** |
| Anomaly precision/recall | measured on **synthetic injected spikes** (known positions) | controlled, not real labels |
| Real anomalies | **cross-checked against known weather events** (below) | corroboration, not labels |
| LLM explanation | checked that it **preserves the computed numbers** | hard (numbers from code) |

**Known-event corroboration.** The extremes and detectors line up with documented reality:

- The all-time **maximum 37.04 °C on 4 Jul 2015** coincides with the **Central European July 2015
  heatwave**; `verify_results.py` asserts the record falls in summer.
- The all-time **minimum** falls in **winter** (Dec–Feb), as physics demands.
- 2014's clearest cold anomaly (26 Jan) shows a **physically consistent signature** — temperature,
  dew point and potential temperature all drop while air density rises together — which the
  multi-sensor reasoning labels a *real event*, not a sensor fault.

**One-command check.** `python3.12 verify_results.py` runs **30 independent checks** (query engine ==
pandas, scoped queries, thresholds, known-event placement, reasoning verdict, forecast bounds,
point-adjusted ≥ strict recall, confusion sums to N, retriever IR metrics, day-type clustering,
query-routing coverage, statistical-rigor checks) and exits non-zero on any failure. See
`docs/evaluation/` for the current per-method breakdown and reproducing commands.

**Point-adjusted metrics.** Alongside strict precision/recall we also report **point-adjusted**
scores (`evaluation.point_adjust`): if a detector flags any point inside a true anomaly segment, the
whole segment is credited — the standard fairer metric for time-series anomalies.

## Quantitative rigor (measured, reproducible)

**Anomaly detection — repeatable protocol** (`evaluation.benchmark`, synthetic spikes over **10
random seeds**, z-score-on-STL-residual detector): strict **P 0.77 ± 0.13, R 0.66 ± 0.07, F1 0.71**;
point-adjusted **P 0.77, R 0.67, F1 0.71**. Reporting mean ± std (not a single run) makes the
numbers defensible.

**Parameter choices justified, not arbitrary** (`anomaly.threshold_sensitivity` /
`contamination_sensitivity`). The z-score anomaly count falls smoothly with the threshold
(2.0 → 5629, 3.0 → 1917, **4.0 → 526**, 5.0 → 128); **4.0** is the knee that keeps only clear events
(~0.75 % of points). Isolation-Forest count scales with the contamination prior
(0.005 → 351, **0.01 → 702**, 0.05 → 3507); **0.01** keeps the flagged rate conservative.

**Forecasting is backtested** (`forecasting.backtest`, rolling-origin, 5 folds, 14-day horizon on
daily means): **MAE 3.55 °C, RMSE 4.28 °C, MASE 0.83** — it **beats a persistence baseline**
(MASE < 1). MAPE is intentionally avoided (temperature crosses 0 °C, which makes it explode).

**LLM explanation faithfulness** (`llm.faithfulness`, 6 sample questions, llama3.2:3b): **0 invented
numbers** — the model never fabricated a figure, so the core trust property holds. In 2 / 6 it
*omitted* one secondary value when summarising a multi-value fact (e.g. wind, or one reasoning
sensor); template mode shows every figure, so nothing is actually lost. The serious failure mode
(hallucinated numbers) did not occur.

## Verdict

The computational core is **correct, sensible, and fully offline**, and now **machine-verified**
(`verify_results.py`, **35/35** as of 16 July 2026 — see `docs/evaluation/`) and **quantitatively evaluated**
(multi-seed anomaly benchmark, parameter sensitivity, forecast backtest with MASE < 1, LLM faithfulness
with 0 invented numbers, and a query-routing coverage benchmark at 100%).
Exact figures are hard ground truth (== pandas); anomaly accuracy is measured on synthetic labels
and corroborated against known real events; the LLM only rewords and provably preserves the numbers.

**Known evaluation gaps (honest, from a supervisor-style review, 15 July 2026 — updated same day once
three of the four were addressed):** the "HCI" half of this project's claims still has **no human
evaluation** — every trust/explainability result here is validated computationally, not with real users,
so it cannot show that a person's trust was actually calibrated appropriately by the interface (the HCI
user study itself is out of scope by explicit decision; the underlying limitation is still real and
stated here for completeness).

The other three gaps this section originally flagged now have real, if partial, evidence rather than
being open:
- **Faithfulness gate ablation**: `llm.faithfulness_ablation()` found 0% invention raw vs. gated on
  ordinary routed questions (an honest null — the gate has never been *observed* rescuing a real
  question). `llm.adversarial_faithfulness_probe()` then deliberately tempted invention with mismatched
  facts and got a real, reproducible hallucination ("3.11 degrees hotter... than 2012"), which the
  actual production gate (`Interface._phrase()`) caught and discarded — demonstrated, not just asserted,
  value, though on a small sample (5 adversarial cases) that showed real run-to-run variance (20–40%
  across two runs the same day).
- **Self-authored ground truth**: `query_coverage.export_blind_worksheet`/`score_blind_labels` make it a
  five-minute task to hand the 72 questions to a real second rater — built and verified, but not yet run
  with an actual human. `llm.independent_intent_judgment` is an automated partial substitute: it shows
  ~65–72% intent/sensor agreement with the hand-authored labels, a real and informative gap, not a
  rubber stamp.
- **Self-fulfilling 100% coverage**: `query_coverage.HELD_OUT_GROUND_TRUTH` — 23 freshly-authored
  questions, written after the original fixes without reference to them — scored 22/23 on first run,
  catching one more genuine routing gap (`"variable"` missing from the spread keywords), now 23/23 after
  fixing it. The 100% is now corroborated on cases that couldn't have been used to build the fixes.

Full method, evidence, and honest caveats for all of the above: `docs/evaluation/` (especially
`05-llm-faithfulness.md` and `06-query-routing-coverage.md`).
