# How v1 is evaluated

v1 doesn't use one evaluation method — it uses **seven**, because seven genuinely different
components need rating: the truth layer, the anomaly detectors, the forecaster, the retriever,
the local LLM, the natural-language router, and the `recommend`/correlation-explainability layer.
Each has its own page in this folder, with the method explained and real, freshly-reproduced
numbers as evidence (not copied from an old report — every figure here was regenerated the day
this folder was written, 15 July 2026; §7 added 16 July 2026).

| # | Doc | What's rated | Headline metric |
| --- | --- | --- | --- |
| 1 | [`01-truth-layer.md`](01-truth-layer.md) | The exact query engine (max/min/mean/correlation/...) | exact match vs. independently-computed pandas |
| 2 | [`02-anomaly-detection.md`](02-anomaly-detection.md) | The anomaly detectors | precision/recall/F1, point-adjusted, multi-seed CI, McNemar significance |
| 3 | [`03-forecasting.md`](03-forecasting.md) | The 14-day ARIMA forecast | MASE vs. a naive persistence baseline |
| 4 | [`04-retrieval.md`](04-retrieval.md) | The RAG retriever (sensor-description lookup) | Precision@k, Recall@k, MRR, MAP, NDCG |
| 5 | [`05-llm-faithfulness.md`](05-llm-faithfulness.md) | The local LLM's phrasing | invented-number detection |
| 6 | [`06-query-routing-coverage.md`](06-query-routing-coverage.md) | The natural-language interface's routing | % of labelled questions routed correctly |
| 7 | [`07-recommend-and-explainability.md`](07-recommend-and-explainability.md) | The `recommend` intent (analog/directional search, honesty caveats) and correlation explainability (strength labels, breakdown) | independently-recomputed restriction/threshold/R² correctness, real bugs found & fixed |

All seven feed into one command that reports pass/fail across the whole system:

```bash
python3.12 verify_results.py   # 35/35 checks passed (16 July 2026)
```

## The one discipline that runs through all seven

Every method here follows the same underlying rule, even though the *mechanics* differ
completely from page to page:

1. **Define ground truth externally, before looking at what the code does.** Independently
   recomputed pandas values, synthetically injected spikes at known positions, documented real
   weather events, hand-labelled query→relevant-document pairs, hand-labelled question→expected-
   routing pairs. Never grade the code against its own output — that's circular and can't find
   anything wrong.
2. **Report a real number with real uncertainty**, not "it looks about right": a percentage, a
   confidence interval, a significance-test p-value.
3. **Report negative or mediocre results honestly** when they happen, rather than only
   publishing the flattering ones. Two concrete examples from this project: the query-routing
   benchmark's *first* run scored 91.8%, not the 100% it reached after real bugs were found and
   fixed (§6); the anomaly z-score detector misses 1 of 5 known real events (§2).
4. **Reproduce, don't just report.** Every number on every page below has the exact command that
   regenerates it, so a claim in the dissertation can be checked, not just trusted.

## Why seven *different* methods, not one

Because "is this correct" means something different for each component:

- For the **truth layer**, correctness is binary and provable — a mean is either equal to
  pandas's mean or it isn't. So it's checked with equality, not a benchmark.
- For the **anomaly detectors**, there's no ground truth in the real data at all (Jena has no
  official anomaly labels), so evaluation has to *manufacture* it — synthetically, and by
  cross-referencing documented real events — before precision/recall even become measurable.
- For the **forecaster**, "good" is relative — a forecast's error only means something compared
  to a baseline, hence MASE (error scaled by a naive persistence baseline) rather than a bare
  MAE.
- For the **retriever**, this is a ranking problem, so it borrows Information Retrieval's own
  metrics (MRR, NDCG) rather than inventing new ones.
- For the **LLM**, the system's design promise is "the model never invents a number" — so its
  evaluation question isn't accuracy, it's honesty: does every figure in its prose trace back to
  a real computed value?
- For the **router**, the question is different again: not "is the computed number right"
  (that's the truth layer's job) but "did the plain-English question get mapped to the right
  computation at all."
- For **`recommend`/correlation explainability**, correctness has an extra dimension beyond
  "is the number right": is the system honest about when its own answer is a *poor* one (a
  restricted search that never gets close to the target), and does an explanatory add-on
  (correlation strength, the seasonal breakdown) trace back to real computed evidence rather than
  a canned phrase.

Using one evaluation method for all seven would either be meaningless for most of them (e.g.
"precision/recall" makes no sense for the truth layer, which is either exactly right or a bug)
or would hide exactly the kind of failure each specific method is designed to catch.
