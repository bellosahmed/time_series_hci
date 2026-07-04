# Measured Results — data log (source of truth for the report)

Concrete numbers measured from the implemented system, to be cited in `REPORT.md`. Each entry notes
how it was produced so it is reproducible. Newest results appended as the limitation-hardening work
proceeds. Seeds are fixed; rerun with `reproduce.sh` / `verify_results.py`.

_Last updated: 4 July 2026._

---

## Test & verification status
- Unit tests: **96 passed, 1 skipped** (`pytest tests/`).
- Independent harness `verify_results.py`: **22/22** counted checks pass (+ 1 advisory LLM-faithfulness vote).
- `reproduce.sh` regenerates both in one command.

## Deployment / data robustness
- Missing dataset now raises a clear, actionable `FileNotFoundError` (names the path + how to get the
  gitignored CSV) instead of a cryptic pandas error. `create_app()` can run behind a production WSGI
  server (waitress/gunicorn); Flask dev server used for local scope (documented).

## Web client testing
- Pure render helpers extracted to `apps/web/chart.js`; behavioural unit suite `chart.test.js`
  (esc, swatch, legendHtml, drawChart) run in Node and driven from pytest (`test_web_js.py`, skipped
  if Node absent). App test confirms the server serves `chart.js`. No full browser available here, so
  true end-to-end browser testing remains future work (documented).

## Data characteristics (Jena, hourly, 70,129 rows)
- Temperature range **−22.7 °C → 37.04 °C**, mean **9.4 °C**.
- Correlation T vs air density **−0.96** (cold air denser); T vs potential temp **+0.997**.

## Query-engine correctness
- Every exact query equals independent pandas ground truth (max/min/mean/correlation, scoped &
  threshold queries, availability, skew) — exact, verified in the harness.

## Anomaly detection
- **Real-event recall (non-synthetic):** z-score/STL-residual detector catches **4 of 5** documented
  events (recall 0.80); misses **2015 July heatwave** (STL absorbs sustained warmth → residual doesn't
  spike). `evaluation.events_recall`, on the full T series, threshold 4.0.
- **Synthetic-spike protocol (labelled), z-score/STL-residual detector** (40 spikes ×6σ on the first
  5000 hourly points, 2000 bootstrap resamples, seed 0):
  - recall **0.775**, 95% CI **[0.639, 0.896]**
  - precision **0.886**, 95% CI **[0.762, 0.975]**
- **Paired significance (McNemar), Isolation Forest vs z-score**, same points:
  p = **1.08 × 10⁻¹⁹**, discordant = **64** (only-IForest-correct 0, only-z-correct 64) — the two make
  significantly different errors; per-point accuracy is imbalance-dominated (IForest over-flags at its
  contamination prior), so this confirms *difference*, not a winner.
- **Multi-seed benchmark (8 seeds), z-score/STL-residual detector** (`evaluation.benchmark`, real T
  series first 4000 pts, 40 spikes ×6σ): recall **0.87** (95% CI **0.73–0.97**), precision **0.92**
  (**0.83–1.00**), F1 **0.90** (**0.80–0.99**). CIs are percentile across seeds; wide because the
  labelled set is small — reported as indicative. `bootstrap_ci` also returns `n` (labelled size).
- Flagged residual anomalies: mean |residual| **8.09 °C** vs **1.00 °C** overall.
- Borderline near-misses (0.75×–1× threshold), 2014 T query: **24 anomaly windows + 81 borderline**.

## Generalisation (domain-agnostic core)
- The actual v1 modules (query engine, anomaly, interface) run unchanged on a non-weather dataset
  (simulated server metrics `cpu_load`, `mem_gb`): exact stats == pandas, anomaly detector catches an
  injected spike, interface answers "max cpu_load" exactly. `tests/test_generalization.py` (3 tests).
- Domain-specific layers = preprocessing (ingestion) + interface sensor vocabulary (aliases). The
  whole architecture also transfers to EU trade (v2 sibling; Appendix B).

## Circular sensor (wind direction) correctness
- Old linear "maximum wind direction" answer: **359.8°** (meaningless).
- Corrected: prevailing (circular mean) direction **196.6° (S)**, circular spread **71.3°**.
- Completed across the interface: trend + forecast on wd (deg) are declined with an explanation
  (linear slope/ARIMA invalid on a wrapping bearing); threshold count stays exact (e.g. wd > 200° =
  31,252 readings, 44.56%) but carries a wrap caveat.

## LLM faithfulness (trust guarantee)
- Across the verification questions, majority-of-3 vote: **0 invented numbers** (3/3 questions clean).
- Trust gate: any explanation introducing an unmatched number is discarded → exact template used.
  Now enforces decimals **and** integers > 12 (counts, years) — catches hallucinated counts ("87
  windows") and wrong years; clock times stripped, counts ≤ 12 tolerant (documented gap). Real LLM
  still 0 invented under the stricter check.

## LLM reproducibility (stochasticity mitigation)
- Generation now sends a fixed `seed` + low `temperature` to Ollama → **reproducible phrasing**:
  two seeded runs of the same prompt produced **identical** output (verified live). Numbers are
  guaranteed upstream (query engine) and checked downstream (faithfulness); a larger model drops in
  via the `model` param.

## Conversational follow-ups (live, Ollama llama3.2:3b)
- "show the anomalies in 2013" → 67 anomalies in 23 windows (+78 borderline).
- "what about 2014?" → rewritten to a 2014 anomaly question → 66 anomalies in 24 windows (+81 borderline).
- "what is the red dot?" → legend intent → anomaly explanation (not the fallback).
- Rewrite guards (live): "hottest in 2013?" → **not** rewritten (self-contained, resolved=None);
  "what about 2014?" → rewritten to a 2014 max question → answered exactly (33.68 °C). Unroutable
  rewrites are rejected (unit-tested), so a garbled rewrite can't silently take over.

## Interface robustness (keyword routing)
- Paraphrase coverage: a 24-case probe initially had **10 misses** (e.g. "how hot did it get?",
  "any outliers?" [plural], "how does X link to Y"). After widening the synonym lists, **23/23**
  route correctly; locked in by `test_intent_paraphrase_coverage`. Novel phrasing falls to the LLM
  planner. Still a defined intent set, not arbitrary language (documented).

## Causality boundary
- Causal questions ("does X cause Y?", "what causes…?") now answered honestly: correlation + explicit
  "correlation, not causation" caveat (paired), or an out-of-scope explanation (unpaired). No silent
  conflation of correlation with cause. `is_causal_query` + 2 tests.

## Clustering
- K-means (k=4) daily-feature clusters, temperature-ordered means ≈ **1.4 / 11.5 / 16.4 / 19.2 °C**
  (season-aligned).

## Forecasting
- ARIMA 14-day forecast: ordered bounds, future dates; backtest **MASE < 1.5**.
