# v1 Evaluation Summary

Evaluation of the **core (computational) pipeline** on the Jena Climate dataset
(2009–2016, 70,129 hourly rows after preprocessing). RAG + LLM and the interface
are intentionally out of scope for v1. Full reproducible workings: `notebooks/02_evaluation.ipynb`.

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

## 2. Anomaly detection

| Method | Flagged | % of points |
| --- | --- | --- |
| Residual z-score (threshold 4σ) | 526 | ~0.75% |
| Isolation Forest (multi-sensor) | 702 | ~1.00% |
| Flagged by **both** | 33 | — |

Flagged residual-anomaly points have a **mean \|residual\| of 8.09 °C vs 1.00 °C overall** —
i.e. the detector selects genuinely extreme deviations, not noise. The two methods agree
only partially (33 in common), which is expected: the baseline targets temperature spikes,
while Isolation Forest reacts to unusual *multi-sensor* combinations.

## 3. Clustering (day-types)

K-means (k=4) on per-day feature vectors yields temperature-ordered clusters with
mean daily temperatures of **1.4, 11.5, 16.4, 19.2 °C** — a clean cold→warm gradient
that maps onto winter → spring/autumn → summer. Clusters are meaningful.

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

## Verdict

The computational core is **correct, sensible, and fully offline**. It is a sound foundation
for the explanation layer (RAG + LLM) and interface to build on next.
