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
On a controlled clean signal (see `notebooks/05`), precision is high too.

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
