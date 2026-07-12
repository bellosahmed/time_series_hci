# v2 Results Verification

Every headline result is **independently re-derived from the raw data and checked**. Reproduce with:

```bash
cd code/v2 && python3.12 verify_results.py      # -> 13/13 checks passed
```

This is separate from the 79 unit tests (`pytest tests/`): the unit tests check the *code*; this
script checks the *claims*.

## Results (13/13 PASS)

| # | Claim verified | Evidence |
| --- | --- | --- |
| 1 | Data integrity | 292×54, 0 missing / 0 gaps / 0 non-positive |
| 2 | **Truth layer** = pandas | query engine max/mean/correlation/balance match exactly |
| 3 | Stationarity diagnosis | ADF: log level non-stationary → seasonal-diff stationary |
| 4 | Detectors catch real crises | all 3 families flag **2008 & COVID** (event-level) |
| 5 | Confusion matrix valid | TP+FP+FN+TN = 292 (5+3+9+275) |
| 6 | **Forecast predicts, not memorises** | 24-month unseen backtest **MAPE 4.0 %** |
| 7 | Shocks detected not predicted | COVID actual breaches the 95 % band |
| 8 | Deep model honest result | AE F1 0.21 ≤ Isolation Forest 0.36 |
| 9 | Clustering valid | 27 countries → 4 groups |
| 10 | Multi-country reasoning | 2020-04 classified "shared shock" |
| 11 | **Trust boundary** | RAG knowledge base contains no euro amounts |
| 12 | BERT > lexical on paraphrase | "factories shut…" → BERT finds COVID, BM25 doesn't |
| 13 | **LLM never alters numbers** | phrasing keeps magnitude 35 and year 2020 exact |

## Why this matters

The project's central claim is **trust**: numbers are computed by code, verified, and only *phrased*
by the LLM. Checks 2, 5, 11 and 13 verify that chain end-to-end — the query engine equals ground
truth, the confusion arithmetic is sound, the retriever holds no numbers, and the language model
preserves the computed figure exactly. Checks 4, 6, 8 confirm the analytical results (real-crisis
detection, out-of-sample forecasting, the honest deep-vs-classical comparison).

The five layers of detector verification (unit tests · visual · ground-truth scoring · cross-method
agreement · historical corroboration) are detailed in notebook 04; this document is the consolidated,
one-command check of the whole pipeline.
