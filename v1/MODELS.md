# Model Comparison

All anomaly-detection models used in v1, on the Jena temperature signal (70,129 hourly points).
Recall / F1 are measured on **100 injected synthetic anomalies** (see `src/evaluation.py`).
Reproduce with `notebooks/08_model_comparison.ipynb`.

| Model | Type | Flags | Train+score time | Recall* | F1* | Detects |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Z-score (on STL residual) | Statistical | 526 | ~4.0 s | 0.45 | 0.32 | large residual spikes |
| ARIMA forecast error | Forecasting | 411 | ~2.7 s | 1.00 | 0.34 | deviations from predicted dynamics |
| Isolation Forest | Classical ML | 699 | ~0.3 s | 1.00 | 0.25 | unusual feature combinations |
| CNN Autoencoder | Deep (reconstruction) | 702 | ~2.5 s | 1.00 | 0.25 | unusual window *shapes* |
| LSTM Autoencoder | Deep (reconstruction) | 702 | ~1.0 s | 1.00 | 0.25 | unusual *ordered* sequences |
| LSTM Forecaster | Deep (forecasting) | 702 | ~2.5 s | 1.00 | 0.25 | sudden *jumps* (one step late) |

\* On synthetic-injected anomalies. Apparent precision/F1 are held down because the real data also
contains genuine background anomalies that get flagged but count as "false positives" against the
synthetic-only labels — so F1 here understates real quality. On a clean controlled signal
(`notebooks/05`) precision is high.

## How to read this

- **Statistical / ARIMA** — cheap, interpretable, seasonality-aware (after STL decomposition).
  ARIMA adds an explicit *forecasting* view of anomalies.
- **Isolation Forest** — fastest; naturally **multi-sensor** (works on all columns at once).
- **Deep models** — capture unusual *shapes / sequences* and reach very high recall, at a small
  training cost (seconds on CPU). The CNN-AE is the simplest default; the LSTM variants and the
  forecaster are comparison baselines.

## Design note

Every model above only **detects / scores** anomalies. The exact numbers a user asks about
(max, min, correlation, counts, anomaly windows) always come from the deterministic
**query engine** (`src/query_engine.py`), never from a model — that separation is what keeps
answers trustworthy.
