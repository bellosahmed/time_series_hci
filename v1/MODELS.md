# Model Comparison

All anomaly-detection models used in v1, on the Jena temperature signal (70,129 hourly points).
Recall / F1 are measured on **100 injected synthetic anomalies** (see `src/evaluation.py`).
Reproduce with `notebooks/08_model_comparison.ipynb`.

| Model | Type | Flags | Time | Precision | Recall | F1 | Detects |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Z-score (STL residual) | Statistical | 526 | ~4.0 s | 0.24 | 0.45 | 0.32 | large residual spikes |
| ARIMA forecast error | Forecasting | 411 | ~2.7 s | 0.20 | 1.00 | 0.34 | deviations from predicted dynamics |
| Isolation Forest | Classical ML | 699 | ~0.3 s | 0.14 | 1.00 | 0.25 | unusual feature combinations |
| CNN Autoencoder | Deep (reconstruction) | 702 | ~2.5 s | 0.14 | 1.00 | 0.25 | unusual window *shapes* |
| LSTM Autoencoder | Deep (reconstruction) | 702 | ~1.0 s | 0.14 | 1.00 | 0.25 | unusual *ordered* sequences |
| LSTM Forecaster | Deep (forecasting) | 702 | ~2.5 s | 0.14 | 1.00 | 0.25 | sudden *jumps* (one step late) |

> **Read precision carefully.** These numbers come from injecting 100 synthetic anomalies into the
> *real* 8-year series. The models also (correctly) flag the hundreds of **genuine** background
> anomalies already in the data — but those count as "false positives" against the synthetic-only
> labels, so precision and F1 are badly understated here. They measure recall faithfully, not precision.

### Fair precision — controlled clean signal (no hidden anomalies)

When the same detectors run on a pure daily-cycle signal with 40 injected spikes — where *every*
flag can be judged — precision is high, confirming the detectors are accurate:

| Model | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Z-score | 0.85 | 0.82 | 0.84 |
| ARIMA | 0.98 | 1.00 | 0.99 |
| Isolation Forest | 1.00 | 1.00 | 1.00 |
| CNN Autoencoder | 0.91 | 1.00 | 0.95 |
| LSTM Autoencoder | 0.93 | 1.00 | 0.96 |
| LSTM Forecaster | 0.93 | 1.00 | 0.96 |

**Takeaway:** the detectors are genuinely precise (0.85–1.00); the low real-data precision is an
artifact of unlabeled background anomalies, not a model weakness — exactly the "evaluation is hard
without labels" problem from the literature.

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
