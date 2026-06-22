# Deep Learning Models — Side Note (v2)

> **Status: reference only — not yet implemented.** This records the deep-learning options we
> considered for v2 (anomaly detection + forecasting) and *why* we'd pick some over others, before
> writing any of them. We'll "list-then-pick" again when we actually build the deep notebook.

## The constraint that drives every choice

The v2 dataset is **292 monthly points × 54 channels**. That is a **short** series (deep learning is
data-hungry) but **wide** (many countries → a genuine multivariate signal). So:

- Models that exploit the **width** (cross-country structure) are favoured.
- Models that need **long sequences** or huge data (Transformers, deep forecasters) will **overfit**.
- Classical baselines (Isolation Forest, SARIMA) are strong here and **must be beaten** to justify
  any deep model — that comparison is itself a result worth reporting.

## Candidate models

| Model | Type | Idea | Fit for 292 pts × 54 ch | Verdict |
| --- | --- | --- | --- | --- |
| **Dense / multivariate Autoencoder** | AE | reconstruct the 54-country vector each month; large error = systemic anomaly | ✅ uses the multivariate width; small & robust | **Use — best v2 fit** |
| **1D CNN Autoencoder** | AE | slide windows over time, learn local patterns, reconstruction error | ✅ small, fast; carried over from v1 | **Use — baseline** |
| **LSTM Autoencoder** | RNN-AE | encode/decode sequences; captures temporal memory | ⚠️ OK, but 292 pts is thin for RNNs | Optional |
| **LSTM / GRU Forecaster** | RNN | predict next month; error = anomaly | ⚠️ overfits easily on a short series | Optional |
| **TCN** (Temporal Conv Net) | CNN | dilated causal convolutions, efficient long memory | ⚠️ nice middle ground, more to tune | Optional |
| **VAE** (Variational AE) | prob. AE | reconstruction *probability* → anomaly likelihood | ⚠️ needs more data to calibrate | Stretch |
| **Transformer / Anomaly Transformer** | attention | self-attention over the sequence | ❌ very data-hungry → will overfit | Literature only |
| **GNN** (countries as a graph) | graph | model country-to-country trade links explicitly | ❌ complex, but a strong **novelty** idea | Literature / future |
| **DeepAR / N-BEATS / TFT** | deep forecasters | probabilistic multi-series forecasting | ❌ designed for many long series | Literature only |

## Recommendation when we reach the deep step

- **Build:** **multivariate Autoencoder** (the natural v2 model — scores *systemic* anomalies across
  all 54 channels, matching the multi-country angle) + **1D CNN-AE** as a v1-comparable baseline.
- **Mention in the report, don't build:** Transformer, GNN, DeepAR/N-BEATS/TFT — "more powerful but
  unsuitable for a 292-point series; future work." (Stating this honestly earns marks.)
- **Always compare against** the classical detectors (Isolation Forest, SARIMA, z-score). If a deep
  model can't beat them here, that *negative* result is a legitimate, reportable finding.

## Practical notes

- Torch + the **RTX 5070** (cu128 build) are ready from v1; v2 deep models would reuse the
  device-aware pattern (`get_device()` → cuda/cpu).
- Keep deep models **small** (few layers, strong regularisation, early stopping) given the data size.
- Evaluate the same way as the classical detectors: against the **real 2008 / COVID** labels, plus
  cross-method agreement (see notebook 04's verification).

_Side note — revisit after the core pipeline (notebooks 06–09) is complete._
