# Deep Learning Models — Side Note (v2)

> **Status: tried.** The dense/multivariate Autoencoder was already built (`deep_anomaly.py`); the
> three "Optional" temporal candidates below (LSTM-AE, CNN-AE, TCN) have now also been built and
> evaluated (`src/deep_temporal.py`), test-first, against the real 2008/COVID labels. Results below the
> candidate table. Kept as permanent, tested modules — not a one-off notebook cell — even though the
> honest result is that none of them beat the classical Isolation Forest baseline decisively.

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
| **Dense / multivariate Autoencoder** | AE | reconstruct the 54-country vector each month; large error = systemic anomaly | ✅ uses the multivariate width; small & robust | **Built — F1 0.21** (loses to Isolation Forest 0.36) |
| **1D CNN Autoencoder** (multivariate, windowed over time) | AE | slide 6-month windows, learn local temporal patterns, reconstruction error | ✅ small, fast; carried over from v1 | **Built — mean F1 0.29 ± 0.09** (5 seeds; close to, not significantly above, the dense AE) |
| **LSTM Autoencoder** (multivariate, windowed over time) | RNN-AE | encode/decode 6-month sequences; captures temporal order the point-in-time AE can't see | ⚠️ OK, but 292 pts is thin for RNNs | **Built — mean F1 0.43 ± 0.12** (5 seeds; best of the four deep configs, but McNemar vs Isolation Forest is *not* significant, p=0.29 — see Result below) |
| **LSTM / GRU Forecaster** | RNN | predict next month; error = anomaly | ⚠️ overfits easily on a short series | Not built — subsumed by the LSTM-AE result above; a forecaster needs even more sequential context than an AE, on the same thin 292 points |
| **TCN** (Temporal Conv Net) | CNN | dilated causal convolutions, efficient long memory | ⚠️ nice middle ground, more to tune | **Built — mean F1 0.13 ± 0.05** (5 seeds; clearly the worst of the four — the most data-hungry architecture on the smallest dataset) |
| **VAE** (Variational AE) | prob. AE | reconstruction *probability* → anomaly likelihood | ⚠️ needs more data to calibrate | Stretch |
| **Transformer / Anomaly Transformer** | attention | self-attention over the sequence | ❌ very data-hungry → will overfit | Literature only |
| **GNN** (countries as a graph) | graph | model country-to-country trade links explicitly | ❌ complex, but a strong **novelty** idea | Literature / future |
| **DeepAR / N-BEATS / TFT** | deep forecasters | probabilistic multi-series forecasting | ❌ designed for many long series | Literature only |

## Result (real 2008/COVID labels, `systemic_features`, 6-month windows for the temporal models)

| Detector | F1 | vs Isolation Forest (0.36) |
| --- | --- | --- |
| Dense AE (point-in-time, no temporal context) | 0.21 | loses |
| TCN-AE | 0.13 ± 0.05 (5 seeds) | loses, worst of the four |
| CNN-AE (windowed) | 0.29 ± 0.09 (5 seeds) | roughly ties |
| LSTM-AE (windowed) | 0.43 ± 0.12 (5 seeds) | **numerically ahead**, but not significant |

**The headline finding is not "LSTM-AE wins" — it's what happened when we checked whether it really
does.** A single seed (0.57) looked like a clear win over Isolation Forest (0.36). Averaging over 5
seeds dropped that to 0.43 ± 0.12, with individual seeds ranging 0.29–0.57 — a spread wide enough to
overlap the baseline. A formal **McNemar test** (best seed vs Isolation Forest) returned **p = 0.29**:
not significant. So the honest conclusion is *"adding temporal context (LSTM-AE, CNN-AE) numerically
helps over the point-in-time dense AE, but none of the four deep configs is significantly better or
worse than the simple classical Isolation Forest baseline on 292 points."* This is exactly the kind
of overclaim this project's own statistical-rigor tools (seed-averaging, bootstrap CI, McNemar) exist
to catch — a single favourable seed would have made a much stronger, wrong claim.

**Why TCN did worst, specifically:** it's the most parameter-dense architecture of the three temporal
models relative to its receptive field (three dilated blocks per window of only 6 months), and it had
the fewest effective training sequences (six-month non-overlapping windows over 292 points leaves only
~46 training examples). This is consistent with, not contrary to, the "more data needed" reading — the
architecture that assumes the most temporal context available is the one starved hardest by having
the least.

**Decision:** kept all three (`deep_temporal.py`) as tested, permanent modules — not deleted — because
the *comparison itself* (temporal vs point-in-time, and the seed-sensitivity finding) is the
reportable result, independent of any one architecture "winning."

## How each architecture works, and why each result happened (methodology + mechanism)

*Written for direct use in the Methods/Discussion sections of the write-up — this is the "how and why,"
not just the numbers above.*

**Shared training procedure (Methods).** All three temporal detectors follow the same recipe, ported
from v1's `deep.py`: consecutive months are windowed into length-6 sequences. Training uses
**non-overlapping** windows (6, 12, 18, … — this avoids near-duplicate windows dominating an already
tiny training set); scoring uses **overlapping** windows (every possible 6-month span), and each
month's final score is the *average* of its reconstruction error across every window that covers it —
the same localisation trick v1 uses, and the reason a single anomalous month is not blurred into an
undifferentiated 6-month block. All models are seeded (`torch.manual_seed`, set **before** model
construction, since weight initialisation happens at construction time) and trained with Adam + MSE
reconstruction loss. Anomalies are flagged at the 95th-percentile score, matching Isolation Forest's
`contamination=0.05` operating point so every config is compared at a like-for-like threshold.

**Architecture 1 — LSTM Autoencoder.** A recurrent encoder reads the 6-month × 54-country window and
compresses it to a 16-dimensional latent vector; a recurrent decoder expands that vector back into a
reconstructed 6-month window. Reconstruction error is the anomaly score. Unlike the point-in-time
dense `AE`, this model can in principle recognise a *shape* — a slow six-month slide into a crisis —
not just a single unusual month in isolation.

**Architecture 2 — 1D CNN Autoencoder.** Two convolutional layers slide a width-3 kernel along the time
axis, so each output position has a receptive field of roughly its two nearest neighbouring months
before the representation is flattened and compressed to a 16-dim latent code and reconstructed
symmetrically. Far fewer effective parameters than the LSTM, and no recurrence, so it is much less
able to memorise a small training set — a structural bias toward *not* overfitting.

**Architecture 3 — TCN Autoencoder (new to the project).** Three dilated causal convolution blocks
(dilations 1, 2, 4) give the widest nominal receptive field of the three architectures, but each block
is strictly **causal** — it may only look backward within the window, never forward — whereas the LSTM
and CNN autoencoders both implicitly see the *entire* window before compressing it. This is the
architecture family named in v1's literature review (Bai, Kolter & Koltun, 2018) but never previously
implemented in the project.

**Why the LSTM-AE's apparent win didn't survive scrutiny — the mechanism, not just the statistic.**
Windowing 292 months into non-overlapping 6-month blocks leaves only **≈46 training sequences** — that
is the real effective sample size the network learns from, not 292. LSTM weight initialisation is
random (seeded per run, but a *different* seed is a genuinely different starting point in parameter
space), and with only ~46 sequences there is not enough data to force every random start toward the
same solution — different seeds converge to different local optima. Because the anomaly threshold is
the 95th percentile of *that run's* error distribution, an unstable model produces an unstable
threshold, which changes *which specific months* cross it from run to run. On the most favourable seed,
the months that happened to cross the threshold aligned well with the 2008/COVID labels (F1 = 0.57); on
the least favourable, a different set crossed it and aligned worse (F1 = 0.29). This is a textbook
symptom of **too little data relative to model capacity**, visible directly in seed-to-seed instability
rather than only in an aggregate under-performance number.

**Why the McNemar test found "not significant" despite a 0.07-F1 gap over the baseline.** McNemar only
counts the months where the two detectors *disagree* — here, 22 out of 292. Of those 22, LSTM-AE was
uniquely correct on 14 and Isolation Forest on 8. Judging whether a process is "biased" from only 22
observations is statistically weak — it is equivalent to flipping a coin 22 times, getting 14 heads,
and asking whether the coin is unfair: a 14/22 split is unremarkable for a fair coin, so the test
correctly declines to call the difference significant, even though the *aggregate* F1 numbers (0.43 vs
0.36) look like a clean win. This is precisely why the project reports point estimate **and** interval
**and** a paired significance test together — any one of the three alone would have told a different,
less complete story.

## Recommendation (now revised in light of the result above)

- **Report the negative result, not a chosen "winner."** Isolation Forest remains the v2 production
  default (fastest, simplest, and statistically indistinguishable from the best deep alternative).
- **Mention in the report, don't build:** Transformer, GNN, DeepAR/N-BEATS/TFT — "more powerful but
  unsuitable for a 292-point series; future work." (Stating this honestly earns marks.)
- **Always compare against** the classical detectors (Isolation Forest, SARIMA, z-score) — and, as
  this result shows, always check significance before trusting a single seed's point estimate.

## Practical notes

- Torch + the **RTX 5070** (cu128 build) are ready from v1; v2 deep models would reuse the
  device-aware pattern (`get_device()` → cuda/cpu).
- Keep deep models **small** (few layers, strong regularisation, early stopping) given the data size.
- Evaluate the same way as the classical detectors: against the **real 2008 / COVID** labels, plus
  cross-method agreement (see notebook 04's verification).

_Side note — revisit after the core pipeline (notebooks 06–09) is complete._
