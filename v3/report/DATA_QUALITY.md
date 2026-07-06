# v3 — Data Quality: Missingness & Class Imbalance (DAD)

An honest audit of the DAD dataset as loaded into the MESSAGES frame, and how each issue is handled.
Numbers are reproducible from `preprocess.load_week()` + `features`.

## 1. Missing data

| where | missing | note |
| --- | ---: | --- |
| Temperature on **PUBLISH** packets | **0 / 32,288** | every sensor reading parses cleanly — no missing measurements |
| `clientid` / `msgid` on all packets | 68.2% | expected: non-MQTT packets (TCP acks, handshakes) have no MQTT fields — *not* missing data |
| **Value grid** (16 sensors × 2,016 five-min windows) | **0.6%** | windows where a sensor's message was dropped or late (median 0% per sensor, max 2.3%) |

**How we handle it.** The dataset is essentially complete. The only genuine gaps are the **0.6%** of
value-grid cells where a sensor didn't report in a window. We **time-interpolate** these on the **value
channel** (`features.sensor_series(df, fill=True)`), so the temperature *residual* used to detect
*modification* isn't distorted by missingness.

**Critical distinction (documented so it isn't "fixed" by mistake):** we do **not** fill gaps on the
**flow channel**. There, *a missing message is the interception attack signal* — interpolating it away
would erase an anomaly. So value-channel gaps are filled; flow-channel gaps are preserved on purpose.

**Effect:** filling shifts the detection metrics **within the bootstrap confidence intervals** (e.g.
value-only AP 0.181 → 0.183, late-fusion 0.168 → 0.171) — i.e. negligibly. It is done as correct data
hygiene, not because it changes the conclusions. Current numbers: `report/MODELS.md`.

## 2. Class imbalance

The anomalies are **rare** — as real attacks are. This is inherent to the data and must be respected,
not "balanced away".

| level | normal | attack | attack rate | imbalance |
| --- | ---: | ---: | ---: | ---: |
| **packet** | 101,039 | 544 | 0.54% | **186 : 1** |
| **window** (5-min) | 1,926 | 90 | 4.46% | **21 : 1** |

Per attack **type** (window level) is also skewed:

| type | attack windows |
| --- | ---: |
| mixed | 60 |
| modification | 26 |
| **duplication** | **4** |

**How we handle it.**
- **We do not resample or synthetically rebalance.** Oversampling/SMOTE or undersampling would falsify a
  security dataset whose whole value is its *real* rarity — and would leak information across the
  episode-clustered attacks.
- **We use imbalance-appropriate metrics:** average precision (area under the PR curve, not accuracy),
  precision/recall/F1 at the best operating point, **point-adjusted** event-level recall, and **bootstrap
  confidence intervals** — never plain accuracy (which a trivial "always normal" classifier would score
  ~95–99% on).
- **We report per-type recall** so the tiny **duplication** stratum (4 windows) is visible and not hidden
  inside an aggregate — a single number would overstate confidence there.
- **Evaluation uses day-blocked CV** for the supervised combiner, so the imbalance-plus-clustering can't
  leak between train and test (it is *because* of this honesty that the supervised model is seen to
  overfit — see `MODELS.md`).

## 3. Summary
DAD is a clean, complete capture with a small (0.6%) value-grid gap that we interpolate on the value
channel only, and a strong, **genuine** class imbalance (186:1 packets, 21:1 windows) that we respect
and evaluate with imbalance-aware, per-type, CI-backed metrics rather than rebalancing.
