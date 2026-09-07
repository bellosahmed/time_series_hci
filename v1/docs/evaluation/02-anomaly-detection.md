# 2. Anomaly detection — precision/recall/F1, with manufactured ground truth

## What's being evaluated

The anomaly detectors in `src/anomaly.py` (z-score, Isolation Forest, LOF, One-Class SVM, the
network-level "portfolio" combiner) and `src/deep.py` (CNN/LSTM autoencoders).

## The core problem this method has to solve first

**The real Jena Climate dataset has no ground-truth anomaly labels.** Nobody marked which of the
70,129 historical readings were "actually" anomalous, so precision and recall — which both
require knowing what's *actually* true — can't be computed directly on the real data. Before any
detector can be scored, evaluation has to manufacture ground truth two different, complementary
ways:

- **Synthetic (exact, artificial).** `evaluation.inject_spikes()` takes a clean signal and
  deliberately injects a known number of spikes at known positions. Now precision/recall are
  computable, because the labels are exact — but they're not real anomalies, they're manufactured
  ones.
- **Real (approximate, genuine).** `evaluation.KNOWN_EVENTS` — five documented real weather
  events within the record's span (e.g. "2015 July heatwave," 1–6 July), hand-picked from
  historical record and cross-checked against the data's own temperature extremes. These are
  real, but the window boundaries are approximate, not point-exact.

Reporting both is the honest move: synthetic labels are exact but artificial; real-event labels
are genuine but approximate. Neither alone tells the whole story.

## Method

**Standard classification metrics** on the synthetic-spike labels:

```python
tp = (pred & truth).sum(); fp = (pred & ~truth).sum(); fn = (~pred & truth).sum()
precision = tp / (tp + fp); recall = tp / (tp + fn); f1 = 2PR / (P + R)
```

**Point-adjustment.** A raw point-by-point score is *unfair* to time-series anomaly detectors: if
a true anomaly lasts 6 hours and a detector flags hour 3 of it, a strict point score credits 1
true positive and penalizes 5 false negatives — even though a human would say "yes, it caught
that event." `evaluation.point_adjust()` applies the standard correction: if a detector flags
*any* point inside a true anomaly segment, the whole segment counts as caught.

**Multi-seed benchmarking with confidence intervals.** A single injection run could get lucky or
unlucky. `evaluation.benchmark()` repeats the inject-and-score cycle across multiple random
seeds and reports mean ± std plus a 95% percentile interval — turning "recall = 0.87" into
"recall = 0.87, 95% CI [0.80, 0.93]," a number that states its own uncertainty.

**Bootstrap confidence intervals.** `evaluation.bootstrap_ci()` resamples the labelled points
with replacement 300+ times and recomputes the metric each time, giving a second, independent
way to state uncertainty on a single labelled run (as opposed to the multi-seed approach, which
varies the *injection*, not the *sample*).

**Paired significance testing (McNemar).** When comparing two detectors head-to-head (e.g. "is
Isolation Forest really better than z-score, or is the difference just noise?"),
`evaluation.mcnemar_test()` looks only at the points where they *disagree* and runs an exact
binomial test on whether that disagreement is skewed toward one detector — the standard paired
test for this exact question, rather than comparing two accuracy numbers and eyeballing the gap.

## Real evidence (reproduced 15 July 2026, `df['T (degC)']`, real Jena data)

**Multi-seed benchmark** (z-score-on-STL-residual, threshold 4.0σ, 4 seeds, 30 injected spikes
each, magnitude 6.0):

```bash
python3.12 -c "
from src import preprocess, evaluation, anomaly, analysis
df = preprocess.align(preprocess.clean(preprocess.load_data('data/jena_climate_2009_2016.csv')), freq='1h')
det = lambda noisy: anomaly.zscore_anomalies(analysis.decompose(noisy, period=24)['resid'].dropna(), threshold=4.0)
print(evaluation.benchmark(df['T (degC)'].iloc[:4000], det, seeds=range(4), n=30, magnitude=6.0))
"
```

| Metric | Strict | Point-adjusted |
| --- | --- | --- |
| Precision | 0.871 ± 0.089 (95% CI [0.758, 0.990]) | 0.871 ± 0.089 |
| Recall | 0.867 ± 0.053 (95% CI [0.803, 0.931]) | 0.867 ± 0.053 |
| F1 | 0.868 ± 0.069 (95% CI [0.780, 0.959]) | 0.868 ± 0.069 |

(Strict and point-adjusted come out identical here because the injected spikes are single-point,
not multi-hour segments — point-adjustment only changes the score when a true anomaly spans
several consecutive points.)

**Bootstrap recall CI** (300 resamples of one labelled 4,000-point run):
`recall = 0.933, 95% CI [0.836, 1.000]`.

**McNemar test, Isolation Forest vs. z-score** on the same labelled points:
`52 discordant points, p = 4.4×10⁻¹⁶` — a vanishingly small p-value, meaning the two detectors'
disagreement is **not noise**; on this run z-score was correct on the discordant points every
single time Isolation Forest wasn't (`only_a_correct=0, only_b_correct=52`), a real, statistically
significant difference in behaviour between the two methods on this labelled sample.

**Known-event recall** (real weather events, z-score detector on the full real record):
`4 of 5 known events caught (recall 0.8)` — missed: the 2015 July heatwave. This is reported
honestly rather than only citing the synthetic numbers, which look better: a detector tuned on
synthetic point-spikes doesn't automatically generalise perfectly to real, gradual weather
events.

## Honest caveats

- Because there's no real ground truth, the synthetic-spike precision/recall numbers above should
  be read as "how well this detector recovers *known, artificial* anomalies," not "how accurate
  this detector is on real Jena Climate anomalies" — those are different claims, and only the
  known-event recall (4/5, above) speaks to the second one, and even that is corroboration on 5
  hand-picked events, not a full audit.
- The z-score detector missing the 2015 July heatwave specifically is worth investigating further
  if this becomes a headline claim in the dissertation — it's the kind of honest negative result
  worth a sentence of discussion rather than burying.

## Network-level comparison (all 14 sensors, 10 methods, injected cross-sectional faults)

Everything above is the single-sensor comparison. The dissertation's Chapter 4 §4.4 also reports a
second, separate comparison run over the full 14-sensor vector rather than one sensor's history, on
the grounds (Chapter 3 §3.6) that a real network fault does not necessarily make any one sensor
individually extreme. That comparison's source, traced here for the first time in writing: 100
injected faults per fault type (spike, drift, stuck-sensor, global-freeze), scored by average
precision, computed in `notebooks/05_deep_training_techniques.ipynb`, cell 48 (`ap_all_methods`),
with the stuck-sensor "rematch" including a dedicated flatline/variance-collapse detector added in
cell 56.

**Real evidence (`notebooks/05_deep_training_techniques.ipynb`, cell 48 output)**:

| Method | Family | Spike (4σ) | Drift (4σ, 24h) | Stuck (24h) |
| --- | --- | --- | --- | --- |
| Mahalanobis | classical | 0.931 | 0.880 | 0.642 |
| LOF | classical | 0.894 | 0.702 | 0.112 |
| Transformer | deep | 0.835 | 0.853 | 0.346 |
| PCA | classical | 0.544 | 0.539 | 0.087 |
| OC-SVM | classical | 0.507 | 0.419 | 0.052 |
| CNN | deep | 0.506 | 0.373 | 0.048 |
| GRU | deep | 0.151 | 0.211 | 0.043 |
| LSTM | deep | 0.093 | 0.248 | 0.044 |
| RNN | deep | 0.054 | 0.334 | 0.056 |
| IsoForest | classical | 0.003 | 0.048 | 0.039 |

On the stuck-sensor fault specifically, a dedicated flatline/variance-collapse detector (cell 56,
not in the table above since it isn't a general-purpose method) reaches 0.791, ahead of every method
in the table for that one fault type — the evidence behind the portfolio design in `anomaly.py`
(§3.6 of the dissertation), which unions the Mahalanobis and flatline detectors rather than shipping
whichever single method scores highest on average.

**Honest caveat on provenance, matching Chapter 3 §3.10's own disclosure**: `Mahalanobis`, `LOF`,
`OC-SVM`, and `IsoForest` are the same classical detectors implemented in `src/anomaly.py` and
covered by the pytest suite; `PCA`, `CNN`, `GRU`, `LSTM`, `RNN`, and `Transformer` were trained and
evaluated in this notebook specifically for the comparison and are not wired into `src/deep.py` or
exercised by any test — the figures are real (reproducible by re-running the notebook cells above)
but, unlike the single-sensor comparison earlier in this document, not re-checkable from the
automated test suite alone.

**Partial mitigation**: the two tables above are persisted as durable CSV artifacts
(`docs/evaluation/results/network_level_ap_by_fault_type.csv` and
`network_level_ap_rematch_with_flatline.csv`) rather than living only in the notebook's cell
outputs, and `python3.12 verify_network_level_results.py` (repo root) checks those CSVs against
every figure this dissertation cites — 12/12 checks passing as of 25 August 2026, including a
real inconsistency this check caught and fixed: Chapter 4 §4.4 originally described the CNN and
One-Class SVM detectors as "around 0.51 to 0.54," but their actual values (0.506, 0.507) round to
0.51 only — the 0.54 figure belonged to PCA, a method not part of this comparison. That script
does not retrain the models or re-run this notebook; it guards specifically against the persisted
numbers drifting out of sync with the dissertation, which is the more likely failure mode.
