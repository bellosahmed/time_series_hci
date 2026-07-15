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
