# v3 — Detector & Fusion Comparison (DAD, real labels)

Two detector channels and their fusion, graded on DAD's **real** attack labels over 2016 five-minute
windows (90 attack windows: mixed 60 / modification 26 / duplication 4). Reproduce with
`experiment.run_experiment(preprocess.load_week())` (deterministic, `random_state=0`).

## Results

*(Value channel uses gap-filled temperatures — see `DATA_QUALITY.md`; the fill shifts numbers within the
bootstrap CIs, i.e. negligibly.)*

| config | AP | AP 95% CI | best-F1 | precision | recall | **F1 (point-adj.)** | **recall (point-adj.)** |
| --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| flow-only    | 0.136 | [0.077, 0.222] | 0.226 | 0.412 | 0.156 | 0.294 | 0.178 |
| value-only   | 0.183 | [0.136, 0.245] | 0.379 | 0.331 | 0.444 | **0.804** | 0.889 |
| early-fusion | 0.147 | [0.108, 0.206] | 0.289 | 0.202 | 0.511 | 0.732 | **1.000** |
| late-fusion  | 0.171 | [0.123, 0.246] | 0.306 | 0.214 | 0.533 | 0.789 | **1.000** |
| supervised (LR, day-blocked) | 0.101 | [0.075, 0.141] | 0.212 | 0.147 | 0.378 | 0.554 | 1.000 |

Per-attack-type recall (window-level, at the best-F1 threshold):

| config | modification | duplication | mixed |
| --- | ---: | ---: | ---: |
| flow-only  | **0.00** | **1.00** | 0.17 |
| value-only | 0.46 | 0.50 | 0.43 |
| early-fusion | 0.42 | 1.00 | 0.52 |
| late-fusion | 0.46 | 1.00 | 0.53 |
| supervised | 0.42 | 1.00 | 0.32 |

## What the numbers say

1. **The two channels are genuinely complementary.** Flow-only has **0.00** recall on *modification*
   (a bad payload leaves timing normal) but **1.00** on *duplication*; the value channel has the
   opposite tilt. Neither single channel can cover all three attacks — a *coverage* limit, not a tuning
   one.

2. **Fusion buys recall at a precision cost** (window-level recall 0.44 → 0.53; precision 0.33 → 0.21):
   an OR-style union of two detectors sums their false positives.

3. **Window-level vs event-level tells different stories — report both.**
   - On *window-level* aggregate (AP, best-F1), **value-only leads** (AP 0.183, F1 0.379).
   - But the **AP confidence intervals overlap heavily** (value [0.136, 0.245] vs late-fusion
     [0.123, 0.246]), so value-only's edge is **not statistically significant** — we should not claim
     one is better on AP.
   - On *event-level* (point-adjusted), **both fusions reach 1.00 recall** (every attack episode is
     caught) and late-fusion's F1 (0.789) is within **0.015** of value-only (0.804).

4. **The *supervised* combiner is the worst — an important negative result.** A logistic regression
   over the two channel scores, evaluated with **day-blocked** leave-one-day-out CV, scores the *lowest*
   AP (0.101). Why: there are only four attack days and **attack type is confounded with day**
   (Wed=modification, Thu=duplication, Fri/Sat=mixed), so when a day is held out, the model trains on
   days whose attacks need a *different* channel than the test day — the learned channel-weighting does
   not transfer, and it overfits. A random-split CV would have leaked (same episode in train and test)
   and looked great; the honest day-blocked split exposes the failure. **Lesson: with scarce,
   episode-clustered labels, a learned fusion generalises *worse* than a simple unsupervised OR** — more
   sophistication is not more skill here.

5. **Verdict (honest).** No configuration is significantly best on AP. If the operational goal is
   *catching every attack episode*, **rule-based fusion is preferred** (point-adjusted recall = 1.00).
   If it is *window-level precision*, the single **value** channel is competitive and simpler. Fusion is
   worth the extra machinery only when both channels are individually strong; here the flow channel is
   weak (AP 0.136), which is why fusion helps event-recall more than window-F1 — and why the *supervised*
   combiner, needing to generalise across day-blocked folds, does worst of all.

## Honesty notes

- **No attacker-identity leakage:** the attacker IP appears only during attacks, so it is deliberately
  excluded from the flow features — the numbers are lower but fair.
- **Untuned detectors** (default Isolation Forest, plain z-score) and **weak window labels** (a window
  is an attack if any labelled packet falls in it); 71% of attacks are only typeable as "mixed".
- **Future:** contamination/window sweeps, a seasonal value model, and a supervised day-blocked
  score-combiner — logged in `EVALUATION.md`.
