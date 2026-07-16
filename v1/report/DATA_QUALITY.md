# Data Quality & Multi-Sensor Verification (v1)

_Generated 3 July 2026 from `src/` functions on the Jena Climate hourly series. All numbers
are computed by code (`analysis.py`, `preprocess.py`, `evaluation.py`); this document only
describes them._

## 1. Data availability

```
                 n_total  n_missing  n_sentinel  n_real  available_pct
p (mbar)          420551          0           0  420551        100.000
T (degC)          420551          0           0  420551        100.000
Tpot (K)          420551          0           0  420551        100.000
Tdew (degC)       420551          0           0  420551        100.000
rh (%)            420551          0           0  420551        100.000
VPmax (mbar)      420551          0           0  420551        100.000
VPact (mbar)      420551          0           0  420551        100.000
VPdef (mbar)      420551          0           0  420551        100.000
sh (g/kg)         420551          0           0  420551        100.000
H2OC (mmol/mol)   420551          0           0  420551        100.000
rho (g/m**3)      420551          0           0  420551        100.000
wv (m/s)          420551          0          18  420533         99.996
max. wv (m/s)     420551          0          20  420531         99.995
wd (deg)          420551          0           0  420551        100.000
```

Every sensor is between 99.995% and 100.000% real (raw, 10-minute resolution, 420,551 rows).
The two wind columns, `wv (m/s)` and `max. wv (m/s)`, are the only ones carrying the `-9999`
sentinel error code — 18 and 20 occurrences respectively (`n_sentinel`) — which
`preprocess.clean` converts to `NaN` before interpolation. No column has any genuine
`n_missing` (gaps in the raw timestamp index); the only "missingness" in this dataset comes
from the sentinel errors.

## 2. Distribution & skewness (shape of the graph)

```
                     mean    median     std   skew  kurtosis       min       max
p (mbar)          989.224   989.580   8.360 -0.403     0.878   934.905  1015.243
T (degC)            9.445     9.423   8.410 -0.021    -0.198   -22.653    37.038
Tpot (K)          283.486   283.463   8.491 -0.045    -0.136   250.970   310.977
Tdew (degC)         4.957     5.230   6.724 -0.380    -0.016   -24.602    23.017
rh (%)             76.035    79.283  16.379 -0.675    -0.362    13.683   100.000
VPmax (mbar)       13.569    11.830   7.720  1.309     2.393     0.980    62.943
VPact (mbar)        9.534     8.872   4.179  0.553    -0.254     0.822    28.168
VPdef (mbar)        4.035     2.197   4.872  2.370     7.394     0.000    45.195
sh (g/kg)           6.022     5.600   2.653  0.565    -0.228     0.517    18.025
H2OC (mmol/mol)     9.640     8.973   4.230  0.558    -0.240     0.828    28.662
rho (g/m**3)     1216.098  1213.863  39.918  0.314     0.141  1084.358  1392.175
wv (m/s)            2.129     1.767   1.468  1.541     3.734     0.000    23.201
max. wv (m/s)       3.530     3.007   2.227  1.283     2.121     0.000    20.015
wd (deg)          174.827   192.367  73.241 -0.725    -0.353     0.000   359.800
```

Temperature (`T (degC)`, skew −0.021) and potential temperature (`Tpot (K)`, skew −0.045) are
essentially symmetric — the mean and median almost coincide. The most skewed sensor in the
dataset is **`VPdef (mbar)` (vapour pressure deficit), skew = 2.370**, with the highest
kurtosis of any sensor (7.394) — a long right tail with occasional extreme spikes (VPdef is
naturally floored at 0, so it cannot go negative but can spike high on hot/dry days). The two
wind columns are also strongly right-skewed with heavy tails (`wv (m/s)`: skew 1.541,
kurtosis 3.734; `max. wv (m/s)`: skew 1.283, kurtosis 2.121), consistent with wind speed being
non-negative and occasionally gusting far above its typical value. `VPmax (mbar)` is similarly
right-skewed (1.309). `kurtosis` flags these heavy-tailed sensors (VPdef, the wind speeds,
VPmax) as the ones most prone to producing z-score outliers.

## 3. How missing data is handled (average vs distribution vs trend)

The pipeline uses **time-aware interpolation** (`preprocess.clean`), which follows the local
trend of the graph, then forward/back-fills any leading/trailing gaps. It does NOT replace
gaps with the column mean. Justification (held-out points, MAE):

```
                  mae
interpolation  0.2629
mean_fill      6.9019
```

Interpolation's held-out MAE (0.2629 °C) is roughly **26x lower** than mean-fill's (6.9019 °C)
on `T (degC)`, because it tracks the local trend of the series rather than flattening every gap
to the same global average.

## 4. Standard deviation → outlier threshold

`anomaly.zscore_anomalies` standardises each point as z = (x − mean) / std (population std,
`ddof=0`) and flags |z| > threshold. Std is the unit of "how far is far"; the threshold sweep
(`anomaly.threshold_sensitivity`) shows counts collapsing as the threshold rises, and the knee
is where only real events survive.

(Note: the detector uses population std `ddof=0` here, whereas the descriptive `std` column in
Section 2 is pandas' default sample std `ddof=1`. The two differ only by a factor of √(n/(n−1)),
which is negligible at n ≈ 70,000, but they are computed for different purposes.)

## 5. Class imbalance

```
{'n_anomaly': 157, 'n_normal': 69972, 'n_total': 70129, 'anomaly_pct': 0.2238731480557259}
```

At the default threshold (z > 3.0) on `T (degC)`, only 157 of 70,129 hourly points are flagged
— **anomaly_pct ≈ 0.22%**. Anomalies are a tiny minority of the data, so raw accuracy would be
misleading (a detector that never fires would already score >99.7% accuracy); precision/recall
and the point-adjusted metrics (`evaluation.py`) are the honest scores.

## 6. Multi-sensor verification (humidity and all other sensors)

```
                  skew  kurtosis  zscore_anomalies  top_corr_sensor  top_corr_r
p (mbar)        -0.403     0.878               449     rho (g/m**3)       0.308
T (degC)        -0.021    -0.198               157         Tpot (K)       0.997
Tpot (K)        -0.045    -0.136               182         T (degC)       0.997
Tdew (degC)     -0.380    -0.016               344     VPact (mbar)       0.968
rh (%)          -0.675    -0.362                51     VPdef (mbar)      -0.844
VPmax (mbar)     1.309     2.393               957         T (degC)       0.951
VPact (mbar)     0.553    -0.254               132  H2OC (mmol/mol)       1.000
VPdef (mbar)     2.370     7.394              1486     VPmax (mbar)       0.876
sh (g/kg)        0.565    -0.228               146  H2OC (mmol/mol)       1.000
H2OC (mmol/mol)  0.558    -0.240               139        sh (g/kg)       1.000
rho (g/m**3)     0.314     0.141               359         Tpot (K)      -0.981
wv (m/s)         1.541     3.734              1076    max. wv (m/s)       0.969
max. wv (m/s)    1.283     2.121               914         wv (m/s)       0.969
wd (deg)        -0.725    -0.353                 0    max. wv (m/s)       0.247
```

The same analysis runs across all **14 sensors**, not just temperature: each has its own
shape (skew/kurtosis), z-score anomaly count, and strongest correlation partner. Humidity
(`rh (%)`) is left-skewed (−0.675) with only 51 z-score anomalies and correlates most strongly
(negatively) with `VPdef (mbar)` (r = −0.844) — physically sensible, since higher relative
humidity means a smaller vapour pressure deficit. `VPdef (mbar)` itself has by far the most
z-score anomalies (1,486) of any sensor, consistent with its heavy right tail from Section 2.
Near-duplicate physical quantities show near-perfect correlation, e.g. `sh (g/kg)` and
`H2OC (mmol/mol)` (r = 1.000) and `T (degC)`/`Tpot (K)` (r = 0.997).

`wd (deg)` (wind direction) is **circular** — it wraps at 360°, so linear z-scores and Pearson
correlation are invalid for it. It is therefore analysed with **circular statistics**: angular
deviation from the circular mean (`circular_zscore_anomalies`) and the Mardia circular–linear
correlation coefficient (`circular_linear_corr`, unsigned, in [0, 1]). Its strongest association
is with `max. wv (m/s)` (circular-linear coefficient **0.247**) — physically sensible, since the
prevailing wind direction relates to gust strength. This corrects an earlier artefact: under the
old *linear* treatment, `wd` reported its strongest correlation as a spurious `p (mbar)`
(r = −0.074), i.e. essentially uncorrelated with everything, purely because Pearson's r is
meaningless on a wrapped angle. The 0 angular anomalies at threshold 3.0 are now a genuine result
— over a full year wind blows from a broad range of directions, so the circular spread is wide
and no single hour is a 3σ angular outlier — rather than the previous linear artefact.

## 7. Ground-truth control (the algorithm gives the right answer)

`verify_results.py` re-derives results a different way and asserts agreement (20/20 checks as of
3 July 2026, when this section was written; 31/31 as of 15 July 2026; **35/35** as of 16 July
2026 — see `docs/evaluation/` for the current, actively-maintained numbers), including
query-engine == pandas and the data-quality functions == pandas. The query engine is the exact
truth layer; the LLM only explains its numbers.

## 8. Verification results (3 July 2026)

Captured from a real run on the Jena Climate hourly series:

- `python3.12 -m pytest tests/ -q` → `54 passed, 1 skipped, 7 warnings in 8.87s`
- `python3.12 verify_results.py` → `20/20 checks passed.` (exit 0)

Feedback status: **A** (skewness / availability / imbalance) ✅ · **B** (all 14 sensors verified) ✅ ·
**C** (documented in this file + notebooks 01 & 02) ✅. Cloud and local-speed work were deferred.
