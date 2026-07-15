# 3. Forecasting — backtested against a baseline, not a bare error number

## What's being evaluated

The 14-day-ahead ARIMA forecast in `src/forecasting.py`, used when a user asks something like
"what will the temperature be next week?"

## Method

**Rolling-origin backtest.** Rather than fitting once and reporting error on a single held-out
window (which can be lucky or unlucky), `forecasting.backtest()` repeats the fit-and-forecast
cycle across several different cutoff points moving backward through the record: fit on
everything up to a cutoff, forecast the next `horizon` days, compare to what actually happened,
then move the cutoff back and repeat. This is the standard way to backtest a time-series model
without letting a single fold decide the reported number.

**MASE instead of a bare error.** A raw MAE ("the forecast is off by 3.9°C on average") doesn't
tell you if that's *good* — good compared to what? `forecasting.backtest()` reports **MASE**
(Mean Absolute Scaled Error): the forecast's error divided by a naive "tomorrow will be like
today" persistence baseline's error, computed on the *same* folds.

```
MASE = MAE(model) / MAE(naive persistence)
```

MASE < 1 means the model is actually earning its complexity by beating the dumbest possible
forecast; MASE ≥ 1 means a naive guess would have done just as well or better — a real, useful
distinction a bare MAE can't make on its own. (MAPE — percentage error — is deliberately avoided
here: temperature crosses 0°C, which makes percentage error blow up or flip sign meaninglessly.)

**Per-fold confidence interval, not a bare point estimate.** A single pooled MASE across a
handful of folds can look better than the model's real, reliable performance if it happens to
land on favourable folds. `backtest()` also returns each fold's own MASE plus the mean/std and a
95% percentile confidence interval across folds — the same statistical-rigor pattern
`evaluation.benchmark` already applies to the anomaly detectors (§2), so one lucky/unlucky stretch
of the seasonal cycle can't be mistaken for the model's real expected performance.

## Real evidence (reproduced 15 July 2026, `df['T (degC)']`, real Jena data, 20 folds, 14-day horizon)

```bash
python3.12 -c "
from src import preprocess, forecasting
df = preprocess.align(preprocess.clean(preprocess.load_data('data/jena_climate_2009_2016.csv')), freq='1h')
print(forecasting.backtest(df['T (degC)'], horizon=14, folds=20))
"
```

```
folds=20, pooled mase=0.865, mase_mean=0.952, mase_std=0.359, 95% CI=[0.388, 1.697]
per_fold_mase = [1.04, 0.55, 0.81, 1.04, 1.01, 1.66, 1.34, 0.86, 0.50, 1.02,
                 0.93, 1.40, 0.64, 0.95, 0.29, 0.69, 0.98, 0.66, 0.94, 1.73]
```

**This is a real result the earlier, 3-fold-only version of this page hid.** With only 3 folds
(the original evaluation, run 15 July 2026 earlier the same day), MASE came out 0.75 — a
comfortable "beats persistence by 25%" headline. With 20 folds properly computed: the *pooled*
MASE is still under 1 (0.865), but the **per-fold mean is 0.952 (barely under 1) and the 95%
confidence interval is [0.388, 1.697] — which crosses 1.0**. Individual folds swing from a strong
win (0.29) to a clear loss (1.73). **The honest, statistically supportable conclusion is: on
average the model edges out naive persistence, but the confidence interval means this can't be
claimed reliably — some 14-day windows it wins decisively, others it loses to just repeating
yesterday's value.** This is a more rigorous and more honest finding than the original single
headline number, and it is the number that should be cited, not the earlier 0.75.

`verify_results.py`'s check (`MASE < 1.5` on the *pooled* figure) still passes — it's a loose
sanity bound, not a claim of reliable superiority; the CI is the number that actually answers
"does this reliably beat the baseline."

## Honest caveats

- MASE is comparing against the *simplest* possible baseline (yesterday = today). It says the
  model is better than doing nothing clever *on average*, not that it's better than every
  alternative forecasting approach or that it *reliably* beats even that simple baseline (see the
  CI above) — a comparison against, e.g., a seasonal-naive baseline would be a further, stronger
  test, not currently made here.
- This is a single-sensor (temperature) evaluation; the interface can forecast other linear
  sensors too (wind direction is explicitly refused as unforecastable — it's circular), but the
  backtest above hasn't been repeated per-sensor.
- 20 folds is still a finite sample of the seasonal cycle (2009–2016); a fold count near the
  dataset's real maximum (180, at 14-day horizon) would tighten the CI further but was judged not
  worth the extra runtime for this pass.
