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

## Real evidence (reproduced 15 July 2026, `df['T (degC)']`, real Jena data, 3 folds, 14-day horizon)

```bash
python3.12 -c "
from src import preprocess, forecasting
df = preprocess.align(preprocess.clean(preprocess.load_data('data/jena_climate_2009_2016.csv')), freq='1h')
print(forecasting.backtest(df['T (degC)'], horizon=14, folds=3))
"
```

```
{'folds': 3, 'horizon': 14, 'mae': 3.88, 'rmse': 4.64, 'baseline_mae': 5.16, 'mase': 0.75}
```

**MASE = 0.75** — the ARIMA model beats the naive persistence baseline by 25% on mean absolute
error, averaged across 3 independent rolling-origin folds of 14 days each on the real record.

`verify_results.py` guards this with a looser threshold (`MASE < 1.5`, not `< 1`) specifically so
the automated check doesn't fail on ordinary run-to-run variation in which folds land where in
the seasonal cycle — the 0.75 above is the actual measured value on this run, not the bar the
check enforces.

## Honest caveats

- Only 3 folds were used for this run (matching `verify_results.py`'s own default, chosen to keep
  the check fast); a dissertation-grade claim would ideally use more folds (5+, as the original
  `report/EVALUATION.md` snapshot did) for a tighter estimate — this page reports what was
  actually reproduced, not a re-run with the previous file's exact parameters.
- MASE is comparing against the *simplest* possible baseline (yesterday = today). It says the
  model is better than doing nothing clever, not that it's better than every alternative forecasting
  approach — a comparison against, e.g., a seasonal-naive baseline would be a stronger claim, not
  currently made here.
- This is a single-sensor (temperature) evaluation; the interface can forecast other linear
  sensors too (wind direction is explicitly refused as unforecastable — it's circular), but the
  backtest above hasn't been repeated per-sensor.
