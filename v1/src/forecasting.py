"""Forecasting-based anomaly detection with ARIMA.

Fit an ARIMA model to the signal, then flag points where the actual value
deviates strongly from the model's one-step prediction (a large in-sample
residual). Hourly temperature is dominated by the daily cycle, so by default we
remove the STL seasonal component first and let ARIMA model the remaining
(deseasonalised) dynamics — this keeps ARIMA orders small and the fit tractable.

Uses statsmodels (already a dependency).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def arima_residuals(series: pd.Series, order: tuple = (2, 1, 2),
                    deseasonalize_period: int | None = 24) -> pd.Series:
    """Return the in-sample ARIMA residuals (actual - one-step prediction).

    The default ``order=(2, 1, 2)`` is a deliberate fixed, low order, not an auto-selected
    one: the series is STL-deseasonalised first (``deseasonalize_period``), which removes the
    daily cycle and leaves short-memory dynamics that a small ARMA captures well. A fixed low
    order keeps the fit fast and stable on ~70k points and avoids the cost/overfitting risk of
    a per-call AIC search; ``order`` is exposed so it can be tuned if needed.
    """
    from statsmodels.tsa.arima.model import ARIMA

    s = series.astype(float).interpolate(limit_direction="both")
    if deseasonalize_period:
        from statsmodels.tsa.seasonal import STL
        seasonal = STL(s, period=deseasonalize_period, robust=False).fit().seasonal
        s = s - seasonal
    fit = ARIMA(s, order=order).fit()
    return fit.resid.rename("arima_resid")


def arima_anomalies(series: pd.Series, order: tuple = (2, 1, 2), threshold: float = 4.0,
                    deseasonalize_period: int | None = 24) -> pd.Series:
    """Flag points whose ARIMA forecast error (z-scored residual) exceeds `threshold`."""
    resid = arima_residuals(series, order=order, deseasonalize_period=deseasonalize_period)
    z = (resid - resid.mean()) / resid.std(ddof=0)
    return (z.abs() > threshold).rename("arima_anomaly")


def forecast(series: pd.Series, steps: int = 14, order: tuple = (2, 1, 2)) -> pd.DataFrame:
    """Forecast `steps` days ahead with an 80% interval.

    Fits ARIMA on the **last year of daily means** (fast, and the recent level/trend is
    what matters for a short horizon). Returns a DataFrame indexed by future dates with
    columns [mean, lower, upper]. Honest limitation: this projects the usual pattern and
    recent trend — it cannot predict shocks (storms, heatwaves), only detect them later.
    """
    from statsmodels.tsa.arima.model import ARIMA

    daily = series.astype(float).resample("1D").mean().dropna()
    recent = daily.iloc[-365:]
    fit = ARIMA(recent, order=order).fit()
    fc = fit.get_forecast(steps=steps)
    mean = fc.predicted_mean
    ci = fc.conf_int(alpha=0.2)
    return pd.DataFrame({"mean": mean.to_numpy(),
                         "lower": ci.iloc[:, 0].to_numpy(),
                         "upper": ci.iloc[:, 1].to_numpy()}, index=mean.index)


def backtest(series: pd.Series, horizon: int = 14, folds: int = 5,
             order: tuple = (2, 1, 2)) -> dict:
    """Rolling-origin backtest of `forecast` on daily means.

    For each fold we fit on data up to a cutoff and forecast `horizon` days, then
    compare to the held-out actuals. Reports **MAE** and **RMSE** (°C), and **MASE**
    — error scaled by a persistence (last-value) baseline; MASE < 1 means the model
    beats naive persistence. (MAPE is deliberately avoided: temperature crosses 0 °C,
    which makes percentage error blow up.)

    Also reports **per-fold MASE** plus its mean/std and a 95% percentile confidence
    interval across folds -- the same statistical-rigor pattern `evaluation.benchmark`
    already applies to the anomaly detectors, so a single lucky/unlucky fold can't be
    mistaken for the model's real expected performance. `mae`/`rmse`/`mase` above remain
    the pooled (all-folds-combined) figures for backward compatibility.
    """
    from statsmodels.tsa.arima.model import ARIMA

    daily = series.astype(float).resample("1D").mean().dropna()
    abs_err: list[float] = []
    naive_err: list[float] = []
    per_fold_mase: list[float] = []
    used = 0
    for k in range(folds):
        cut = len(daily) - (k + 1) * horizon
        if cut < 400:
            break
        train, actual = daily.iloc[:cut], daily.iloc[cut:cut + horizon].to_numpy()
        fit = ARIMA(train.iloc[-365:], order=order).fit()
        pred = fit.get_forecast(steps=len(actual)).predicted_mean.to_numpy()
        fold_abs_err = np.abs(actual - pred)
        fold_naive_err = np.abs(actual - train.iloc[-1])   # persistence baseline
        abs_err.extend(fold_abs_err)
        naive_err.extend(fold_naive_err)
        fold_mae, fold_base = float(np.mean(fold_abs_err)), float(np.mean(fold_naive_err))
        per_fold_mase.append(fold_mae / fold_base if fold_base else float("nan"))
        used += 1
    mae = float(np.mean(abs_err))
    base = float(np.mean(naive_err))
    pf = np.array(per_fold_mase)
    return {
        "folds": used, "horizon": horizon,
        "mae": mae,
        "rmse": float(np.sqrt(np.mean(np.square(abs_err)))),
        "baseline_mae": base,
        "mase": mae / base if base else float("nan"),
        "per_fold_mase": per_fold_mase,
        "mase_mean": float(np.mean(pf)),
        "mase_std": float(np.std(pf)),
        "mase_ci_low": float(np.percentile(pf, 2.5)),
        "mase_ci_high": float(np.percentile(pf, 97.5)),
    }
