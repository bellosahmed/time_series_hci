"""SARIMA forecasting for v3 — forecast hourly temperature and backtest against a naive baseline.

DAD covers only 7 days (168 hourly points), so v1's approach (STL-deseasonalise, then a fixed-order
ARIMA tuned for ~70k hourly points) is overkill; SARIMAX is fit directly to the short hourly series
with a 24-hour seasonal term. Mirrors v1's `forecasting.py` API (`forecast`, `backtest`) so results are
directly comparable. Tested against the real DAD week, the honest result is a **negative** one: the
80% forecast interval is sane, but the backtest MASE is ~1.4 — worse than naive persistence — because
data-centre temperature is already very stable and a 168-hour history is too short for SARIMA to learn
much beyond that. Reported as-is rather than tuned to look better.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def hourly_temperature(df: pd.DataFrame, sensor: float | None = None) -> pd.Series:
    """Hourly-mean temperature series (across all sensors, or one sensor if given)."""
    sub = df if sensor is None else df[df["clientid"] == sensor]
    return sub["temperature"].dropna().resample("1h").mean().interpolate(limit_direction="both")


def forecast(series: pd.Series, steps: int = 24, order: tuple = (1, 0, 1),
            seasonal_order: tuple = (1, 0, 1, 24)) -> pd.DataFrame:
    """Forecast `steps` hours ahead with an 80% interval, using SARIMAX with a 24h seasonal term."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # statsmodels' own ConvergenceWarning on a short series
        fit = SARIMAX(series.astype(float), order=order, seasonal_order=seasonal_order,
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fc = fit.get_forecast(steps=steps)
    mean = fc.predicted_mean
    ci = fc.conf_int(alpha=0.2)
    return pd.DataFrame({"mean": mean.to_numpy(),
                         "lower": ci.iloc[:, 0].to_numpy(),
                         "upper": ci.iloc[:, 1].to_numpy()}, index=mean.index)


def backtest(series: pd.Series, horizon: int = 24, folds: int = 3, order: tuple = (1, 0, 1),
            seasonal_order: tuple = (1, 0, 1, 24)) -> dict:
    """Rolling-origin backtest of `forecast` on the hourly series.

    Reports **MAE**, **RMSE** (°C) and **MASE** — error scaled by a persistence (last-value) baseline;
    MASE < 1 means the model beats naive persistence. On DAD's short week it does not (see module
    docstring), which is reported honestly rather than hidden.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    abs_err: list[float] = []
    naive_err: list[float] = []
    used = 0
    for k in range(folds):
        cut = len(series) - (k + 1) * horizon
        if cut < 48:
            break
        train, actual = series.iloc[:cut], series.iloc[cut:cut + horizon].to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = SARIMAX(train.astype(float), order=order, seasonal_order=seasonal_order,
                          enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            pred = fit.get_forecast(steps=len(actual)).predicted_mean.to_numpy()
        abs_err.extend(np.abs(actual - pred))
        naive_err.extend(np.abs(actual - train.iloc[-1]))   # persistence baseline
        used += 1
    mae = float(np.mean(abs_err))
    base = float(np.mean(naive_err))
    return {
        "folds": used, "horizon": horizon,
        "mae": mae,
        "rmse": float(np.sqrt(np.mean(np.square(abs_err)))),
        "baseline_mae": base,
        "mase": mae / base if base else float("nan"),
    }
