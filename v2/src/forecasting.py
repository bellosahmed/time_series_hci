"""
v2 forecasting for the EU monthly trade dataset.

SARIMA (seasonal ARIMA) configured as notebook 03's ADF test prescribed: difference d=1 and
seasonal difference D=1 at period 12. Two jobs:
  * forecast() - project a series forward with a 95% prediction band;
  * backtest()  - the honesty check: train on the past, predict UNSEEN future months, measure error.
                  This is how we tell "genuinely predicts" from "memorised the training data".

Deterministic + importable; no plotting.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

DEFAULT_ORDER = (1, 1, 1)
DEFAULT_SEASONAL = (1, 1, 1, 12)


def fit_sarima(series: pd.Series, order=DEFAULT_ORDER, seasonal_order=DEFAULT_SEASONAL):
    """Fit a SARIMA model (lenient on stationarity/invertibility for robustness)."""
    return SARIMAX(series, order=order, seasonal_order=seasonal_order,
                   enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)


@dataclass
class Forecast:
    mean: pd.Series       # central forecast
    lower: pd.Series      # lower edge of the prediction band
    upper: pd.Series      # upper edge


def forecast(series: pd.Series, steps: int = 56, log: bool = True,
             alpha: float = 0.05, order=DEFAULT_ORDER, seasonal_order=DEFAULT_SEASONAL) -> Forecast:
    """Forecast `steps` months ahead with a (1-alpha) prediction band. log=True models on log scale."""
    y = np.log(series) if log else series
    res = fit_sarima(y, order=order, seasonal_order=seasonal_order)
    fc = res.get_forecast(steps=steps)
    mean = fc.predicted_mean
    ci = fc.conf_int(alpha=alpha)
    lower, upper = ci.iloc[:, 0], ci.iloc[:, 1]
    if log:
        mean, lower, upper = np.exp(mean), np.exp(lower), np.exp(upper)
    return Forecast(mean, lower, upper)


@dataclass
class Backtest:
    mape: float          # mean absolute % error on UNSEEN months (lower = better)
    rmse: float          # root mean squared error (same units as series)
    coverage: float      # % of actuals that fell inside the 95% band (~95 if well-calibrated)
    pred: pd.Series
    actual: pd.Series
    lower: pd.Series
    upper: pd.Series

    def __str__(self) -> str:
        return f"MAPE={self.mape:.1f}%  RMSE={self.rmse:,.0f}  band-coverage={self.coverage:.0f}%"


def backtest(series: pd.Series, horizon: int = 24, log: bool = True, alpha: float = 0.05,
             order=DEFAULT_ORDER, seasonal_order=DEFAULT_SEASONAL) -> Backtest:
    """Train on all but the last `horizon` months, forecast them, and score against the actuals."""
    train, test = series.iloc[:-horizon], series.iloc[-horizon:]
    y = np.log(train) if log else train
    res = fit_sarima(y, order=order, seasonal_order=seasonal_order)
    fc = res.get_forecast(steps=horizon)
    mean = fc.predicted_mean
    ci = fc.conf_int(alpha=alpha)
    lower, upper = ci.iloc[:, 0], ci.iloc[:, 1]
    if log:
        mean, lower, upper = np.exp(mean), np.exp(lower), np.exp(upper)
    mean.index = lower.index = upper.index = test.index

    err = (mean - test) / test
    mape = float(np.mean(np.abs(err)) * 100)
    rmse = float(np.sqrt(np.mean((mean - test) ** 2)))
    coverage = float(((test >= lower) & (test <= upper)).mean() * 100)
    return Backtest(mape, rmse, coverage, mean, test, lower, upper)


def backtest_from(series: pd.Series, cutoff: str, horizon: int, **kw) -> Backtest:
    """Backtest where training ends at `cutoff` and we predict the next `horizon` months.

    Use this to test a specific window (e.g. forecast across COVID to show a shock is
    unpredictable — the actuals will fall OUTSIDE the band, which is how we'd detect it live).
    """
    train = series.loc[:cutoff]
    future = series.loc[cutoff:].iloc[1:horizon + 1]
    combined = pd.concat([train, future])
    return backtest(combined, horizon=len(future), **kw)
