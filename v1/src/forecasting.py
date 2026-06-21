"""Forecasting-based anomaly detection with ARIMA.

Fit an ARIMA model to the signal, then flag points where the actual value
deviates strongly from the model's one-step prediction (a large in-sample
residual). Hourly temperature is dominated by the daily cycle, so by default we
remove the STL seasonal component first and let ARIMA model the remaining
(deseasonalised) dynamics — this keeps ARIMA orders small and the fit tractable.

Uses statsmodels (already a dependency).
"""

from __future__ import annotations

import pandas as pd


def arima_residuals(series: pd.Series, order: tuple = (2, 1, 2),
                    deseasonalize_period: int | None = 24) -> pd.Series:
    """Return the in-sample ARIMA residuals (actual - one-step prediction)."""
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
