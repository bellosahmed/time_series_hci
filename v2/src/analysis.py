"""
v2 behaviour analysis for the EU monthly trade dataset.

Deterministic, importable analytics used by notebook 03, the tests, and later steps:
  * trend (slope per country),
  * STL decomposition + trend/seasonal strength (period = 12 months),
  * rolling statistics (local level & volatility),
  * stationarity (ADF) and autocorrelation (ACF/PACF) to inform forecasting models,
  * cross-country correlations and lead-lag (does one country move before another?),
  * data caveats: per-country coverage.

No plotting here — the notebook does the plots.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, pacf

from .preprocess import countries, flow_columns


# --------------------------------------------------------------------------- trend
def trend_slope(series: pd.Series) -> float:
    """OLS slope of the series against time index (units = value per month)."""
    y = series.to_numpy(dtype=float)
    x = np.arange(len(y))
    return float(np.polyfit(x, y, 1)[0])


def trend_slopes(df: pd.DataFrame, cols: list[str] | None = None) -> pd.Series:
    """Trend slope for each column (use a log-transformed df to read these as growth rates)."""
    cols = cols or flow_columns(df)
    return pd.Series({c: trend_slope(df[c]) for c in cols}).sort_values()


# --------------------------------------------------------------------------- decomposition
def decompose(series: pd.Series, period: int = 12, robust: bool = True):
    """STL decomposition (trend + seasonal + residual). Monthly data -> period = 12."""
    return STL(series, period=period, robust=robust).fit()


def _strength(numerator_resid: pd.Series, component: pd.Series) -> float:
    """Hyndman's feature strength: 1 - Var(resid) / Var(resid + component), clipped to [0, 1]."""
    var_r = np.var(numerator_resid)
    var_rc = np.var(numerator_resid + component)
    if var_rc == 0:
        return 0.0
    return float(max(0.0, 1.0 - var_r / var_rc))


def trend_strength(stl) -> float:
    return _strength(stl.resid, stl.trend)


def seasonal_strength(stl) -> float:
    return _strength(stl.resid, stl.seasonal)


# --------------------------------------------------------------------------- rolling
def rolling_stats(series: pd.Series, window: int = 12) -> pd.DataFrame:
    """Rolling mean (local level) and std (local volatility)."""
    return pd.DataFrame({
        "mean": series.rolling(window).mean(),
        "std": series.rolling(window).std(),
    })


# --------------------------------------------------------------------------- stationarity
@dataclass
class ADFResult:
    stat: float
    pvalue: float
    stationary: bool      # True if we reject the unit-root null at 5%

    def __str__(self) -> str:
        verdict = "stationary" if self.stationary else "NON-stationary (has trend/unit root)"
        return f"ADF stat={self.stat:.3f}, p={self.pvalue:.4f} -> {verdict}"


def adf_test(series: pd.Series, alpha: float = 0.05) -> ADFResult:
    """Augmented Dickey-Fuller test for stationarity."""
    s = series.dropna()
    stat, pvalue, *_ = adfuller(s, autolag="AIC")
    return ADFResult(stat=float(stat), pvalue=float(pvalue), stationary=bool(pvalue < alpha))


def acf_pacf(series: pd.Series, nlags: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Autocorrelation and partial autocorrelation up to nlags (informs ARIMA orders)."""
    s = series.dropna()
    return acf(s, nlags=nlags), pacf(s, nlags=nlags)


# --------------------------------------------------------------------------- correlation
def correlations(df: pd.DataFrame, cols: list[str] | None = None, growth: bool = True) -> pd.DataFrame:
    """Correlation matrix. growth=True correlates month-over-month % change (removes shared trend)."""
    cols = cols or flow_columns(df)
    sub = df[cols]
    if growth:
        sub = sub.pct_change()
    return sub.corr()


def top_pairs(corr: pd.DataFrame, n: int = 5) -> pd.Series:
    """The n most correlated distinct column pairs (upper triangle, no self-pairs)."""
    m = corr.where(~np.eye(len(corr), dtype=bool)).stack()
    m = m[m.index.get_level_values(0) < m.index.get_level_values(1)]
    return m.sort_values(ascending=False).head(n)


def lead_lag(a: pd.Series, b: pd.Series, max_lag: int = 6) -> tuple[int, float]:
    """Best lag k (in months) maximising corr(a_t, b_{t+k}) on growth.

    k>0 means `a` leads `b` (a moves first). Returns (best_lag, best_corr).
    """
    ga, gb = a.pct_change().dropna(), b.pct_change().dropna()
    idx = ga.index.intersection(gb.index)
    ga, gb = ga.loc[idx], gb.loc[idx]
    best_lag, best_corr = 0, -2.0
    for k in range(-max_lag, max_lag + 1):
        c = ga.corr(gb.shift(-k))
        if c is not None and c > best_corr:
            best_lag, best_corr = k, float(c)
    return best_lag, best_corr


# --------------------------------------------------------------------------- caveats
def coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Per-country first/last month with data and the count of valid points (data caveat check)."""
    rows = {}
    for c in countries(df):
        cols = [c + "_EXP", c + "_IMP"]
        cols = [x for x in cols if x in df.columns]
        valid = df[cols].dropna(how="all")
        rows[c] = {"first": valid.index.min(), "last": valid.index.max(), "n": len(valid)}
    return pd.DataFrame(rows).T
