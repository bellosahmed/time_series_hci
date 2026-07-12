"""
v2 query engine — THE TRUTH LAYER.

Every number a user can ask for is computed here, deterministically, from the data. This is the
source of truth: the interface and (later) the local LLM call these functions and may only *phrase*
the results — they never compute or invent a number. Each function is exact and unit-tested against
plain pandas.

Vocabulary (trade): EXP = exports, IMP = imports, balance = EXP-IMP (surplus/deficit),
total = EXP+IMP. Values are EUR millions, monthly.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .preprocess import countries

FLOWS = {"EXP": "_EXP", "IMP": "_IMP"}


def _col(country: str, flow: str) -> str:
    flow = flow.upper()
    if flow not in FLOWS:
        raise ValueError(f"flow must be EXP or IMP, got {flow!r}")
    return f"{country}{FLOWS[flow]}"


@dataclass
class Extremum:
    country: str
    flow: str
    month: str
    value: float

    def __str__(self) -> str:
        return f"{self.country} {self.flow}: {self.value:,.1f} EUR m in {self.month}"


# --------------------------------------------------------------------------- extremes
def get_max(df: pd.DataFrame, country: str, flow: str) -> Extremum:
    """Largest single month for a country's flow."""
    s = df[_col(country, flow)]
    return Extremum(country, flow.upper(), str(s.idxmax().date()), float(s.max()))


def get_min(df: pd.DataFrame, country: str, flow: str) -> Extremum:
    """Smallest single month for a country's flow."""
    s = df[_col(country, flow)]
    return Extremum(country, flow.upper(), str(s.idxmin().date()), float(s.min()))


# --------------------------------------------------------------------------- stats
def get_mean_std(df: pd.DataFrame, country: str, flow: str) -> tuple[float, float]:
    s = df[_col(country, flow)]
    return float(s.mean()), float(s.std())


def get_correlation(df: pd.DataFrame, a: str, b: str, flow: str = "EXP", growth: bool = True) -> float:
    """Correlation between two countries' flow. growth=True uses month-over-month % change."""
    sa, sb = df[_col(a, flow)], df[_col(b, flow)]
    if growth:
        sa, sb = sa.pct_change(), sb.pct_change()
    return float(sa.corr(sb))


# --------------------------------------------------------------------------- balance / total
def trade_balance(df: pd.DataFrame, country: str, month: str | None = None):
    """Exports - imports. A month string returns one number; None returns the full series."""
    bal = df[_col(country, "EXP")] - df[_col(country, "IMP")]
    return float(bal.loc[month]) if month else bal


def total_trade(df: pd.DataFrame, country: str, month: str | None = None):
    """Exports + imports."""
    tot = df[_col(country, "EXP")] + df[_col(country, "IMP")]
    return float(tot.loc[month]) if month else tot


def biggest_surplus(df: pd.DataFrame, month: str) -> Extremum:
    """Country with the largest trade surplus (EXP-IMP) in a given month."""
    bals = {c: trade_balance(df, c, month) for c in countries(df)}
    c = max(bals, key=bals.get)
    return Extremum(c, "BALANCE", month, bals[c])


def biggest_deficit(df: pd.DataFrame, month: str) -> Extremum:
    """Country with the largest trade deficit (most negative balance) in a given month."""
    bals = {c: trade_balance(df, c, month) for c in countries(df)}
    c = min(bals, key=bals.get)
    return Extremum(c, "BALANCE", month, bals[c])


# --------------------------------------------------------------------------- change
def growth(df: pd.DataFrame, country: str, flow: str, start: str, end: str) -> float:
    """Percentage change in a flow between two months."""
    s = df[_col(country, flow)]
    return float((s.loc[end] / s.loc[start] - 1) * 100)


def worst_drop(df: pd.DataFrame, country: str, flow: str) -> Extremum:
    """Largest single-month percentage fall for a flow (month-over-month)."""
    s = df[_col(country, flow)]
    pct = s.pct_change() * 100
    m = pct.idxmin()
    return Extremum(country, flow.upper(), str(m.date()), float(pct.loc[m]))


# --------------------------------------------------------------------------- ranking / summary
def rank_countries(df: pd.DataFrame, flow: str, month: str, n: int = 5) -> pd.Series:
    """Top-n countries by a flow's value in a given month."""
    vals = {c: float(df[_col(c, flow)].loc[month]) for c in countries(df)}
    return pd.Series(vals).sort_values(ascending=False).head(n)


def country_summary(df: pd.DataFrame, country: str) -> dict:
    """A compact, exact fact-sheet for one country (everything the LLM might need)."""
    exp, imp = df[_col(country, "EXP")], df[_col(country, "IMP")]
    bal = exp - imp
    return {
        "country": country,
        "months": int(len(df)),
        "exp_mean": float(exp.mean()),
        "imp_mean": float(imp.mean()),
        "exp_max": get_max(df, country, "EXP").__dict__,
        "exp_min": get_min(df, country, "EXP").__dict__,
        "latest_balance": float(bal.iloc[-1]),
        "avg_balance": float(bal.mean()),
        "surplus_share": float((bal > 0).mean()),   # fraction of months in surplus
    }
