"""
v2 multi-country reasoning — the interpretation layer.

Anomaly detection (notebook 04) says *when* something is wrong. Reasoning says *what* and *who*:
when several countries are abnormal in the same month, is it a **shared shock** (the whole region
moved together) or an **isolated** one-country event? This is the bridge to the explanation/LLM
layer: the numbers here are computed deterministically; the LLM later only phrases them.

Deterministic + importable; no plotting.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .anomaly import seasonal_naive_z


# --------------------------------------------------------------------------- cross-section
def cross_section_z(logged: pd.DataFrame, flow: str = "_EXP", period: int = 12) -> pd.DataFrame:
    """Per-country standardised year-on-year growth (seasonal-naive z), one column per country.

    Rows = months, columns = country codes. A row is the 'cross-section' — how every country
    looked that month, on a comparable scale.
    """
    cols = [c for c in logged.columns if c.endswith(flow)]
    return pd.DataFrame({c[: -len(flow)]: seasonal_naive_z(logged[c], period) for c in cols})


def systemic_timeline(cz: pd.DataFrame, thresh: float = 2.0) -> pd.Series:
    """How many countries were abnormal (|z| >= thresh) each month."""
    return (cz.abs() >= thresh).sum(axis=1)


# --------------------------------------------------------------------------- per-month reasoning
@dataclass
class ShockReport:
    month: str
    n_total: int          # countries with data this month
    n_down: int           # abnormally low (<= -thresh)
    n_up: int             # abnormally high (>= +thresh)
    fraction: float       # share of countries abnormal in the dominant direction
    direction: str        # 'down' | 'up' | 'none'
    kind: str             # 'shared shock' | 'isolated' | 'normal'
    top: list = field(default_factory=list)   # most extreme contributors (country, z)

    def sentence(self) -> str:
        if self.kind == "normal":
            return f"{self.month}: no abnormal countries — a normal month."
        who = ", ".join(f"{c} ({z:+.1f}σ)" for c, z in self.top)
        return (f"{self.month}: {self.n_down + self.n_up}/{self.n_total} countries abnormal, "
                f"mostly {self.direction} — a {self.kind}. Largest moves: {who}.")


def month_breakdown(cz: pd.DataFrame, month, top: int = 5) -> pd.Series:
    """The country z-scores for one month, sorted most-negative first."""
    return cz.loc[month].dropna().sort_values()


def shared_shock(cz: pd.DataFrame, month, thresh: float = 2.0, systemic_frac: float = 0.30) -> ShockReport:
    """Classify a month as a shared shock, isolated event, or normal — with the evidence."""
    row = cz.loc[month].dropna()
    n_total = len(row)
    down = row[row <= -thresh]
    up = row[row >= thresh]
    n_down, n_up = len(down), len(up)

    if n_down == 0 and n_up == 0:
        return ShockReport(str(pd.Timestamp(month).date()), n_total, 0, 0, 0.0, "none", "normal")

    if n_down >= n_up:
        direction, abn, frac = "down", down, n_down / n_total
        top = [(c, float(row[c])) for c in abn.sort_values().index[:5]]
    else:
        direction, abn, frac = "up", up, n_up / n_total
        top = [(c, float(row[c])) for c in abn.sort_values(ascending=False).index[:5]]

    kind = "shared shock" if frac >= systemic_frac else "isolated"
    return ShockReport(str(pd.Timestamp(month).date()), n_total, n_down, n_up,
                       round(float(frac), 2), direction, kind, top)


def explain_month(cz: pd.DataFrame, month, thresh: float = 2.0) -> ShockReport:
    """Convenience wrapper returning the classified ShockReport for a month."""
    return shared_shock(cz, month, thresh=thresh)


def find_isolated_months(cz: pd.DataFrame, thresh: float = 2.0) -> pd.DatetimeIndex:
    """Months where exactly one country was abnormal — single-country (idiosyncratic) events."""
    counts = systemic_timeline(cz, thresh)
    return counts[counts == 1].index


def co_movement(cz: pd.DataFrame, a: str, b: str) -> float:
    """Correlation of two countries' standardised growth — do they move together?"""
    return float(cz[a].corr(cz[b]))
