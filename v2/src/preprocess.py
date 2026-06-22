"""
v2 preprocessing for the EU monthly trade dataset.

Design follows the EDA findings (notebook 01):
  * the data is already clean (0 missing, no gaps, all values > 0) -> cleaning is minimal;
  * values grow ~exponentially and span a huge range -> offer a **log transform**;
  * useful derived signals: **trade balance** (EXP - IMP) and **total trade** (EXP + IMP).

Everything here is deterministic and importable (the notebooks + tests use it). No plotting.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

EXP, IMP = "_EXP", "_IMP"          # column suffixes


# --------------------------------------------------------------------------- load
def load_trade(path: str | Path) -> pd.DataFrame:
    """Read the trade CSV into a time-sorted DataFrame with a monthly DatetimeIndex."""
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    df.index.name = "month"
    return df


def countries(df: pd.DataFrame) -> list[str]:
    """Country codes present as either an EXP or IMP column (e.g. 'DE', 'FR')."""
    return sorted({c[:-4] for c in df.columns if c.endswith(EXP) or c.endswith(IMP)})


def flow_columns(df: pd.DataFrame) -> list[str]:
    """The raw trade-flow columns (exports & imports), excluding any derived columns."""
    return [c for c in df.columns if c.endswith(EXP) or c.endswith(IMP)]


# --------------------------------------------------------------------------- validate
@dataclass
class QualityReport:
    n_rows: int
    n_cols: int
    start: str
    end: str
    missing_cells: int
    missing_months: int
    nonpositive: int

    @property
    def is_clean(self) -> bool:
        return self.missing_cells == 0 and self.missing_months == 0 and self.nonpositive == 0

    def __str__(self) -> str:
        flag = "CLEAN ✓" if self.is_clean else "ISSUES ✗"
        return (f"[{flag}] {self.n_rows} months x {self.n_cols} cols, {self.start}->{self.end} | "
                f"missing cells={self.missing_cells}, missing months={self.missing_months}, "
                f"non-positive={self.nonpositive}")


def validate(df: pd.DataFrame) -> QualityReport:
    """Check the three things that silently break time-series models."""
    full = pd.date_range(df.index.min(), df.index.max(), freq="MS")
    flows = df[flow_columns(df)]
    return QualityReport(
        n_rows=len(df),
        n_cols=df.shape[1],
        start=str(df.index.min().date()),
        end=str(df.index.max().date()),
        missing_cells=int(df.isna().sum().sum()),
        missing_months=int(len(full.difference(df.index))),
        nonpositive=int((flows <= 0).sum().sum()),   # only flows must be positive
    )


# --------------------------------------------------------------------------- derive
def trade_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Per-country trade balance = exports - imports (positive = surplus). Columns 'CC_BAL'."""
    out = {f"{c}_BAL": df[c + EXP] - df[c + IMP]
           for c in countries(df) if c + EXP in df and c + IMP in df}
    return pd.DataFrame(out, index=df.index)


def total_trade(df: pd.DataFrame) -> pd.DataFrame:
    """Per-country total trade = exports + imports (overall activity). Columns 'CC_TOT'."""
    out = {f"{c}_TOT": df[c + EXP] + df[c + IMP]
           for c in countries(df) if c + EXP in df and c + IMP in df}
    return pd.DataFrame(out, index=df.index)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return raw flows plus derived balance and total columns."""
    return pd.concat([df, trade_balance(df), total_trade(df)], axis=1)


# --------------------------------------------------------------------------- transform
def log_transform(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Natural-log the (strictly positive) flow columns; leave any others untouched.

    Why: trade grows ~exponentially, so a 30% drop in 2008 looks tiny next to 2026 levels.
    On a log scale a proportional change is the same size in any year, so anomaly detectors
    treat the 2008 and 2020 crashes consistently.
    """
    cols = cols or flow_columns(df)
    out = df.copy()
    out[cols] = np.log(out[cols])
    return out


# --------------------------------------------------------------------------- pipeline
def preprocess(path: str | Path, *, log: bool = False, features: bool = False) -> pd.DataFrame:
    """Load -> (optionally) add balance/total -> (optionally) log-transform the flows."""
    df = load_trade(path)
    if features:
        df = add_features(df)
    if log:
        df = log_transform(df)
    return df
