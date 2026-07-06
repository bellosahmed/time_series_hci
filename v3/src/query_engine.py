"""The truth layer for v3 — exact, deterministic queries over the MESSAGES frame.

Every function here returns a figure computed directly from the data with pandas; the NL interface
and the LLM only *present* these numbers, never compute them. Each is a thin, independently
checkable operation (see `verify_results.py`), which is what makes the answers verifiable.
"""
from __future__ import annotations

import pandas as pd

DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
_STATS = {"max", "min", "mean", "std"}


def _scope(df: pd.DataFrame, day: str | None = None) -> pd.DataFrame:
    return df[df["day"] == day] if day else df


def total_messages(df: pd.DataFrame, day: str | None = None) -> int:
    return int(len(_scope(df, day)))


def count_attacks(df: pd.DataFrame, day: str | None = None, attack_type: str | None = None) -> int:
    """Number of labelled attack packets, optionally scoped to a day and/or attack type."""
    sub = _scope(df, day)
    sub = sub[sub["label"] == 1]
    if attack_type is not None:
        sub = sub[sub["attack_type"] == attack_type]
    return int(len(sub))


def attacks_by_day(df: pd.DataFrame) -> dict[str, int]:
    atk = df[df["label"] == 1]
    return {d: int((atk["day"] == d).sum()) for d in DAYS}


def attack_breakdown(df: pd.DataFrame) -> dict[str, int]:
    """{attack_type: count} over labelled packets (day-level type, per the dataset's limits)."""
    atk = df[df["label"] == 1]
    return {str(k): int(v) for k, v in atk["attack_type"].value_counts().sort_index().items()}


def attacker_ips(df: pd.DataFrame) -> list[str]:
    """Distinct source IPs among labelled attack packets (the DAD attacker is 10.6.56.41)."""
    return sorted(df.loc[df["label"] == 1, "src_ip"].dropna().unique().tolist())


def sensor_ids(df: pd.DataFrame) -> list[float]:
    return sorted(df["clientid"].dropna().unique().tolist())


def temperature_stat(df: pd.DataFrame, stat: str = "mean", sensor: float | None = None,
                     day: str | None = None) -> float:
    """Exact max/min/mean/std of the reported temperatures, optionally per sensor and/or day."""
    if stat not in _STATS:
        raise ValueError(f"stat must be one of {_STATS}")
    sub = _scope(df, day)
    if sensor is not None:
        sub = sub[sub["clientid"] == sensor]
    col = sub["temperature"].dropna()
    return float(getattr(col, stat)()) if len(col) else float("nan")


def message_rate(df: pd.DataFrame, freq: str = "1h", day: str | None = None) -> pd.Series:
    """Message count per time bucket (used for 'busiest hour', traffic-volume questions)."""
    return _scope(df, day).resample(freq).size()


def busiest_hour(df: pd.DataFrame) -> tuple[pd.Timestamp, int]:
    rate = message_rate(df, "1h")
    return (rate.idxmax(), int(rate.max()))
