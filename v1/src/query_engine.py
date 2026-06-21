"""Exact query engine — the TRUTH layer.

Responsibilities (Phase 5):
Compute exact answers deterministically. The LLM must NEVER compute these values.
- max/min, mean/std/range
- correlations
- missing-data count
- anomaly windows
- cluster membership
- per-sensor summaries

Each function returns a structured result (value + sensor + timestamp where relevant)
so the explanation layer can describe it without recomputing anything.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Extremum:
    """An exact max/min result: the value, the sensor, and when it occurred."""

    sensor: str
    kind: str  # "max" or "min"
    value: float
    when: pd.Timestamp


def _check(df: pd.DataFrame, sensor: str) -> None:
    if sensor not in df.columns:
        raise KeyError(f"Sensor {sensor!r} not in columns: {list(df.columns)}")


def get_max(df: pd.DataFrame, sensor: str) -> Extremum:
    """Exact maximum of `sensor` with the timestamp at which it occurred."""
    _check(df, sensor)
    s = df[sensor]
    return Extremum(sensor, "max", float(s.max()), s.idxmax())


def get_min(df: pd.DataFrame, sensor: str) -> Extremum:
    """Exact minimum of `sensor` with the timestamp at which it occurred."""
    _check(df, sensor)
    s = df[sensor]
    return Extremum(sensor, "min", float(s.min()), s.idxmin())


def get_mean_std(df: pd.DataFrame, sensor: str) -> dict:
    """Exact mean, std, and range of `sensor`."""
    _check(df, sensor)
    s = df[sensor]
    return {
        "sensor": sensor,
        "mean": float(s.mean()),
        "std": float(s.std()),
        "min": float(s.min()),
        "max": float(s.max()),
        "range": float(s.max() - s.min()),
    }


def get_correlation(df: pd.DataFrame, a: str, b: str) -> float:
    """Exact Pearson correlation between two sensors."""
    _check(df, a)
    _check(df, b)
    return float(df[a].corr(df[b]))


def count_missing(df: pd.DataFrame, sensor: str | None = None) -> int | dict:
    """Exact count of missing values for `sensor`, or per-sensor dict if None."""
    if sensor is None:
        return {c: int(df[c].isna().sum()) for c in df.columns}
    _check(df, sensor)
    return int(df[sensor].isna().sum())


def get_anomaly_windows(flags: pd.Series) -> list[dict]:
    """Return anomaly intervals as a list of {start, end, n_points} dicts."""
    flags = flags.astype(bool)
    out: list[dict] = []
    start = prev = None
    count = 0
    for ts, is_anom in flags.items():
        if is_anom:
            if start is None:
                start = ts
                count = 0
            count += 1
        elif start is not None:
            out.append({"start": start, "end": prev, "n_points": count})
            start = None
        prev = ts
    if start is not None:
        out.append({"start": start, "end": prev, "n_points": count})
    return out


def sensor_summary(df: pd.DataFrame, sensor: str) -> dict:
    """Structured summary (min/max/mean/std/range/missing) for one sensor."""
    summary = get_mean_std(df, sensor)
    summary["missing"] = count_missing(df, sensor)
    summary["max_at"] = get_max(df, sensor).when
    summary["min_at"] = get_min(df, sensor).when
    return summary
