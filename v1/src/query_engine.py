"""Exact query engine — the TRUTH layer.

Responsibilities (Phase 5):
Compute exact answers deterministically. The LLM must NEVER compute these values.
- max/min, mean/std/range
- correlations
- missing-data count
- anomaly windows
- per-sensor summaries

(Cluster membership/day-types are a separate concern, owned by `clustering.py` and called
directly from `interface.py` — not part of this file.)

Each function returns a structured result (value + sensor + timestamp where relevant)
so the explanation layer can describe it without recomputing anything.
"""

from __future__ import annotations

import pandas as pd


def _check(df: pd.DataFrame, sensor: str) -> None:
    if sensor not in df.columns:
        raise KeyError(f"Sensor {sensor!r} not in columns: {list(df.columns)}")


def fmt_when(ts, with_time: bool = True) -> str:
    """Human-friendly timestamp, e.g. '19 Aug 2012 13:00' (or '19 Aug 2012')."""
    ts = pd.Timestamp(ts)
    return ts.strftime("%-d %b %Y %H:%M" if with_time else "%-d %b %Y")


def get_max(df: pd.DataFrame, sensor: str) -> dict:
    """Exact maximum of `sensor`: {sensor, kind="max", value, when} (when = its timestamp)."""
    _check(df, sensor)
    s = df[sensor]
    return {"sensor": sensor, "kind": "max", "value": float(s.max()), "when": s.idxmax()}


def get_min(df: pd.DataFrame, sensor: str) -> dict:
    """Exact minimum of `sensor`: {sensor, kind="min", value, when} (when = its timestamp)."""
    _check(df, sensor)
    s = df[sensor]
    return {"sensor": sensor, "kind": "min", "value": float(s.min()), "when": s.idxmin()}


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


def slice_window(df: pd.DataFrame, start, end, label: str | None = None) -> tuple[pd.DataFrame | None, str]:
    """Slice `df` to a date window, returning (data, human label).

    Shared by `Interface` and `Agent`'s tool executors, which each used to carry their own
    near-duplicate copy of this (`interface._slice` / `agent._scope`) with a real behavioural gap
    between them -- see below.

    If `start`/`end` are both falsy, returns `(df, "")` -- no window named, use everything.

    If `label` is given (a human-friendly period name, e.g. "summer 2012", already resolved by the
    deterministic parser), it's used directly in the returned text; an EMPTY result returns
    ``(None, label)`` so the caller can report "no data for X" instead of silently proceeding on
    zero rows.

    If `label` is omitted, one is synthesized directly from `start`/`end` (correctly handling a
    one-sided open range: "from 2013-06-01" / "up to 2013-08-31") -- this is `agent.py`'s use case,
    since its tool calls only ever have raw dates, no pre-resolved period name. An empty result in
    this mode ALSO now returns ``(None, label)``: previously `agent._scope` did not check this and
    silently returned an empty DataFrame, which crashed downstream `query_engine` calls (e.g.
    `get_max` on an empty Series raises `ValueError`) -- a real, previously-untested bug, fixed here
    as part of unifying the two copies, not just deduplicated.
    """
    if not start and not end:
        return df, ""
    sub = df.loc[start:end]
    if label is not None:
        if len(sub):
            return sub, f" in {label}"
        return None, label
    if start and end:
        auto_label = f"{start} to {end}"
        prefix = " in "
    else:
        auto_label = f"{start}" if start else f"{end}"
        prefix = " from " if start else " up to "
    if len(sub):
        return sub, f"{prefix}{auto_label}"
    return None, auto_label


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


def threshold_events(df: pd.DataFrame, sensor: str, op: str, value: float) -> dict:
    """Exact count + first/last time a sensor was above ('>') or below ('<') a value."""
    _check(df, sensor)
    s = df[sensor]
    mask = s > value if op == ">" else s < value
    hits = s[mask]
    n = int(mask.sum())
    return {
        "sensor": sensor,
        "op": op,
        "value": float(value),
        "count": n,
        "pct": float(100.0 * n / len(s)) if len(s) else 0.0,
        "first": hits.index.min() if n else None,
        "last": hits.index.max() if n else None,
    }


def weather_report(df: pd.DataFrame, sensors: list[str]) -> dict:
    """Mean/min/max of each requested sensor over the given rows (a 'weather' snapshot)."""
    out: dict[str, dict] = {}
    for s in sensors:
        if s in df.columns:
            col = df[s].astype(float)
            out[s] = {"mean": float(col.mean()), "min": float(col.min()), "max": float(col.max())}
    return out


def sensor_summary(df: pd.DataFrame, sensor: str) -> dict:
    """Structured summary (min/max/mean/std/range/missing) for one sensor."""
    summary = get_mean_std(df, sensor)
    summary["missing"] = count_missing(df, sensor)
    summary["max_at"] = get_max(df, sensor)["when"]
    summary["min_at"] = get_min(df, sensor)["when"]
    return summary
