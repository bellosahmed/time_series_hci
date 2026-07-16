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


_SEASON_MONTHS = {"winter": (12, 1, 2), "spring": (3, 4, 5), "summer": (6, 7, 8), "autumn": (9, 10, 11)}


def correlation_breakdown(df: pd.DataFrame, a: str, b: str) -> dict:
    """A deeper look at why `get_correlation(df, a, b)` is the strength it is, for the
    "correlation" intent's "why is that only moderate?" follow-up.

    - ``r_squared``: fraction of one sensor's variance a straight-line fit to the other
      "explains" -- the rest comes from other factors, which is the real answer to "why isn't
      it stronger".
    - ``spearman``: rank correlation. Close to the Pearson value -> the relationship is
      genuinely close to linear, not a stronger nonlinear pattern the linear number is hiding.
    - ``by_season``: correlation computed separately within each meteorological season -- shows
      whether the relationship holds steady year-round or is driven by one part of the year
      (whole-record pooling then averages very different regimes into one middling number).
    """
    _check(df, a)
    _check(df, b)
    pearson = float(df[a].corr(df[b]))
    spearman = float(df[a].rank().corr(df[b].rank()))
    by_season = {}
    for name, months in _SEASON_MONTHS.items():
        mask = df.index.month.isin(months)
        sub_a, sub_b = df[a][mask], df[b][mask]
        if len(sub_a) >= 2:
            by_season[name] = float(sub_a.corr(sub_b))
    return {"pearson": pearson, "r_squared": pearson ** 2, "spearman": spearman,
            "by_season": by_season}


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


def rank_calendar_days(df: pd.DataFrame, full_df: pd.DataFrame, sensor: str, ascending: bool,
                        top_n: int = 5, quantile: float = 0.75) -> dict:
    """Rank the calendar days in `df` (an already-scoped slice, e.g. "every June") by `sensor`'s
    daily mean, for the "recommend" intent's directional mode ("recommend hot days in June").

    `ascending=False` ranks for the HIGHEST values ("hottest"/"windiest"); `ascending=True` for
    the LOWEST ("coldest"/"driest"). The top candidate is checked against `full_df` (the whole
    record, not just the scope) at `quantile` — so a "hot" scope that never actually gets hot
    (e.g. "hot days in December") is flagged as not genuinely hot rather than silently presented
    as a real recommendation.

    Returns ``{"candidates": [{"date", "value"}, ...], "meets_threshold": bool, "threshold": float}``.
    Empty ``candidates`` if `df` has no daily data for `sensor`.
    """
    daily = df[sensor].resample("1D").mean().dropna()
    if not len(daily):
        return {"candidates": [], "meets_threshold": False, "threshold": None}
    ranked = daily.sort_values(ascending=ascending).head(top_n)
    candidates = [{"date": ts, "value": float(v)} for ts, v in ranked.items()]

    full_daily = full_df[sensor].resample("1D").mean().dropna()
    q = quantile if not ascending else 1 - quantile
    threshold = float(full_daily.quantile(q))
    best = candidates[0]["value"]
    meets = (best >= threshold) if not ascending else (best <= threshold)
    return {"candidates": candidates, "meets_threshold": meets, "threshold": threshold}


def find_analog_periods(df: pd.DataFrame, sensors: list[str], target_start, target_end,
                         freq: str, top_n: int = 5, search_start=None, search_end=None) -> dict:
    """Rank other periods (day/month/year, per `freq`) by weather similarity to the target
    period, for the "recommend" intent's analog mode ("dates similar to 13 February 2013").

    Buckets the record at `freq` (``"D"``/``"MS"``/``"YS"``), takes each bucket's mean over
    `sensors`, z-scores each sensor across all buckets (so differing units/scales — °C vs mbar —
    don't dominate the distance), then ranks every *other* bucket by Euclidean distance to the
    target bucket in that normalized space.

    `search_start`/`search_end`, if given, restrict which *candidate* buckets are eligible (e.g.
    "similar days, staying within April 2013") — z-scoring still uses the whole record, so
    distances stay on a consistent scale regardless of how narrow the search is.

    Returns ``{"target": {...} | None, "candidates": [{"when", "distance", <sensor values>}, ...]}``
    — empty candidates if the target bucket has no data or fewer than 2 other buckets exist.
    """
    grouped = df[sensors].resample(freq).mean().dropna(how="all")
    if len(grouped) < 2:
        return {"target": None, "candidates": []}

    target_mask = (grouped.index >= pd.Timestamp(target_start)) & (grouped.index <= pd.Timestamp(target_end))
    if not target_mask.any():
        return {"target": None, "candidates": []}

    target_row = grouped.loc[target_mask].mean()
    others = grouped.loc[~target_mask].dropna()
    if search_start is not None or search_end is not None:
        lo = pd.Timestamp(search_start) if search_start is not None else grouped.index.min()
        hi = pd.Timestamp(search_end) if search_end is not None else grouped.index.max()
        others = others.loc[(others.index >= lo) & (others.index <= hi)]
    if not len(others):
        return {"target": target_row.to_dict(), "candidates": []}

    mu, sigma = grouped.mean(), grouped.std().replace(0, 1)
    z_target = (target_row - mu) / sigma
    z_others = (others - mu) / sigma
    dist = ((z_others - z_target) ** 2).sum(axis=1) ** 0.5
    ranked = dist.sort_values().head(top_n)
    candidates = [{"when": ts, "distance": float(d), **others.loc[ts].to_dict()}
                  for ts, d in ranked.items()]
    return {"target": target_row.to_dict(), "candidates": candidates}
