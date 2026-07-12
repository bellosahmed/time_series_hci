"""Multi-sensor reasoning: explain *why* a moment looks anomalous.

Instead of judging temperature alone, this looks at what every sensor did at a
given timestamp (relative to its own normal range) and reports which sensors
moved and in which direction. This is the basis for causal-style explanations
like "temperature dropped while humidity rose" — a cross-sensor explanation feature.

It only *describes* what the data shows (using exact z-scores); it does not
invent causes. Any narrative phrasing can be added by the LLM later.
"""

from __future__ import annotations

import pandas as pd

from .query_engine import fmt_when


def sensor_zscores(df: pd.DataFrame, when) -> pd.Series:
    """Z-score of every sensor at timestamp `when` (vs its own mean/std)."""
    row = df.loc[when]
    z = (row - df.mean()) / df.std(ddof=0)
    return z.sort_values(key=lambda s: s.abs(), ascending=False)


def describe_moment(df: pd.DataFrame, when, top: int = 4, threshold: float = 1.5) -> str:
    """Human-readable description of which sensors were unusual at `when`."""
    z = sensor_zscores(df, when)
    notable = z[z.abs() >= threshold].head(top)
    if notable.empty:
        return f"At {fmt_when(when)}, no sensor was notably far from its normal level."
    parts = []
    for sensor, score in notable.items():
        direction = "high" if score > 0 else "low"
        parts.append(f"{sensor} unusually {direction} ({score:+.1f}σ)")
    # Confidence: many sensors moving together = corroborated real event; one alone = possible fault.
    n_notable = int((z.abs() >= threshold).sum())
    if n_notable >= 3:
        verdict = " Several sensors moved together, so this looks like a real weather event."
    elif n_notable == 1:
        verdict = " Only one sensor moved, so this may be a sensor glitch rather than real weather."
    else:
        verdict = ""
    return f"At {fmt_when(when)}: " + "; ".join(parts) + "." + verdict


def describe_network_moment(
    df: pd.DataFrame, when, flatline_sensor: str | None = None, top: int = 4, threshold: float = 1.5
) -> str:
    """Explain a *network-level* anomaly (from `anomaly.sensor_network_anomalies` or similar).

    A frozen sensor can trip the network detector while looking perfectly normal on a marginal
    z-score (its *level* is unremarkable — only its *variance* has collapsed), so the plain
    `describe_moment` story would misleadingly say "nothing looks unusual". When the caller knows
    which sensor triggered the flat-line detector (`anomaly.flatline_culprit`), report that fault
    directly instead; otherwise fall back to the marginal multi-sensor description.
    """
    if flatline_sensor is not None:
        z = sensor_zscores(df, when)
        level = z.get(flatline_sensor, 0.0)
        return (f"At {fmt_when(when)}: {flatline_sensor} appears frozen (its short-window variance "
                f"has collapsed, even though its level is unremarkable at {level:+.1f}σ) — "
                f"this looks like a sensor glitch (stuck reading) rather than real weather.")
    return describe_moment(df, when, top=top, threshold=threshold)


def co_movement(df: pd.DataFrame, when, target: str, top: int = 3) -> str:
    """Explain the target's deviation in terms of correlated sensors moving together.

    Combines an exact correlation with the actual co-direction at this moment.
    """
    z = sensor_zscores(df, when)
    if target not in z:
        return ""
    tdir = "high" if z[target] > 0 else "low"
    corr = df.corr(numeric_only=True)[target].drop(target)
    others = corr.reindex(corr.abs().sort_values(ascending=False).index).head(top)
    bits = []
    for sensor, r in others.items():
        relation = "moves with" if r > 0 else "moves opposite to"
        odir = "high" if z.get(sensor, 0) > 0 else "low"
        bits.append(f"{sensor} ({relation} {target}, r={r:+.2f}) was {odir}")
    return f"{target} was {tdir}; " + "; ".join(bits) + "."
