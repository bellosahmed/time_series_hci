"""Explain *why* a window looks anomalous — v3's reasoning layer.

Ported from v1's multi-sensor `describe_moment`, adapted to v3's two-channel design: instead of one
flat sensor array, a window has a **value** channel (per-sensor temperature z-scores) and a **flow**
channel (per-feature traffic-behaviour z-scores). This reports which channel and which feature moved
furthest from its own normal range at a given window, so an anomaly answer can name a driver instead
of just a number. It only *describes* what the two channels show; it never invents a cause.
"""

from __future__ import annotations

import pandas as pd

from .channels import value_features


def value_zscores(sensor_df: pd.DataFrame, when) -> pd.Series:
    """Per-sensor z-score (value channel) at window `when`, largest |z| first."""
    z = value_features(sensor_df).loc[when]
    return z.reindex(z.abs().sort_values(ascending=False).index)


def flow_zscores(flow_df: pd.DataFrame, when) -> pd.Series:
    """Per-feature z-score (flow channel, against each feature's own mean/std) at window `when`."""
    z = (flow_df - flow_df.mean()) / flow_df.std(ddof=0)
    z = z.replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
    row = z.loc[when]
    return row.reindex(row.abs().sort_values(ascending=False).index)


def describe_window(sensor_df: pd.DataFrame, flow_df: pd.DataFrame, when, threshold: float = 2.0) -> str:
    """Human-readable explanation of which channel and feature was unusual at window `when`.

    Compares the top value-channel z (a single-sensor temperature outlier) against the top
    flow-channel z (a traffic-behaviour outlier) and names whichever is more extreme as the likely
    driver — describing the data, not asserting the window is a genuine attack.
    """
    vz = value_zscores(sensor_df, when)
    fz = flow_zscores(flow_df, when)
    v_top, v_val = (vz.index[0], float(vz.iloc[0])) if len(vz) else (None, 0.0)
    f_top, f_val = (fz.index[0], float(fz.iloc[0])) if len(fz) else (None, 0.0)

    when_str = f"{when:%Y-%m-%d %H:%M}" if hasattr(when, "strftime") else str(when)
    parts = []
    if v_top is not None and abs(v_val) >= threshold:
        direction = "high" if v_val > 0 else "low"
        parts.append(f"sensor {v_top} unusually {direction} ({v_val:+.1f}σ)")
    if f_top is not None and abs(f_val) >= threshold:
        parts.append(f"{f_top} unusually far from its normal range ({f_val:+.1f}σ)")

    if not parts:
        return f"At {when_str}, neither channel was notably far from its normal range."
    driver = ("value channel (a sensor reading)" if abs(v_val) >= abs(f_val)
              else "flow channel (traffic behaviour)")
    return f"At {when_str}: " + "; ".join(parts) + f". The {driver} looks like the main driver."
