"""M3 runner — the two-channel + fusion comparison, graded on the real labels.

Builds a common window grid, computes the value and flow channel scores, forms early/late fusion, and
evaluates all four configs (AP, best-F1 operating point, and recall per attack type) against DAD's
ground-truth labels.
"""
from __future__ import annotations

import pandas as pd

from . import channels, features, fusion
from . import evaluation as ev


def build_windows(df: pd.DataFrame, window: str = "5min"):
    """Align labels, flow features and the per-sensor temperature grid on one window index."""
    labels = features.window_labels(df, window)
    flow = features.flow_features(df, window).reindex(labels.index).fillna(0.0)
    # value channel: fill the small temperature gaps (flow keeps its gaps — they are the intercept signal)
    sensors = features.sensor_series(df, window, fill=True).reindex(labels.index)
    return labels, flow, sensors


def run_experiment(df: pd.DataFrame, window: str = "5min") -> pd.DataFrame:
    labels, flow, sensors = build_windows(df, window)
    y = labels["label"].to_numpy()
    atype = labels["attack_type"].to_numpy()

    vfeat = channels.value_features(sensors)
    value_s = channels.value_score(sensors)
    flow_s = channels.flow_score(flow)
    # day per window (day-blocked groups for the supervised combiner; ffill covers empty windows)
    groups = (df["day"].resample(window).agg(lambda s: s.iloc[0] if len(s) else None)
              .reindex(labels.index).ffill().bfill())
    configs = {
        "flow_only": flow_s,
        "value_only": value_s,
        "early_fusion": fusion.early_fusion(vfeat, flow),
        "late_fusion": fusion.late_fusion(value_s, flow_s, rule="max"),
        "supervised": fusion.supervised_scores(value_s, flow_s, y, groups.to_numpy()),
    }

    rows = []
    for name, scores in configs.items():
        s = scores.reindex(labels.index).fillna(0.0).to_numpy()
        bf = ev.best_f1(s, y)
        pa = ev.best_f1_point_adjusted(s, y)
        lo, hi = ev.bootstrap_ci(s, y)
        ptr = ev.per_type_recall(s, y, atype, bf["threshold"])
        rows.append({
            "config": name,
            "AP": ev.average_precision(s, y), "AP_lo": lo, "AP_hi": hi,
            "f1": bf["f1"], "precision": bf["precision"], "recall": bf["recall"],
            "f1_pa": pa["f1"], "recall_pa": pa["recall"],
            **{f"recall_{k}": v for k, v in ptr.items()},
        })
    return pd.DataFrame(rows)
