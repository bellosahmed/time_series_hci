"""Feature extraction for v3 — turn the MESSAGES frame into the two detector channels.

- **VALUE channel** (`sensor_series`): per-sensor temperature, aligned to a regular grid. This is
  where a *modification* attack shows up (an out-of-pattern temperature under a legitimate client id).
- **FLOW channel** (`flow_features`): windowed network-traffic aggregates. This is where *duplication*
  (rate up / gaps down / replayed msgids) and *interception* (rate down / gaps up) show up.

**Deliberate design choice (avoid label leakage):** we do **not** use the attacker's IP identity as a
feature (`attacker_share`, `unique_src`). In DAD the attacker `10.6.56.41` appears *only* during
attacks, so any feature encoding it is the label in disguise — it would make detection trivially
perfect and prove nothing. The flow features are all **behavioural aggregates** (rate, timing, replay)
that a real detector could see without knowing which host is malicious. See `report/EVALUATION.md`.
"""
from __future__ import annotations

import pandas as pd


def sensor_series(df: pd.DataFrame, freq: str = "5min", fill: bool = False) -> pd.DataFrame:
    """VALUE channel: temperature per sensor (client id), resampled to a common `freq` grid.

    With ``fill=True`` the small gaps (windows where a sensor's message was dropped or late — ~0.6% of
    the grid) are time-interpolated, so the temperature *residual* is not distorted by missingness. We
    only fill the **value** channel: the **flow** channel deliberately keeps its gaps, because *missing
    messages are the interception signal* — filling them there would erase an attack.
    """
    pub = df[df["temperature"].notna()]
    wide = pub.pivot_table(values="temperature", index=pub.index, columns="clientid", aggfunc="mean")
    wide = wide.resample(freq).mean()
    if fill:
        wide = wide.interpolate(method="time", limit_direction="both")
    return wide


def _dup_ratio(sub: pd.DataFrame) -> float:
    """Fraction of messages in a window that replay an earlier (client id, msg id) — i.e. duplication."""
    key = sub.dropna(subset=["msgid"])
    if len(key) == 0:
        return 0.0
    return key.duplicated(subset=["clientid", "msgid"]).sum() / len(key)


def flow_features(df: pd.DataFrame, window: str = "1min") -> pd.DataFrame:
    """FLOW channel: behavioural traffic features per time window (no attacker-identity leakage)."""
    iat = df.index.to_series().diff().dt.total_seconds()          # inter-arrival gap, seconds
    feats = pd.DataFrame({
        "msg_rate":        df.resample(window).size(),            # duplication ↑ / interception ↓
        "publish_rate":    df["temperature"].notna().resample(window).sum(),
        "iat_mean":        iat.resample(window).mean(),           # interception ↑ / duplication ↓
        "iat_std":         iat.resample(window).std(),
    })
    # per-window duplicate-msgid ratio (Grouper passes the whole sub-frame, unlike resample().apply)
    feats["dup_msgid_ratio"] = df.groupby(pd.Grouper(freq=window)).apply(_dup_ratio)  # duplication signal
    return feats.fillna(0.0).astype(float)


def _dominant_type(s: pd.Series) -> str:
    t = s[s != "none"]
    return t.value_counts().idxmax() if len(t) else "none"


def window_labels(df: pd.DataFrame, window: str = "1min") -> pd.DataFrame:
    """Weak window labels: a window is an attack if ANY labelled packet falls in it (+ its type)."""
    label = df["label"].resample(window).max().fillna(0).astype("int8")
    attack_type = df["attack_type"].resample(window).apply(_dominant_type)
    return pd.DataFrame({"label": label, "attack_type": attack_type})
