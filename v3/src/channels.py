"""The two detector channels.

- **value** (`value_score`): per-window max |z| of the sensor temperatures — a modified reading breaks
  its sensor's pattern and spikes the z-score. `value_features` is the per-sensor z matrix used by
  early fusion (z = 0 where a sensor is silent → neutral).
- **flow** (`flow_score`): an Isolation Forest over the behavioural flow features — duplication and
  interception distort rate/timing/replay, which the forest isolates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def value_features(sensor_df: pd.DataFrame) -> pd.DataFrame:
    """Per-sensor z-score matrix of temperature (0 where a sensor didn't report)."""
    z = (sensor_df - sensor_df.mean()) / sensor_df.std(ddof=0)
    return z.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def value_score(sensor_df: pd.DataFrame) -> pd.Series:
    """VALUE channel score: the largest absolute per-sensor z in each window."""
    return value_features(sensor_df).abs().max(axis=1)


def flow_score(flow_df: pd.DataFrame, random_state: int = 0) -> pd.Series:
    """FLOW channel score: Isolation Forest anomaly score (higher = more anomalous)."""
    model = IsolationForest(random_state=random_state)
    model.fit(flow_df.to_numpy())
    return pd.Series(-model.score_samples(flow_df.to_numpy()), index=flow_df.index)
