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


def seasonal_value_features(sensor_df: pd.DataFrame) -> pd.DataFrame:
    """Per-sensor z-score against a **time-of-day** baseline (mean/std within each hour-of-day).

    The plain `value_features` z-scores against each sensor's global mean/std, so a reading that is
    normal for the day as a whole but wrong for *this hour* (a peak value at a trough hour) hides inside
    the global range. Grouping by hour-of-day models the diurnal cycle, so the residual reflects
    deviation from the expected value at that time. Falls back to 0 where an hour has no spread.
    """
    hour = sensor_df.index.hour
    z = sensor_df.groupby(hour).transform(
        lambda c: (c - c.mean()) / c.std(ddof=0)
    )
    return z.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def seasonal_value_score(sensor_df: pd.DataFrame) -> pd.Series:
    """Seasonal VALUE score: largest absolute time-of-day z in each window."""
    return seasonal_value_features(sensor_df).abs().max(axis=1)


def flow_score(
    flow_df: pd.DataFrame, random_state: int = 0, method: str = "iforest"
) -> pd.Series:
    """FLOW channel score (higher = more anomalous).

    ``method``:
    - ``"iforest"`` (default): Isolation Forest anomaly score.
    - ``"mahalanobis"``: distance to a Ledoit-Wolf covariance of the standardised flow features — a
      *cross-sectional* model that catches a window jointly off the feature distribution even when no
      single feature is extreme (the notebook-05 lesson: covariance beats axis-aligned splits for that
      geometry, and Isolation Forest dilutes a single-feature outlier).
    - ``"combined"``: the per-window max of the two (rank-normalised), so each catches what it is best at.
    """
    X = flow_df.to_numpy()
    if method in ("mahalanobis", "combined"):
        from sklearn.covariance import LedoitWolf
        from sklearn.preprocessing import StandardScaler

        Xs = StandardScaler().fit_transform(X)
        maha = pd.Series(LedoitWolf().fit(Xs).mahalanobis(Xs), index=flow_df.index)
        if method == "mahalanobis":
            return maha
    iso_model = IsolationForest(random_state=random_state)
    iso_model.fit(X)
    iso = pd.Series(-iso_model.score_samples(X), index=flow_df.index)
    if method == "iforest":
        return iso
    # combined: rank-normalise both to [0,1] and take the max
    r_iso = iso.rank(pct=True)
    r_maha = maha.rank(pct=True)
    return pd.concat([r_iso, r_maha], axis=1).max(axis=1)
