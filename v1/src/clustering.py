"""Clustering: group similar behaviour (normal vs abnormal, or similar sensors).

Responsibilities (Phase 4):
- engineer window/feature vectors
- K-means and/or hierarchical clustering
- interpret clusters; expose membership lookup
"""

from __future__ import annotations

import pandas as pd


def kmeans_cluster(features: pd.DataFrame, k: int = 3) -> pd.Series:
    """Cluster feature vectors with K-means; return labels. TODO."""
    raise NotImplementedError


def cluster_membership(labels: pd.Series, index) -> int:
    """Return the cluster label for a given index. TODO."""
    raise NotImplementedError
