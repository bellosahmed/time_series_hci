"""Fusion of the two channels — the headline experiment.

- **late** (`late_fusion`): rank-normalise each channel's score to [0,1] (so their scales are
  comparable) then combine — `max` is an OR-like union of the two specialists.
- **early** (`early_fusion`): concatenate the value features and the flow features into one matrix and
  fit a single Isolation Forest on the joint space.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut


def _rank(s: pd.Series) -> pd.Series:
    return s.rank(pct=True)


def late_fusion(value_s: pd.Series, flow_s: pd.Series, rule: str = "max", weight: float = 0.5) -> pd.Series:
    v, f = _rank(value_s), _rank(flow_s)
    m = pd.concat([v, f], axis=1).fillna(0.0)
    if rule == "max":
        return m.max(axis=1)
    if rule == "mean":
        return weight * m.iloc[:, 0] + (1 - weight) * m.iloc[:, 1]
    raise ValueError(f"unknown fusion rule: {rule}")


def early_fusion(value_feats: pd.DataFrame, flow_feats: pd.DataFrame, random_state: int = 0) -> pd.Series:
    X = pd.concat([value_feats, flow_feats], axis=1).fillna(0.0)
    model = IsolationForest(random_state=random_state)
    model.fit(X.to_numpy())
    return pd.Series(-model.score_samples(X.to_numpy()), index=X.index)


def supervised_scores(value_s: pd.Series, flow_s: pd.Series, labels, groups) -> pd.Series:
    """Learned fusion: out-of-fold P(attack) from logistic regression over the two channel scores.

    Uses **day-blocked** leave-one-group-out CV (groups = day) so an attack *episode* never appears in
    both train and test — the honest way to score a supervised combiner on episode-clustered attacks.
    A rule-based OR weights the channels equally; this *learns* the weighting from the labels (e.g. to
    trust the value channel more where the flow channel is noisy).
    """
    v = value_s.rank(pct=True).to_numpy()
    f = flow_s.rank(pct=True).to_numpy()
    X = np.column_stack([v, f])
    y = np.asarray(labels)
    groups = np.asarray(groups)
    oof = np.zeros(len(y), dtype=float)
    for train, test in LeaveOneGroupOut().split(X, y, groups):
        if len(np.unique(y[train])) < 2:            # need both classes to fit
            continue
        clf = LogisticRegression(max_iter=1000).fit(X[train], y[train])
        oof[test] = clf.predict_proba(X[test])[:, 1]
    return pd.Series(oof, index=value_s.index)
