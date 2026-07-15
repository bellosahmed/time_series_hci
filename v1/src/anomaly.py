"""Anomaly detection: flag abnormal behaviour, comparing methods.

Responsibilities (Phase 3):
- baseline: z-score on residuals / rolling-window thresholds
- classical ML: Isolation Forest on engineered features
- optional (later): 1D CNN, only if baseline + ML are insufficient

All detectors return a boolean Series (True = anomaly), aligned to the input index,
so they can be compared directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def zscore_anomalies(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Flag points whose global z-score exceeds `threshold` in absolute value."""
    s = series.astype(float)
    z = (s - s.mean()) / s.std(ddof=0)
    return (z.abs() > threshold).rename("zscore_anomaly")


def zscore_borderline(
    series: pd.Series, threshold: float = 3.0, lower_factor: float = 0.75
) -> pd.Series:
    """Flag *near-threshold* points: ``lower_factor*threshold < |z| <= threshold``.

    These are the "almost anomalies" — unusual enough to be worth showing, but below the
    flagging cut-off. Surfacing them lets the UI communicate uncertainty (a soft band around
    the hard yes/no line) instead of a binary verdict. Disjoint from ``zscore_anomalies`` by
    construction. Returns a boolean Series aligned to the input.
    """
    s = series.astype(float)
    z = ((s - s.mean()) / s.std(ddof=0)).abs()
    return ((z > lower_factor * threshold) & (z <= threshold)).rename("zscore_borderline")


def rolling_zscore_anomalies(
    series: pd.Series, window: int = 24, threshold: float = 3.0
) -> pd.Series:
    """Flag points far from their *local* rolling mean (robust to slow trends)."""
    s = series.astype(float)
    mean = s.rolling(window, min_periods=window // 2).mean()
    std = s.rolling(window, min_periods=window // 2).std(ddof=0)
    z = (s - mean) / std.replace(0, np.nan)
    return (z.abs() > threshold).fillna(False).rename("rolling_zscore_anomaly")


def isolation_forest_anomalies(
    features: pd.DataFrame, contamination: float = 0.01, random_state: int = 42
) -> pd.Series:
    """Fit Isolation Forest on (standardised) features; return a boolean Series.

    `features` may be one or many columns (multi-sensor). Rows scoring as outliers
    are marked True.
    """
    X = features.astype(float).to_numpy()
    X = StandardScaler().fit_transform(X)
    model = IsolationForest(contamination=contamination, random_state=random_state)
    labels = model.fit_predict(X)  # -1 = outlier, 1 = inlier
    return pd.Series(labels == -1, index=features.index, name="iforest_anomaly")


def lof_anomalies(
    features: pd.DataFrame, n_neighbors: int = 20, contamination: float = 0.01
) -> pd.Series:
    """Local Outlier Factor: flag points in low-density regions relative to their neighbours.

    LOF is a density-based detector — a point is anomalous if it sits in a sparser region
    than its `n_neighbors` nearest neighbours do. Complements Isolation Forest (which splits
    globally) by catching *local* outliers. Returns a boolean Series (True = anomaly).
    """
    from sklearn.neighbors import LocalOutlierFactor

    X = StandardScaler().fit_transform(features.astype(float).to_numpy())
    labels = LocalOutlierFactor(
        n_neighbors=n_neighbors, contamination=contamination
    ).fit_predict(X)  # -1 = outlier
    return pd.Series(labels == -1, index=features.index, name="lof_anomaly")


def ocsvm_anomalies(
    features: pd.DataFrame, nu: float = 0.01, gamma: str = "scale",
    max_fit: int = 15000, random_state: int = 42,
) -> pd.Series:
    """One-Class SVM: learn a boundary around the bulk of the data; points outside are anomalies.

    `nu` upper-bounds the fraction of outliers. A kernel One-Class SVM fits in ~O(n²), so on
    long series we fit on a random subsample of at most `max_fit` rows and then score *every*
    point — keeping it tractable on the full ~70k-row series while still labelling all of it.
    Returns a boolean Series (True = anomaly).
    """
    from sklearn.svm import OneClassSVM

    X = StandardScaler().fit_transform(features.astype(float).to_numpy())
    model = OneClassSVM(nu=nu, gamma=gamma)
    if len(X) > max_fit:
        rng = np.random.default_rng(random_state)
        model.fit(X[rng.choice(len(X), max_fit, replace=False)])
    else:
        model.fit(X)
    return pd.Series(model.predict(X) == -1, index=features.index, name="ocsvm_anomaly")


def mahalanobis_anomalies(
    features: pd.DataFrame, contamination: float = 0.01
) -> pd.Series:
    """Cross-sectional covariance detector: flag rows inconsistent with the sensor correlations.

    Fits a shrunk (Ledoit-Wolf) mean + covariance on the standardised sensor vectors, then scores
    every row by its Mahalanobis distance to that joint model. Unlike per-sensor z-scores, this
    reads the *instantaneous* multi-sensor vector, so it catches a single channel that is
    inconsistent with its correlated neighbours even when it is not extreme on its own — the common
    IoT single-sensor fault. No training loop, no time window. The top ``contamination`` fraction of
    distances are flagged. Returns a boolean Series (True = anomaly) aligned to ``features``.
    """
    from sklearn.covariance import LedoitWolf

    X = StandardScaler().fit_transform(features.astype(float).to_numpy())
    dist = LedoitWolf().fit(X).mahalanobis(X)     # squared Mahalanobis distance per row
    n_flag = max(1, int(round(contamination * len(X))))
    cutoff = np.sort(dist)[-n_flag]               # keep the n_flag largest distances
    return pd.Series(dist >= cutoff, index=features.index, name="mahalanobis_anomaly")


def flatline_anomalies(
    features: pd.DataFrame, window: int = 4, min_ratio: float = 0.02
) -> pd.Series:
    """Variance-collapse detector: flag timestamps where a sensor's short-window variance dies.

    A stuck / frozen sensor holds a constant value, so its rolling standard deviation collapses to
    ~0 while its neighbours keep varying. Reconstruction and covariance detectors are structurally
    blind to this (a constant segment is trivially "normal"); a variance check is the right tool.
    For each sensor we compare its ``window``-length rolling std to that sensor's own median rolling
    std (its normal wobble); a row is flagged when *any* sensor's ratio falls below ``min_ratio``.
    Returns a boolean Series (True = anomaly) aligned to ``features``.

    ``min_ratio`` defaults to 0.02 — a genuinely stuck sensor drops to ~0 variance, so requiring a
    collapse below 2% of a sensor's normal wobble catches real freezes while leaving merely-quiet
    real windows (e.g. an overnight-stable humidity sensor) unflagged. Looser values (0.1) over-fire
    on real data — on the Jena set they flag ~15% of rows vs ~2% at 0.02.
    """
    Z = StandardScaler().fit_transform(features.astype(float).to_numpy())
    rstd = pd.DataFrame(Z).rolling(window).std(ddof=0)
    base = rstd.median().to_numpy() + 1e-9        # per-sensor normal rolling std
    ratio = rstd.to_numpy() / base
    ratio = np.where(np.isnan(ratio), np.inf, ratio)       # warm-up rows -> inf (never flagged)
    collapsed = ratio.min(axis=1) < min_ratio              # any sensor's variance collapsed
    return pd.Series(collapsed, index=features.index, name="flatline_anomaly")


def sensor_network_anomalies(
    features: pd.DataFrame, contamination: float = 0.01, window: int = 4
) -> pd.Series:
    """Recommended multi-sensor portfolio: Mahalanobis (cross-sectional) OR flat-line (variance).

    Notebook 05's headline finding is that **no single detector wins** on a correlated sensor array:
    a covariance model (Mahalanobis) dominates cross-sectional faults (spike/drift/single-stuck) but
    is beaten on frozen sensors by a variance detector, while the default Isolation Forest *collapses*
    (it splits axis-aligned and averages a single-channel fault across the normal dimensions). This
    portfolio unions the two winners so both fault geometries are covered cheaply, without a deep net.
    A row is anomalous if *either* detector flags it. Returns a boolean Series aligned to ``features``.
    """
    maha = mahalanobis_anomalies(features, contamination=contamination)
    flat = flatline_anomalies(features, window=window)
    return (maha | flat).rename("sensor_network_anomaly")


def network_culprit(features: pd.DataFrame, timestamp) -> str:
    """Name the sensor most responsible for a network anomaly at ``timestamp`` (arg-max attribution).

    Standardises each sensor and returns the one with the largest absolute deviation at that row —
    the cheap, honest attribution the notebook uses (arg-max per-sensor error). Attribution is easier
    than detection: on the sensor array it is near-perfect at moderate fault sizes even for detectors
    that barely flag the fault. Returns the column name.
    """
    std = features.std(ddof=0).replace(0, np.nan)
    z = ((features.loc[timestamp] - features.mean()) / std).abs()
    return str(z.idxmax())


def flatline_culprit(
    features: pd.DataFrame, when, window: int = 4, min_ratio: float = 0.02
) -> str | None:
    """Name the sensor whose short-window variance has collapsed at ``when`` (frozen-sensor attribution).

    Pairs with ``flatline_anomalies``: that function only says *whether* a row is flagged, this says
    *which* sensor caused it, using the same ratio test. Returns ``None`` if no sensor's variance has
    actually collapsed at this row (so callers can fall back to a different explanation).
    """
    Z = StandardScaler().fit_transform(features.astype(float).to_numpy())
    rstd = pd.DataFrame(Z, index=features.index, columns=features.columns).rolling(window).std(ddof=0)
    base = rstd.median() + 1e-9
    ratio = rstd.loc[when] / base
    if ratio.isna().all() or ratio.min() >= min_ratio:
        return None
    return str(ratio.idxmin())


def matrix_profile_anomalies(
    series: pd.Series, window: int = 24, threshold: float = 4.0
) -> pd.Series:
    """Matrix-Profile *discord* detector (via stumpy) — shape-based, not point-based.

    The matrix profile stores, for every length-``window`` subsequence, the z-normalised
    Euclidean distance to its most similar subsequence elsewhere in the series. A subsequence
    with an unusually large nearest-neighbour distance is a **discord**: a shape that repeats
    nowhere else — the classic time-series anomaly. Unlike z-score / Isolation Forest (which
    judge points), this catches anomalous *patterns* (an odd daily curve), so it complements
    them in the detector comparison.

    A discord is flagged with a robust cutoff — median + ``threshold`` × (scaled MAD) of the
    profile, resistant to the discords themselves — and its anchor (subsequence start) point is
    marked. Returns a boolean Series (True = anomaly) aligned to ``series``'s index. Requires
    the optional ``stumpy`` dependency.
    """
    import stumpy

    s = series.astype(float).dropna()
    m = int(window)
    flags = pd.Series(False, index=series.index, name="matrixprofile_anomaly")
    if len(s) <= m + 1:
        return flags
    mp = np.asarray(stumpy.stump(s.to_numpy(), m=m)[:, 0], dtype=float)  # NN distance per subsequence
    med = float(np.median(mp))
    mad = float(np.median(np.abs(mp - med))) or 1e-9
    is_discord = mp > med + threshold * 1.4826 * mad          # 1.4826 -> MAD approximates std
    flags.loc[s.index[np.flatnonzero(is_discord)]] = True     # mark each discord's start point
    return flags


def threshold_sensitivity(series: pd.Series,
                          thresholds=(2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)) -> dict:
    """Anomaly count at each z-score threshold — evidence for the chosen value.

    Pass the (STL-residual) series. Lower thresholds flag more points (more false
    positives); higher thresholds flag fewer (risk missing real events). The default
    4.0 is the knee where only clear events survive on this data.
    """
    return {float(t): int(zscore_anomalies(series, threshold=t).sum()) for t in thresholds}


def contamination_sensitivity(features: pd.DataFrame,
                              contaminations=(0.005, 0.01, 0.02, 0.05)) -> dict:
    """Isolation-Forest anomaly count at each `contamination` prior (sweep to justify it)."""
    return {float(c): int(isolation_forest_anomalies(features, contamination=c).sum())
            for c in contaminations}
