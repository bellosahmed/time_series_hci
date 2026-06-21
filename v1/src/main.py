"""Pipeline orchestration / entry point.

Runs the core analytics pipeline end-to-end on a time-series dataset:
load -> preprocess -> analyse -> detect anomalies -> cluster -> exact queries.

RAG + local LLM and the user interface are deliberately out of scope for v1's core;
this script proves the computational ("truth") half works.

Usage:
    python -m src.main --data ../data/jena_climate_2009_2016.csv --sensor "T (degC)"
"""

from __future__ import annotations

import argparse

from . import analysis, anomaly, clustering, preprocess, query_engine


def run(data_path: str, sensor: str = "T (degC)", freq: str = "1h") -> dict:
    """Run the end-to-end pipeline; return a dict of headline results."""
    # 1. Load + preprocess
    raw = preprocess.load_data(data_path)
    df = preprocess.clean(raw)
    df = preprocess.align(df, freq=freq)

    # 2. Behaviour analysis (daily cycle on hourly data => period 24)
    decomp = analysis.decompose(df[sensor], period=24)
    top_corr = analysis.top_correlations(df, sensor, n=3)

    # 3. Anomaly detection on the residual + multi-sensor Isolation Forest
    resid_anom = anomaly.zscore_anomalies(decomp["resid"].dropna(), threshold=4.0)
    iforest = anomaly.isolation_forest_anomalies(df, contamination=0.01)

    # 4. Clustering of day-types
    feats = clustering.daily_features(df)
    labels = clustering.kmeans_cluster(feats, k=4)

    # 5. Exact queries (truth layer)
    summary = query_engine.sensor_summary(df, sensor)

    return {
        "rows": len(df),
        "sensor_summary": summary,
        "top_correlations": top_corr.to_dict(),
        "n_residual_anomalies": int(resid_anom.sum()),
        "n_iforest_anomalies": int(iforest.sum()),
        "n_day_clusters": int(labels.nunique()),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Time-series HCI core pipeline")
    p.add_argument("--data", default="../data/jena_climate_2009_2016.csv")
    p.add_argument("--sensor", default="T (degC)")
    p.add_argument("--freq", default="1h")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    results = run(args.data, sensor=args.sensor, freq=args.freq)
    print("Pipeline results:")
    for k, v in results.items():
        print(f"  {k}: {v}")
