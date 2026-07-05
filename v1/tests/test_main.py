"""End-to-end smoke test for the pipeline entry point (`src/main.py`).

Builds a tiny Jena-format CSV (so it is fast and needs no real dataset) and runs the whole
load → preprocess → analyse → detect → cluster → query pipeline through `main.run`.
"""

import numpy as np
import pandas as pd

from src import main


def _write_jena_csv(path, days: int = 12):
    """Write a small CSV in the Jena layout ('Date Time' + sensors, 10-min spacing)."""
    idx = pd.date_range("2020-01-01", periods=days * 24 * 6, freq="10min")
    rng = np.random.default_rng(0)
    t = 10 + 8 * np.sin(np.arange(len(idx)) * 2 * np.pi / (24 * 6)) + rng.normal(0, 1, len(idx))
    df = pd.DataFrame({
        "Date Time": idx.strftime("%d.%m.%Y %H:%M:%S"),
        "T (degC)": t,
        "rho (g/m**3)": 1300 - 4 * t + rng.normal(0, 3, len(idx)),
        "wv (m/s)": rng.uniform(0, 5, len(idx)),
    })
    df.loc[5, "wv (m/s)"] = -9999.0                       # a sentinel to exercise cleaning
    df.to_csv(path, index=False)
    return path


def test_pipeline_runs_end_to_end(tmp_path):
    csv = _write_jena_csv(tmp_path / "mini.csv")
    res = main.run(str(csv), sensor="T (degC)", freq="1h")
    assert res["rows"] > 0
    assert res["n_day_clusters"] >= 1
    assert {"sensor_summary", "top_correlations",
            "n_residual_anomalies", "n_iforest_anomalies"} <= res.keys()
    # the query-engine summary is exact and self-consistent
    s = res["sensor_summary"]
    assert s["min"] <= s["mean"] <= s["max"]
