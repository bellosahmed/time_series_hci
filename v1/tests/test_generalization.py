"""Generalisation evidence: v1's computational core runs unchanged on a NON-weather domain.

These exercise the *actual* v1 modules (query engine, anomaly detector, interface) on simulated
server-infrastructure metrics — a domain with nothing in common with Jena climate — to show the
core is dataset-agnostic, not hard-wired to weather. (The dataset-specific layers are preprocessing,
which adapts ingestion per source, and the interface's sensor vocabulary, which is swapped per domain
— see the v2 trade system.) The tests pass because the code already generalises; that is the point.
"""

import numpy as np
import pandas as pd

from src import anomaly, interface
from src import query_engine as qe


def _server_df(n=1000, seed=0):
    idx = pd.date_range("2022-01-01", periods=n, freq="h")
    rng = np.random.default_rng(seed)
    load = 50 + 10 * np.sin(np.arange(n) / 24) + rng.normal(0, 2, n)   # daily cycle + noise
    df = pd.DataFrame({"cpu_load": load, "mem_gb": rng.normal(16, 1, n)}, index=idx)
    df.iloc[500, df.columns.get_loc("cpu_load")] = 200.0               # an injected outage spike
    return df


def test_query_engine_generalises_to_non_weather_data():
    df = _server_df()
    assert qe.get_max(df, "cpu_load").value == df["cpu_load"].max()          # exact, any domain
    assert np.isclose(qe.get_mean_std(df, "cpu_load")["mean"], df["cpu_load"].mean())
    assert np.isclose(qe.get_correlation(df, "cpu_load", "mem_gb"),
                      df["cpu_load"].corr(df["mem_gb"]))


def test_anomaly_detector_generalises_to_non_weather_data():
    df = _server_df()
    flags = anomaly.zscore_anomalies(df["cpu_load"], threshold=4.0)
    assert bool(flags.iloc[500])          # catches the injected spike in a different domain


def test_interface_answers_on_new_domain_with_explicit_column():
    df = _server_df()
    ans = interface.Interface(df).ask("max cpu_load")
    assert ans.intent == "max"
    assert ans.value.value == df["cpu_load"].max()   # same interface, new domain, exact number
