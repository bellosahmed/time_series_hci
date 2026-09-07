"""Guard against silent drift in the network-level anomaly-detection figures.

Run:  python3.12 verify_network_level_results.py

Chapter 4 §4.4 of the dissertation cites average-precision figures for a 14-sensor,
10-method network-level anomaly comparison. Unlike every other headline figure in the
project, that comparison was computed in an exploratory notebook
(`notebooks/05_deep_training_techniques.ipynb`, cells 48 and 56), not in `src/`, so it
is not exercised by `verify_results.py` or the pytest suite -- see the honest disclosure
in Chapter 3 §3.10 and `docs/evaluation/02-anomaly-detection.md`.

This script does NOT retrain the models or re-run the notebook (that would require
GPU/torch and several minutes, and could drift slightly across hardware or library
versions even with fixed seeds). What it DOES do: check that the two persisted result
tables the notebook exported --
`docs/evaluation/results/network_level_ap_by_fault_type.csv` and
`network_level_ap_rematch_with_flatline.csv` -- still contain exactly the figures the
dissertation cites. That protects against a much more likely failure mode than
non-reproducible training: someone editing the CSV, regenerating it from a different
run, or the dissertation's numbers silently drifting out of sync with the artifact that
is supposed to back them.

Exits non-zero if any check fails.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "docs" / "evaluation" / "results"

checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, bool(ok), detail))


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) < tol


# ---- table 1: AP by method x fault type (cell 48) ----
by_fault = pd.read_csv(RESULTS / "network_level_ap_by_fault_type.csv", index_col="method")

check("Mahalanobis wins the spike fault at 0.931 (Ch4 §4.4: '0.93 ... F1 of 0.91')",
      close(by_fault.loc["Mahalanobis", "spike_4sigma"], 0.931))
check("Mahalanobis on drift matches 0.880 (Ch4 §4.4: 'a gradual drift (0.88)')",
      close(by_fault.loc["Mahalanobis", "drift_4sigma_24h"], 0.880))
check("Mahalanobis on stuck-sensor matches 0.642 (Ch4 §4.4: 'a single stuck sensor (0.64)')",
      close(by_fault.loc["Mahalanobis", "stuck_24h"], 0.642))
check("LOF spike matches 0.894 (Ch4 §4.4: 'a density-based Local Outlier Factor detector reaches 0.89')",
      close(by_fault.loc["LOF", "spike_4sigma"], 0.894))
check("Transformer spike matches 0.835 (Ch4 §4.4 / Ch3 §3.6: 'a deep Transformer baseline ... reaches 0.835')",
      close(by_fault.loc["Transformer", "spike_4sigma"], 0.835))
check("IsoForest spike collapses to 0.003 (Ch4 §4.4: 'collapses to an average precision of 0.003')",
      close(by_fault.loc["IsoForest", "spike_4sigma"], 0.003))
check("CNN and OC-SVM spike both round to 0.51 (Ch4 §4.4: 'both around 0.51'; "
      "fixed 2026-08-25 -- was previously misstated in the dissertation as '0.51 to 0.54', "
      "which only matched PCA, a method not part of this comparison)",
      round(by_fault.loc["CNN", "spike_4sigma"], 2) == 0.51
      and round(by_fault.loc["OC-SVM", "spike_4sigma"], 2) == 0.51)
check("GRU spike matches 0.151 (Ch4 §4.4: 'a GRU at 0.15')",
      close(by_fault.loc["GRU", "spike_4sigma"], 0.151))
check("RNN spike matches 0.054 (Ch4 §4.4: 'a plain RNN at 0.05')",
      close(by_fault.loc["RNN", "spike_4sigma"], 0.054))

# ---- table 2: the rematch, including the flatline/variance-collapse detector (cell 56) ----
rematch = pd.read_csv(RESULTS / "network_level_ap_rematch_with_flatline.csv", index_col="detector")

check("Flatline detector wins the stuck-sensor fault at 0.791 "
      "(Ch4 §4.4: 'a dedicated variance-collapse detector reaches 0.79')",
      close(rematch.loc["Flatline (var)", "stuck_24h"], 0.791))
check("Mahalanobis still beats Flatline on stuck-sensor at 0.642 vs 0.791 is FALSE -- "
      "Flatline wins that one fault type by design (portfolio rationale, Ch3 §3.6)",
      rematch.loc["Flatline (var)", "stuck_24h"] > rematch.loc["Mahalanobis", "stuck_24h"])
check("Mahalanobis and Flatline figures agree between the two persisted tables "
      "(Mahalanobis spike/drift/stuck identical in both CSVs)",
      close(by_fault.loc["Mahalanobis", "spike_4sigma"], rematch.loc["Mahalanobis", "spike_4sigma"])
      and close(by_fault.loc["Mahalanobis", "drift_4sigma_24h"], rematch.loc["Mahalanobis", "drift_4sigma_24h"])
      and close(by_fault.loc["Mahalanobis", "stuck_24h"], rematch.loc["Mahalanobis", "stuck_24h"]))

# ---- report ----
print("=" * 64)
passed = sum(ok for _, ok, _ in checks)
for name, ok, detail in checks:
    print(f"  {'✓' if ok else '✗ FAIL'}  {name}" + (f"   [{detail}]" if detail and not ok else ""))
print("=" * 64)
print(f"{passed}/{len(checks)} checks passed.")
print("\nNote: this verifies the persisted CSV artifacts match the dissertation's cited")
print("figures. It does not re-run the deep-learning training in notebooks/05_deep_")
print("training_techniques.ipynb -- see that notebook and docs/evaluation/02-anomaly-")
print("detection.md for how these numbers were originally produced.")
sys.exit(0 if passed == len(checks) else 1)
