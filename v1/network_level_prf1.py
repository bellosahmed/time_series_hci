"""Extend the reproducibility check in verify_network_level_from_checkpoints.py with precision,
recall, and F1 at each method's best-F1 operating point on the spike-fault (4-sigma) PR curve,
plus a bar-chart figure for the dissertation appendix.

Average precision (already reported in Chapter 4 and Appendix C) summarises a method's whole
precision-recall curve into one number. It does not by itself say what precision or recall looks
like at any one decision threshold. This script picks, for each method independently, the
threshold on its own PR curve that maximises F1, and reports precision/recall/F1 there. This is
the standard way to turn an AP-style ranking metric into an actual confusion-matrix comparison,
and it is fair across methods with very different score distributions because each method's
threshold is chosen relative to its own curve, not a single fixed score cutoff shared across
architectures with incomparable scales.

Run:  python3.12 network_level_prf1.py   (from the v1 project root, with saved_models_05/ present)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

ROOT = Path(__file__).resolve().parent
V1 = Path(__file__).resolve().parent
sys.path.insert(0, str(V1))
from src import config, preprocess  # noqa: E402

# reuse the exact architecture builders from the checkpoint-reproduction script
sys.path.insert(0, str(V1))
import importlib.util
spec = importlib.util.spec_from_file_location("vnlfc", V1 / "verify_network_level_from_checkpoints.py")
vnlfc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vnlfc)

MODELS_DIR = V1 / "notebooks" / "saved_models_05"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Chapter 5 SS5.2 / Appendix C note the CNN checkpoint's forward pass is not deterministic under
# cuDNN's default algorithm-selection. Fix it here too, so this script's CNN figure matches the
# 0.51 average precision already cited, rather than landing anywhere in the 0.39-0.54 spread.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("Loading the real Jena Climate record...")
df = preprocess.align(preprocess.clean(preprocess.load_data(config.DATA_PATH)), freq="1h")
print(f"  {df.shape[0]} hourly readings x {df.shape[1]} sensors")

ckpt = np.load(MODELS_DIR / "standardiser.npz", allow_pickle=True)
mu, sd, window = ckpt["mu"], ckpt["sd"], int(ckpt["window"])

models = {}
for name, builder in vnlfc.BUILDERS.items():
    m = builder(window, len(mu))
    m.load_state_dict(torch.load(MODELS_DIR / f"{name}.pt", map_location=DEVICE))
    m.to(DEVICE).eval()
    models[name] = m
print("  reloaded 5 deep checkpoints:", list(models))

Z = vnlfc.apply_standardise(df, mu, sd)

print("Refitting the classical detectors on the same clean, standardised data...")
_pca = PCA(n_components=5).fit(Z)
_cov = LedoitWolf().fit(Z)
_iso = IsolationForest(n_estimators=200, random_state=0).fit(Z)
_sub = np.random.default_rng(0).choice(len(Z), 5000, replace=False)
_lof = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(Z[_sub])
_ocs = OneClassSVM(nu=0.05, gamma="scale").fit(Z[_sub])


def classical_scores(Zn):
    rec = _pca.inverse_transform(_pca.transform(Zn))
    return {"PCA": ((Zn - rec) ** 2).sum(1),
            "Mahalanobis": _cov.mahalanobis(Zn),
            "IsoForest": -_iso.score_samples(Zn),
            "LOF": -_lof.score_samples(Zn),
            "OC-SVM": -_ocs.decision_function(Zn)}


print("Re-running the spike (4-sigma) fault injection, same seed as the notebook (42)...")
pn, pl, _ = vnlfc.inject_sensor_faults(df, n=100, magnitude=4.0, seed=42)
Zn = vnlfc.apply_standardise(pn, mu, sd)

all_scores = {}
for name, m in models.items():
    all_scores[name] = vnlfc.network_score(vnlfc.recon_error_matrix(m, Zn, window))
for name, sc in classical_scores(Zn).items():
    all_scores[name] = sc

rows = []
for name, sc in all_scores.items():
    ap = average_precision_score(pl, sc)
    prec, rec, thr = precision_recall_curve(pl, sc)
    f1 = np.where((prec + rec) > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    best = int(np.argmax(f1[:-1])) if len(f1) > 1 else 0
    rows.append({"method": name, "average_precision": round(ap, 3),
                 "precision": round(float(prec[best]), 3),
                 "recall": round(float(rec[best]), 3),
                 "f1": round(float(f1[best]), 3)})

table = pd.DataFrame(rows).sort_values("average_precision", ascending=False).reset_index(drop=True)
pd.set_option("display.width", 160)
print()
print("Precision / recall / F1 at each method's best-F1 threshold (spike, 4-sigma fault):")
print(table.to_string(index=False))

out_csv = V1 / "docs" / "evaluation" / "results" / "network_level_prf1_spike.csv"
table.to_csv(out_csv, index=False)
print(f"\nsaved {out_csv}")

# --- figure: grouped bar chart, AP / precision / recall / F1 per method ---
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(table))
w = 0.2
metrics = ["average_precision", "precision", "recall", "f1"]
labels = ["Average precision", "Precision", "Recall", "F1"]
colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2"]
for i, (met, lab, col) in enumerate(zip(metrics, labels, colors)):
    ax.bar(x + (i - 1.5) * w, table[met], width=w, label=lab, color=col)
ax.set_xticks(x)
ax.set_xticklabels(table["method"], rotation=30, ha="right")
ax.set_ylabel("Score")
ax.set_title("Network-level anomaly detection: 10-method comparison, spike (4-sigma) fault")
ax.legend()
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
fig_path = Path(__file__).resolve().parents[2] / "Report" / "approved_chapters" / "figures" / "fig_network_level_prf1.png"
plt.savefig(fig_path, dpi=150)
print(f"saved {fig_path}")
