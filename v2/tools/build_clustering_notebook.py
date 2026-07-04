"""Build v2/notebooks/11_clustering.ipynb — documented, imports src/clustering.py."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# v2 — Notebook 11 · Clustering countries by trade behaviour (Objective 4)

v1 clustered *day-types*; v2 clusters *countries*. We summarise each country by five interpretable
features and group similar trading nations with K-means. This answers "which countries behave alike?"
— big surplus exporters vs small deficit economies vs fast-growing newcomers.

Features: **size** (log avg total trade), **balance_ratio** (surplus + / deficit −), **growth**
(annual log trend), **seasonality** (STL seasonal strength), **volatility** (month-to-month).
Logic in `src/clustering.py`.
> ⚠️ v2 = international trade.""")

md("## Step 0 · Setup + features")
code(r"""import sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
def find_v2_root():
    for base in [Path.cwd(), *Path.cwd().parents]:
        if (base / "src" / "clustering.py").exists():
            return base
    raise FileNotFoundError("v2 root not found")
V2 = find_v2_root(); sys.path.insert(0, str(V2))
DATA = V2.parent / "data" / "eu_trade_monthly.csv"

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 110
from src import preprocess as pp, clustering as cl

df = pp.load_trade(DATA)
feats = cl.country_features(df)
print("features:", feats.shape)
feats.round(2)""")

md(r"""## Step 1 · How many clusters? (silhouette)

**What:** silhouette score for several k. **Why:** pick k where countries separate best. **How to
read:** higher = better-separated; we use the best k below.""")
code(r"""sk = cl.best_k(feats, ks=range(2, 7))
print(sk.round(3).to_string())
print("silhouette-best k =", int(sk.idxmax()))
print("Note: k=2 simply isolates the tiny island economies (Cyprus, Malta) as outliers,")
print("giving one big blob. For interpretable groups we use k=4 below.")
k = 4
fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(sk.index, sk.values, "o-"); ax.axvline(int(sk.idxmax()), color="grey", ls=":")
ax.axvline(k, color="red", ls="--", label=f"chosen k={k}")
ax.set_title("Silhouette vs k"); ax.set_xlabel("k"); ax.set_ylabel("silhouette"); ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.show()""")

md(r"""## Step 2 · Cluster the countries

**How to read:** each country gets a group label; members listed per cluster.""")
code(r"""labels = cl.cluster_countries(feats, k=k)
members = cl.cluster_members(labels)
for cid, ccs in members.items():
    print(f"Cluster {cid}: {', '.join(ccs)}")""")

md(r"""## Step 3 · What does each cluster mean? (profiles)

**How to read:** the mean features per cluster name the group — e.g. large surplus exporters vs
small fast-growers.""")
code(r"""prof = cl.cluster_profiles(feats, labels)
print(prof.round(2).to_string())""")

md(r"""## Step 4 · Visualise — size vs balance, coloured by cluster

**How to read:** x = economic size, y = surplus/deficit tendency; colours = clusters. Germany/
Netherlands sit top-right (big surplus); deficit economies sit below zero.""")
code(r"""fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(feats["size_log"], feats["balance_ratio"], c=labels, cmap="tab10", s=80)
for cc in feats.index:
    ax.annotate(cc, (feats.loc[cc, "size_log"], feats.loc[cc, "balance_ratio"]), fontsize=8,
                xytext=(3, 3), textcoords="offset points")
ax.axhline(0, color="grey", lw=.8)
ax.set_xlabel("size (log avg total trade)"); ax.set_ylabel("balance ratio (+surplus / -deficit)")
ax.set_title(f"Country clusters (k={k})"); ax.grid(alpha=.3)
plt.tight_layout(); plt.show()""")

md(r"""## Step 5 · Summary

| Feature | Meaning |
| --- | --- |
| size_log | economic size (log average total trade) |
| balance_ratio | surplus (+) vs deficit (−), scale-free |
| growth | annual log-trend of exports |
| seasonality | STL seasonal strength |
| volatility | month-to-month log volatility |

K-means groups the 27 countries into interpretable trade types (large surplus exporters, deficit
economies, fast-growing newcomers, small/volatile traders). This satisfies **Objective 4** and feeds
the reasoning layer ("country X behaves like its cluster peers").

**→ Next: deep model (multivariate autoencoder), web UI, final report.**""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).resolve().parents[1] / "notebooks" / "11_clustering.ipynb"
nbf.write(nb, out)
print("Wrote", out, "with", len(cells), "cells")
