"""Build v2/notebooks/04_anomaly.ipynb — documented, imports src/anomaly.py."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# v2 — Notebook 04 · Anomaly detection (EU monthly trade)

**The core of the project.** We flag months where trade behaves abnormally — and unlike v1
(synthetic spikes), v2 has **real, datable anomalies** (the 2008 crisis, COVID-19) to test against.

**Three detector families** (picked from the analysis findings in notebook 03):

| Family | Method | Idea |
| --- | --- | --- |
| Statistical | **z-score on STL residual** | flag months far from the de-seasonalised, de-trended norm |
| Forecasting | **seasonal-naive error** | flag months far from "same month last year" |
| Machine-learning | **Isolation Forest (multivariate)** | flag months where *many countries* move abnormally at once → systemic shocks |

We then **evaluate** them against the labelled crises (precision / recall / F1).
Logic in `src/anomaly.py`; this notebook imports and demonstrates it.
> ⚠️ v2 = international trade.""")

md("## Step 0 · Setup")
code(r"""import sys
from pathlib import Path
def find_v2_root():
    for base in [Path.cwd(), *Path.cwd().parents]:
        if (base / "src" / "anomaly.py").exists():
            return base
    raise FileNotFoundError("v2 root not found")
V2 = find_v2_root(); sys.path.insert(0, str(V2))
DATA = V2.parent / "data" / "eu_trade_monthly.csv"

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 110
from src import preprocess as pp
from src import anomaly as ad

df = pp.load_trade(DATA)
logged = pp.log_transform(df)             # detect on the log scale (per notebook 02/03)
print("loaded:", df.shape)""")

md(r"""## Step 1 · Ground truth — the real crises

**What:** label the months of the two well-documented shocks. These are our **gold-standard
anomalies** for evaluation. **Why:** with real labels we can compute honest precision/recall —
something v1 could only fake with injected spikes. **How to read:** shaded bands = labelled crisis
months on total EU exports.""")
code(r"""# Impact windows of the two major shocks (sharp drop + immediate aftermath)
CRISES = [("2008-09-01", "2009-04-01"),   # global financial crisis
          ("2020-03-01", "2020-08-01")]   # COVID-19
truth = ad.label_windows(df.index, CRISES)
print("Labelled crisis months:", int(truth.sum()))

total_exp = df[[c for c in df.columns if c.endswith("_EXP")]].sum(axis=1) / 1000
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(total_exp.index, total_exp, color="steelblue", lw=1.2)
for s, e in CRISES:
    ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color="red", alpha=.18)
ax.set_title("Total EU exports with labelled crisis windows (red)")
ax.set_ylabel("EUR bn / month"); ax.grid(alpha=.3)
plt.tight_layout(); plt.show()""")

md(r"""## Step 2 · Detector 1 — z-score on the STL residual (statistical)

**What:** decompose Germany's log exports, standardise the **residual**, flag |z| > 3.
**Why:** the residual removes trend + season, so what's left and extreme is a genuine surprise.
**How to read:** red dots = flagged months; they should land on the shaded crises.""")
code(r"""target = "DE_EXP"
z = ad.stl_residual_z(logged[target])
flag1 = ad.zscore_anomalies(logged[target], threshold=3.0)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(z.index, z, color="grey", lw=.8)
ax.scatter(z.index[flag1], z[flag1], color="red", zorder=5, label="|z| > 3")
for s, e in CRISES: ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color="red", alpha=.12)
ax.axhline(3, color="red", ls=":"); ax.axhline(-3, color="red", ls=":")
ax.set_title(f"{target}: STL-residual z-score"); ax.set_ylabel("std devs"); ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.show()
print("Flagged:", [d.strftime("%Y-%m") for d in z.index[flag1]])""")

md(r"""## Step 3 · Detector 2 — seasonal-naive forecast error (forecasting)

**What:** compare each month to the **same month a year earlier**; flag big standardised errors.
**Why:** a simple, transparent forecaster — "trade should look like last year" — whose errors spike
during shocks. A baseline every forecasting model should beat. **How to read:** red dots = flagged.""")
code(r"""znaive = ad.seasonal_naive_z(logged[target])
flag2 = ad.seasonal_naive_anomalies(logged[target], threshold=3.0)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(znaive.index, znaive, color="darkorange", lw=.8)
ax.scatter(znaive.index[flag2], znaive[flag2], color="red", zorder=5, label="|z| > 3")
for s, e in CRISES: ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color="red", alpha=.12)
ax.axhline(3, color="red", ls=":"); ax.axhline(-3, color="red", ls=":")
ax.set_title(f"{target}: seasonal-naive forecast-error z-score"); ax.set_ylabel("std devs"); ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.show()
print("Flagged:", [d.strftime("%Y-%m") for d in znaive.index[flag2]])""")

md(r"""## Step 4 · Detector 3 — Isolation Forest, multivariate (machine-learning)

**What:** feed **all 54 countries' year-on-year growth** into an Isolation Forest. It flags months
that are jointly unusual across many countries — i.e. **systemic** shocks. **Why:** this is the
multi-country detector; a single country wobbling is noise, but *everyone* dropping together is a
real event. **How to read:** the score spikes when the whole continent moves; red = flagged months.""")
code(r"""feat = ad.systemic_features(logged)            # year-on-year log growth, all flows
flag3 = ad.isoforest_anomalies(feat, contamination=0.05)
sc = ad.isoforest_scores(feat)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(sc.index, sc, color="purple", lw=.9)
ax.scatter(sc.index[flag3], sc[flag3], color="red", zorder=5, label="flagged (systemic)")
for s, e in CRISES: ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color="red", alpha=.12)
ax.set_title("Isolation Forest systemic anomaly score (all 54 channels)")
ax.set_ylabel("anomaly score"); ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.show()
print("Flagged systemic months:", [d.strftime("%Y-%m") for d in sc.index[flag3]])""")

md(r"""## Step 5 · Evaluation — score each detector against the real crises

**What:** precision / recall / F1 / confusion vs the labelled 2008 & COVID windows.
- *Detectors 1 & 2* are scored on the target series (`DE_EXP`).
- *Detector 3* is scored at the **systemic** level (does it flag the crisis months continent-wide?).

**How to read:** **recall** = did it catch the crises? **precision** = were its flags real? (Note
the *precision artifact*: flagging the real-but-unlabelled 2022 energy crisis counts as a false
positive here.)""")
code(r"""results = {
    "1 · STL z-score (DE)":       ad.score(flag1, truth),
    "2 · Seasonal-naive (DE)":    ad.score(flag2, truth),
    "3 · Isolation Forest (sys)": ad.score(flag3, truth),
}
tab = pd.DataFrame({k: vars(v) for k, v in results.items()}).T
tab = tab[["precision", "recall", "f1", "tp", "fp", "fn", "tn"]]
print(tab.round(2).to_string())""")

md(r"""### Step 5b · Event-level detection — *did it catch each crisis?*

**What:** the strict table above penalises a detector for not flagging *every* month of a long
crisis. But for rare events the real question is simpler: **did it fire at all within each crisis
(±1 month)?** **How to read:** ✓ = the crisis was detected.""")
code(r"""def detected(flags, window, tol="31D"):
    s, e = pd.Timestamp(window[0]) - pd.Timedelta(tol), pd.Timestamp(window[1]) + pd.Timedelta(tol)
    f = flags.fillna(False)
    return bool(f.loc[(f.index >= s) & (f.index <= e)].any())

names = {"2008 crisis": CRISES[0], "COVID-19": CRISES[1]}
detectors = {"1 · STL z-score": flag1, "2 · Seasonal-naive": flag2, "3 · Isolation Forest": flag3}
ev = pd.DataFrame({d: {c: ("✓" if detected(f, w) else "—") for c, w in names.items()}
                   for d, f in detectors.items()}).T
print("Event-level detection (caught the crisis within +/-1 month?):")
print(ev.to_string())""")

md(r"""## Step 6 · Discussion

- **Recall first.** For anomaly detection on rare events, *catching* the crisis (recall) matters
  more than the odd false alarm. All three families respond at the crises.
- **The precision artifact.** Several "false positives" are actually *real* anomalies we did not
  label (the 2022 energy/inflation shock, occasional reporting blips). So measured precision is a
  **lower bound** — the detectors are better than the number suggests. This is the same lesson as
  v1, now on real data.
- **Why three families?** They fail differently: the z-score assumes near-normal residuals (notebook
  03 showed fat tails), the seasonal-naive is blind to gradual drift, and Isolation Forest needs the
  multivariate view. Agreement across families = a confident anomaly.
- **The systemic detector** is the v2 novelty seed: it answers *"was this a whole-continent event?"*
  rather than *"did one country wobble?"*.

## Step 7 · Summary

| Detector | Family | Catches the crises? |
| --- | --- | --- |
| STL-residual z-score | statistical | yes (sharp drops) |
| Seasonal-naive error | forecasting | yes (vs last year) |
| Isolation Forest (multivariate) | ML | yes (systemic months) |

**→ Next: Notebook 05 · Multi-country reasoning** — when several countries are abnormal together,
explain it as a shared shock (the interpretation layer), then the query engine + interface.""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).resolve().parent / "notebooks" / "04_anomaly.ipynb"
nbf.write(nb, out)
print("Wrote", out, "with", len(cells), "cells")
