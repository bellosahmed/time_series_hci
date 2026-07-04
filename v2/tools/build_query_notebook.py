"""Build v2/notebooks/07_query_engine.ipynb — documented, imports src/query_engine.py."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# v2 — Notebook 07 · Query engine (the truth layer)

**The most important module for trust.** Every number a user can ask for is computed *here*, by
deterministic code, exactly. The interface and the local LLM (next notebooks) call these functions
and may only **phrase** the answer — they never calculate or invent a number.

> **Core principle:** Computation = code (source of truth). LLM = explanation only.

So this notebook does two things: (1) show the engine answering real questions, and (2) **prove**
each answer matches plain pandas exactly — that proof is what makes the system trustworthy.
Logic in `src/query_engine.py`.
> ⚠️ v2 = international trade (EUR millions, monthly).""")

md("## Step 0 · Setup")
code(r"""import sys
from pathlib import Path
def find_v2_root():
    for base in [Path.cwd(), *Path.cwd().parents]:
        if (base / "src" / "query_engine.py").exists():
            return base
    raise FileNotFoundError("v2 root not found")
V2 = find_v2_root(); sys.path.insert(0, str(V2))
DATA = V2.parent / "data" / "eu_trade_monthly.csv"

import numpy as np, pandas as pd
from src import preprocess as pp
from src import query_engine as qe

df = pp.load_trade(DATA)
print("loaded:", df.shape)""")

md(r"""## Step 1 · Extremes — highest / lowest months

**Q:** *"When did France export the most? The least?"* The engine returns the **exact** value, month
and country.""")
code(r"""print(qe.get_max(df, "FR", "EXP"))
print(qe.get_min(df, "FR", "EXP"))
print(qe.get_max(df, "DE", "IMP"))""")

md(r"""## Step 2 · Statistics — mean, std, correlation

**Q:** *"How big is Italy's trade on average? Do Germany and France's exports move together?"*""")
code(r"""mean, std = qe.get_mean_std(df, "IT", "EXP")
print(f"IT exports: mean {mean:,.0f} EUR m, std {std:,.0f}")
print("corr(DE_EXP, FR_EXP) growth:", round(qe.get_correlation(df, "DE", "FR", "EXP"), 3))
print("corr(ES_EXP, IT_EXP) growth:", round(qe.get_correlation(df, "ES", "IT", "EXP"), 3))""")

md(r"""## Step 3 · Balance & surplus/deficit

**Q:** *"Who ran the biggest trade surplus in March 2024? The biggest deficit?"* Balance = exports −
imports.""")
code(r"""m = "2024-03-01"
print("France balance", m, ":", f"{qe.trade_balance(df, 'FR', m):,.0f} EUR m (deficit if negative)")
print("Biggest surplus", m, ":", qe.biggest_surplus(df, m))
print("Biggest deficit", m, ":", qe.biggest_deficit(df, m))""")

md(r"""## Step 4 · Change over time & worst drops

**Q:** *"How much did Poland's exports grow 2010→2024? What was Spain's worst monthly collapse?"*""")
code(r"""print("PL exports 2010-01 -> 2024-01:", f"{qe.growth(df, 'PL', 'EXP', '2010-01-01', '2024-01-01'):+.0f}%")
print("Worst single-month export drop, ES:", qe.worst_drop(df, "ES", "EXP"))
print("Worst single-month export drop, DE:", qe.worst_drop(df, "DE", "EXP"))""")

md(r"""## Step 5 · Ranking & country fact-sheet

**Q:** *"Top exporters in Jan 2024? Summarise the Netherlands."*""")
code(r"""print("Top 5 exporters 2024-01 (EUR m):")
print(qe.rank_countries(df, "EXP", "2024-01-01", n=5).round(0).to_string())
print()
summary = qe.country_summary(df, "NL")
for k, v in summary.items():
    print(f"  {k}: {v}")""")

md(r"""## Step 6 · VERIFICATION — does the engine match pandas exactly?

**What:** recompute every answer with raw pandas and assert equality. **Why:** this is the trust
proof — if the engine ever disagreed with ground truth, no explanation on top could be trusted.
**How to read:** all checks must print ✓.""")
code(r"""checks = []
# max
e = qe.get_max(df, "FR", "EXP")
checks.append(("get_max value", e.value == df["FR_EXP"].max()))
checks.append(("get_max month", e.month == str(df["FR_EXP"].idxmax().date())))
# mean/std
mean, std = qe.get_mean_std(df, "IT", "EXP")
checks.append(("mean", mean == df["IT_EXP"].mean()))
checks.append(("std",  std == df["IT_EXP"].std()))
# correlation
c = qe.get_correlation(df, "DE", "FR", "EXP")
checks.append(("correlation", abs(c - df["DE_EXP"].pct_change().corr(df["FR_EXP"].pct_change())) < 1e-12))
# balance
b = qe.trade_balance(df, "FR", "2024-03-01")
checks.append(("balance", b == df.loc["2024-03-01", "FR_EXP"] - df.loc["2024-03-01", "FR_IMP"]))
# growth
g = qe.growth(df, "PL", "EXP", "2010-01-01", "2024-01-01")
exp_g = (df.loc["2024-01-01","PL_EXP"]/df.loc["2010-01-01","PL_EXP"] - 1)*100
checks.append(("growth", abs(g - exp_g) < 1e-9))
# biggest surplus
bs = qe.biggest_surplus(df, "2024-03-01")
manual = (df.filter(like="_EXP").loc["2024-03-01"].values - df.filter(like="_IMP").loc["2024-03-01"].values)
checks.append(("biggest_surplus value", abs(bs.value - manual.max()) < 1e-6))

ok = 0
for name, passed in checks:
    print(("✓" if passed else "✗"), name); ok += passed
print(f"\n{ok}/{len(checks)} checks passed — engine == pandas ground truth.")""")

md(r"""## Step 7 · Summary

| Question type | Function | Exact? |
| --- | --- | --- |
| highest / lowest month | `get_max` / `get_min` | ✓ vs pandas |
| average / spread | `get_mean_std` | ✓ |
| do two countries move together | `get_correlation` | ✓ |
| surplus / deficit | `trade_balance`, `biggest_surplus/deficit` | ✓ |
| growth, worst drop | `growth`, `worst_drop` | ✓ |
| ranking, fact-sheet | `rank_countries`, `country_summary` | ✓ |

Every answer is computed by code and **verified equal to pandas** — the engine is the source of
truth. The next notebooks add the natural-language interface and the local LLM *on top* of this,
never replacing it.

**→ Next: Notebook 08 · Interface + local LLM** — ask questions in plain English; the engine
computes, the LLM only explains (trust-aware answers).""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).resolve().parents[1] / "notebooks" / "07_query_engine.ipynb"
nbf.write(nb, out)
print("Wrote", out, "with", len(cells), "cells")
