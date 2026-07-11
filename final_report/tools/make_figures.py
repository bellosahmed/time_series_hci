#!/usr/bin/env python3.12
"""Cross-cutting figure for final_report/SUMMARY.md.

The per-version galleries are rendered by make_v1_figures.py / make_v2_figures.py
/ make_v3_figures.py (run from each version directory). This file renders the one
cross-version chart that belongs to no single version: the simple-vs-complex
comparison. Numbers are copied from 03_results_and_findings.md — nothing new is
computed. Re-run: `python3.12 tools/make_figures.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # tools/ (for _figstyle)

from _figstyle import BLUE, INK, INK2, MUTED, ORANGE, grid, save
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True)


# Simple/classical vs deep/learned, once per version.
# Sources: 03_results (v1 controlled-signal F1, v2 strict F1, v3 AP).
def fig_simple_vs_complex():
    panels = [
        ("v1 — Weather", "F1 (controlled signal)",
         ["Isolation\nForest", "ARIMA", "Deep\nauto-encoder"],
         [1.00, 0.99, 0.95], [BLUE, BLUE, ORANGE], 1.15),
        ("v2 — Trade", "Strict F1 (real labels)",
         ["Isolation\nForest", "STL\nz-score", "Multivariate\nauto-encoder"],
         [0.36, 0.27, 0.21], [BLUE, BLUE, ORANGE], 0.46),
        ("v3 — Network", "Average precision",
         ["Value-only\n(z-score)", "Late\nfusion", "Supervised\ncombiner"],
         [0.183, 0.171, 0.101], [BLUE, BLUE, ORANGE], 0.24),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.7))
    for ax, (title, ylab, labels, vals, cols, top) in zip(axes, panels):
        bars = ax.bar(range(len(vals)), vals, color=cols, width=0.66, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + top * 0.02, f"{v:.2f}",
                    ha="center", va="bottom", color=INK2, fontsize=9)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_ylim(0, top)
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_title(title, color=INK, fontsize=11)
        grid(ax)
    fig.suptitle("Deep / learned methods (orange) never beat the simpler ones (blue)",
                 color=INK, fontsize=12.5, fontweight="bold", y=1.03)
    fig.text(0.5, -0.02,
             "Three domains, three metrics — the same result each time: on small, seasonal, "
             "label-scarce data the flexible model over-fits.",
             ha="center", color=MUTED, fontsize=8.5)
    fig.tight_layout()
    save(fig, OUT / "cross_1_simple_vs_complex.png")


if __name__ == "__main__":
    print("rendering ->", OUT)
    fig_simple_vs_complex()
    print("done.")
