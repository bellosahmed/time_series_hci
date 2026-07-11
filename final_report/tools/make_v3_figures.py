#!/usr/bin/env python3.12
"""v3 (IoT network security / DAD) — five figures from the REAL data.

Run from the v3 directory:  cd code/v3 && python3.12 ../final_report/tools/make_v3_figures.py
Figures 1–3 are computed from the parsed pcap dataframe (scapy, via v3's own
preprocess). Figures 4–5 visualise the verified evaluation numbers from
v3/report/EVALUATION.md and final_report/03_results_and_findings.md.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath("."))                       # v3/ (for `src`)
sys.path.insert(0, str(Path(__file__).resolve().parent))       # tools/ (for _figstyle)

import numpy as np
import pandas as pd
from _figstyle import AQUA, BLUE, GRID, INK, INK2, MUTED, ORANGE, SURFACE, caption, grid, save
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src import preprocess

OUT = (Path(__file__).resolve().parent.parent / "figures" / "v3")
OUT.mkdir(parents=True, exist_ok=True)

DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
DAYLAB = {"mon": "Mon", "tue": "Tue", "wed": "Wed", "thu": "Thu",
          "fri": "Fri", "sat": "Sat", "sun": "Sun"}
TYPE_COL = {"modification": BLUE, "duplication": ORANGE, "mixed": AQUA}

print("parsing DAD pcaps (scapy) ...")
df = preprocess.load_week()
attacks = df[df["label"] == 1]
print(f"  {len(df):,} packets, {len(attacks)} labelled attacks")


# 1 — labelled attack packets per day, coloured by attack type
def f1():
    piv = (attacks.groupby(["day", "attack_type"]).size()
           .unstack(fill_value=0).reindex(DAYS, fill_value=0))
    types = [t for t in ["modification", "duplication", "mixed"] if t in piv.columns]
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    bottom = np.zeros(len(DAYS))
    for t in types:
        vals = piv[t].values
        ax.bar(range(len(DAYS)), vals, bottom=bottom, color=TYPE_COL[t],
               zorder=3, label=t)
        bottom += vals
    for i, tot in enumerate(bottom):
        if tot > 0:
            ax.text(i, tot + 4, int(tot), ha="center", va="bottom", color=INK2, fontsize=9)
    ax.set_xticks(range(len(DAYS))); ax.set_xticklabels([DAYLAB[d] for d in DAYS])
    ax.set_ylabel("Labelled attack packets")
    ax.set_title("v3 — the real attack schedule (544 packets, matches the paper)", color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    grid(ax)
    caption(fig, "Wed = modification, Thu = duplication, Fri/Sat = mixed; Mon/Tue/Sun clean. This exact "
                 "schedule is what verify_results.py checks against the paper.")
    save(fig, OUT / "v3_1_attacks_per_day.png")


# 2 — Wednesday temperatures: modification tampering shows in the value channel
def f2():
    wed = df[(df["day"] == "wed") & df["temperature"].notna()].copy()
    normal = wed[wed["label"] == 0]
    tamper = wed[wed["label"] == 1]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.scatter(normal.index, normal["temperature"], s=8, color=BLUE, alpha=0.5,
               zorder=2, label="normal readings")
    ax.scatter(tamper.index, tamper["temperature"], s=26, color=ORANGE, zorder=3,
               edgecolor="white", lw=0.4, label=f"modification attack ({len(tamper)})")
    ax.set_ylabel("Reported temperature (°C)")
    ax.set_title("v3 — modification attacks distort the value channel (Wednesday)", color=INK)
    leg = ax.legend(fontsize=9, loc="upper right", framealpha=0.92)
    leg.get_frame().set_facecolor(SURFACE); leg.get_frame().set_edgecolor("none")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    grid(ax)
    caption(fig, "Tampered messages carry temperatures that leave the normal sensor band — which is why a "
                 "simple z-score (the value channel) catches modification but the flow channel cannot.")
    save(fig, OUT / "v3_2_modification_value_channel.png")


# 3 — the honest reason the flow channel is weak: attack and normal windows
#     overlap almost completely on the flow features (5-min windows).
def f3():
    from src import features
    ff = features.flow_features(df, window="5min").join(
        features.window_labels(df, window="5min"))
    normal = ff[ff["label"] == 0]["msg_rate"].clip(upper=200)
    atk = ff[ff["label"] == 1]["msg_rate"].clip(upper=200)
    bins = np.linspace(0, 200, 41)
    fig, ax = plt.subplots(figsize=(8.6, 3.7))
    ax.hist(normal, bins=bins, density=True, color=BLUE, alpha=0.55, zorder=3,
            label=f"normal windows ({len(normal)})")
    ax.hist(atk, bins=bins, density=True, histtype="step", color=ORANGE, lw=2.2,
            zorder=4, label=f"attack windows ({len(atk)})")
    ax.set_xlabel("Messages per 5-min window (flow-channel feature)")
    ax.set_ylabel("Density")
    ax.set_title("v3 — why the flow channel is weak: attacks hide in normal traffic", color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    grid(ax)
    caption(fig, "Attack and normal windows sit on almost the same distribution (attack msg-rate median "
                 "50 vs 50 normal); only a handful of duplication windows surge — which is exactly why "
                 "fusion lifts recall but not precision.")
    save(fig, OUT / "v3_3_flow_channel_weakness.png")


# 4 — per-attack-type recall: the two channels are complementary
#     (verified numbers, v3/report/EVALUATION.md M3 table)
def f4():
    types = ["Modification", "Duplication", "Mixed"]
    value = [0.46, 0.50, 0.43]
    flow = [0.00, 1.00, 0.17]
    x = range(len(types)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    b1 = ax.bar([i - w / 2 for i in x], value, w, color=BLUE, zorder=3,
                label="Value channel (temperature)")
    b2 = ax.bar([i + w / 2 for i in x], flow, w, color=ORANGE, zorder=3,
                label="Flow channel (network traffic)")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                    f"{b.get_height():.2f}", ha="center", va="bottom", color=INK2, fontsize=9)
    ax.set_xticks(list(x)); ax.set_xticklabels(types)
    ax.set_ylabel("Recall (per attack type)"); ax.set_ylim(0, 1.12)
    ax.set_title("v3 — the two channels are complementary, not redundant", color=INK)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    grid(ax)
    caption(fig, "Flow is blind to modification (0.00) but perfect on duplication (1.00); value has the "
                 "opposite tilt — each channel sees attacks the other cannot.")
    save(fig, OUT / "v3_4_channel_complementarity.png")


# 5 — average precision with 95% bootstrap CIs (verified numbers)
def f5():
    configs = ["flow-only", "supervised\n(day-blocked)", "early-fusion", "late-fusion", "value-only"]
    ap = [0.136, 0.101, 0.147, 0.171, 0.183]
    lo = [0.077, 0.075, 0.108, 0.123, 0.136]
    hi = [0.222, 0.141, 0.206, 0.246, 0.245]
    fig, ax = plt.subplots(figsize=(7.6, 3.7))
    for i, (a, l, h) in enumerate(zip(ap, lo, hi)):
        best = configs[i] == "value-only"
        worst = configs[i].startswith("supervised")
        col = BLUE if best else (ORANGE if worst else MUTED)
        lab = f"{a:.3f}" + ("  ← best" if best else ("  ← worst (learned)" if worst else ""))
        ax.plot([l, h], [i, i], color=col, lw=2.4, solid_capstyle="round", zorder=2)
        ax.scatter([a], [i], s=70, color=col, zorder=3, edgecolor=SURFACE, linewidth=1.2)
        ax.text(h + 0.006, i, lab, va="center", ha="left",
                color=(col if (best or worst) else INK2), fontsize=9,
                fontweight=("bold" if (best or worst) else "normal"))
    ax.set_yticks(range(len(configs))); ax.set_yticklabels(configs)
    ax.set_xlabel("Average precision  (dot = estimate, bar = 95% bootstrap CI)")
    ax.set_xlim(0.05, 0.30)
    ax.set_title("v3 — no detector config is significantly best on AP", color=INK, pad=12)
    grid(ax, axis="x")
    caption(fig, "Every interval overlaps its neighbours — the honest read is a tie, not a winner; the one "
                 "clear signal is that the learned combiner is worst.")
    save(fig, OUT / "v3_5_ap_confidence_intervals.png")


if __name__ == "__main__":
    print("rendering ->", OUT)
    f1(); f2(); f3(); f4(); f5()
    print("v3 done.")
