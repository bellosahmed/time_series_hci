#!/usr/bin/env python3.12
"""v1 (Weather / Jena) — five figures from the REAL data.

Run from the v1 directory:  cd code/v1 && python3.12 ../final_report/tools/make_v1_figures.py
Loads the Jena CSV through v1's own preprocess (clean + hourly align); every plot
is computed from that dataframe. No numbers are invented.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath("."))                       # v1/ (for `src`)
sys.path.insert(0, str(Path(__file__).resolve().parent))       # tools/ (for _figstyle)

import numpy as np
import pandas as pd
from _figstyle import BLUE, GRID, INK, INK2, ORANGE, caption, grid, save
import matplotlib.pyplot as plt

from src import preprocess

OUT = (Path(__file__).resolve().parent.parent / "figures" / "v1")
OUT.mkdir(parents=True, exist_ok=True)
T = "T (degC)"

print("loading Jena ...")
df = preprocess.load_data("../data/jena_climate_2009_2016.csv")
df = preprocess.clean(df)
df = preprocess.align(df, "1h")
t = df[T]
print(f"  {df.shape[0]} hourly rows, {df.shape[1]} sensors")


# 1 — full-series seasonality (daily mean across 2009–2016)
def f1():
    daily = t.resample("D").mean()
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(daily.index, daily.values, color=BLUE, lw=0.8)
    ax.set_title("v1 — eight years of temperature: a strong yearly cycle", color=INK)
    ax.set_ylabel("Temperature (°C), daily mean")
    grid(ax)
    caption(fig, "Every winter–summer swing repeats — the seasonal signal STL later isolates as the "
                 "'expected' component.")
    save(fig, OUT / "v1_1_seasonality.png")


# 2 — monthly climatology (mean per calendar month, ±1 std band)
def f2():
    g = t.groupby(t.index.month)
    mean, std = g.mean(), g.std()
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    x = range(1, 13)
    ax.fill_between(x, mean - std, mean + std, color=BLUE, alpha=0.18, zorder=2,
                    label="± 1 std (year-to-year spread)")
    ax.plot(x, mean.values, color=BLUE, lw=2, marker="o", ms=5, zorder=3,
            label="Mean")
    ax.set_xticks(list(x))
    ax.set_xticklabels(months)
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("v1 — average temperature by calendar month", color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    grid(ax)
    caption(fig, "The seasonal shape is highly regular (~ -1 °C in Jan to ~19 °C in Jul); the band shows "
                 "how much any single year departs.")
    save(fig, OUT / "v1_2_monthly_climatology.png")


# 3 — distribution of hourly temperature
def f3():
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.hist(t.dropna().values, bins=60, color=BLUE, zorder=3)
    ax.axvline(t.mean(), color=ORANGE, lw=2, zorder=4,
               label=f"mean {t.mean():.1f} °C")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Hours (count)")
    ax.set_title("v1 — distribution of hourly temperature (2009–2016)", color=INK)
    ax.legend(frameon=False, fontsize=9)
    grid(ax)
    caption(fig, f"Roughly bell-shaped but left-skewed by cold snaps; range "
                 f"{t.min():.1f} to {t.max():.1f} °C over {int(t.notna().sum()):,} hours.")
    save(fig, OUT / "v1_3_distribution.png")


# 4 — correlation heatmap across the 14 sensors
def f4():
    corr = df.corr()
    labels = [c.split(" (")[0] for c in corr.columns]
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("v1 — how the 14 sensors move together", color=INK, pad=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Pearson correlation", color=INK2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    caption(fig, "Temperature, dew point and vapour-pressure form a tight warm block; density (rho) is "
                 "strongly anti-correlated — physics the reasoning layer uses to explain anomalies.")
    save(fig, OUT / "v1_4_sensor_correlation.png")


# 5 — anomaly-detection demo: inject synthetic spikes (v1's real eval method),
#     then show the rolling z-score detector recovering them.
def f5():
    sl = t["2014-07-01":"2014-07-31"].copy()
    rng = np.random.default_rng(0)
    inject_idx = sl.index[rng.choice(len(sl), size=4, replace=False)]
    signs = [1, -1, 1, -1]
    for ts, s in zip(sorted(inject_idx), signs):
        sl.loc[ts] = sl.loc[ts] + s * rng.uniform(18, 24)     # a clear out-of-envelope spike

    # Robust detector: residual vs a centred rolling median, thresholded on a
    # MAD-scaled sigma (the median baseline is not dragged by a single spike, so
    # the anomaly does not mask itself the way a rolling-mean/std would).
    baseline = sl.rolling(25, center=True, min_periods=6).median()
    resid = sl - baseline
    sigma = 1.4826 * (resid - resid.median()).abs().median()
    detected = sl[resid.abs() > 3.5 * sigma]
    hits = sum(ts in set(detected.index) for ts in inject_idx)

    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(sl.index, sl.values, color=BLUE, lw=1.0, zorder=2, label="Temperature (with injected spikes)")
    ax.scatter(sorted(inject_idx), [sl.loc[ts] for ts in sorted(inject_idx)],
               facecolor="none", edgecolor=INK2, s=150, lw=1.6, zorder=3,
               label=f"injected anomaly ({len(inject_idx)})")
    ax.scatter(detected.index, detected.values, color=ORANGE, s=42, zorder=4,
               edgecolor="white", linewidth=0.7, label=f"detected ({len(detected)})")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("v1 — detection is measured on injected anomalies (Jena has no real labels)", color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=1)
    grid(ax)
    caption(fig, f"With no ground truth, v1 injects known spikes and checks recovery: a residual-vs-median "
                 f"detector recovers {hits}/{len(inject_idx)} here — the honest, reproducible way to score "
                 f"a label-free series.")
    save(fig, OUT / "v1_5_anomaly_demo.png")


if __name__ == "__main__":
    print("rendering ->", OUT)
    f1(); f2(); f3(); f4(); f5()
    print("v1 done.")
