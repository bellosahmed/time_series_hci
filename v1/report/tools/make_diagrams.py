"""Generate the real, data-backed figures requested for the dissertation:

1. fig_anomaly_bounds.png   - temperature series with detected anomalies highlighted
                               (Mahalanobis network-level detector output on the real record).
2. fig_forecast_interval.png - the 14-day-ahead forecast band, and the 20-fold backtest MASE
                                distribution against a naive-persistence baseline (MASE=1).
3. fig_faithfulness_bars.png - bar chart of the faithfulness-gate invention rates *as already
                                reported* in Chapter 4 SS4.7 (2026-08-22 run): not re-run here,
                                since the model is stochastic and a fresh run would not match the
                                figures already written into the dissertation text.

Run from code/v1:  python3.12 report/tools/make_diagrams.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from src import anomaly, forecasting, preprocess

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True)
DATA = Path(__file__).resolve().parents[2] / "data" / "jena_climate_2009_2016.csv"
SENSOR = "T (degC)"

print("Loading and cleaning the real Jena Climate record...")
raw = preprocess.load_data(str(DATA))
df = preprocess.clean(raw)
df = preprocess.align(df, freq="1h")
print(f"  {len(df):,} hourly rows, {df.index.min()} to {df.index.max()}")

# ---------------------------------------------------------------------------
# Figure 1: anomaly detection bounds (network-level Mahalanobis+flatline portfolio)
# ---------------------------------------------------------------------------
print("Running the network-level anomaly portfolio (Mahalanobis + flat-line)...")
window = df.loc["2015-06-15":"2015-08-15"]
flags = anomaly.sensor_network_anomalies(window, contamination=0.01)

fig, ax = plt.subplots(figsize=(10, 4.2))
ax.plot(window.index, window[SENSOR], color="#2f5fc7", linewidth=0.9, label="Temperature (deg C)")
flagged = window.index[flags.astype(bool)]
ax.scatter(flagged, window.loc[flagged, SENSOR], color="#d64545", s=22, zorder=5,
           label=f"Flagged anomalous ({int(flags.sum())} of {len(window)} hours)")
ax.set_title("Network-level anomaly detection (Mahalanobis + flat-line portfolio)\n"
              "15 Jun - 15 Aug 2015, real Jena Climate record")
ax.set_ylabel("Temperature (deg C)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "fig_anomaly_bounds.png", dpi=160)
plt.close(fig)
print("  saved fig_anomaly_bounds.png")

# ---------------------------------------------------------------------------
# Figure 2: forecast confidence band + backtest MASE distribution
# ---------------------------------------------------------------------------
print("Fitting the ARIMA forecast and running the 20-fold backtest (this takes a while)...")
daily = df[SENSOR].resample("1D").mean()
fc = forecasting.forecast(daily, steps=14)
bt = forecasting.backtest(daily, horizon=14, folds=20)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

# left: forecast band on the trailing 90 days of history
hist = daily.iloc[-90:]
ax = axes[0]
ax.plot(hist.index, hist.values, color="#2f5fc7", linewidth=1.1, label="Observed daily mean")
ax.plot(fc.index, fc["mean"], color="#dd8b1f", linewidth=1.4, label="14-day forecast")
ax.fill_between(fc.index, fc["lower"], fc["upper"], color="#dd8b1f", alpha=0.25, label="Uncertainty band")
ax.set_title("14-day-ahead forecast with uncertainty band")
ax.set_ylabel("Temperature (deg C)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.legend(loc="upper left", fontsize=8)
ax.grid(alpha=0.25)

# right: per-fold MASE against the persistence baseline (MASE=1)
ax = axes[1]
fold_scores = np.asarray(bt["per_fold_mase"])
folds_idx = np.arange(1, len(fold_scores) + 1)
colors = ["#2f9e5f" if s < 1 else "#d64545" for s in fold_scores]
ax.bar(folds_idx, fold_scores, color=colors)
ax.axhline(1.0, color="#0f172a", linewidth=1.2, linestyle="--", label="Naive persistence (MASE=1)")
ax.set_title(f"20-fold backtest MASE (mean {bt['mase_mean']:.3f}, "
             f"95% CI {bt['mase_ci_low']:.3f}-{bt['mase_ci_high']:.3f})")
ax.set_xlabel("Fold")
ax.set_ylabel("MASE (lower is better)")
ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.25, axis="y")

fig.tight_layout()
fig.savefig(OUT / "fig_forecast_interval.png", dpi=160)
plt.close(fig)
print("  saved fig_forecast_interval.png")
print(f"  pooled MASE={bt['mase']:.3f}, mean={bt['mase_mean']:.3f}, "
      f"95% CI=({bt['mase_ci_low']:.3f}, {bt['mase_ci_high']:.3f})")

# ---------------------------------------------------------------------------
# Figure 3: faithfulness-gate invention rate, as already reported in Ch4 SS4.7
# ---------------------------------------------------------------------------
labels = ["Raw / ungated\n(ablation)", "Gated\n(ablation)", "Gated, adversarial\nprobe"]
rates = [0.0, 20.0, 20.0]
colors = ["#94a3b8", "#2f9e5f", "#dd8b1f"]

fig, ax = plt.subplots(figsize=(6.5, 4.2))
bars = ax.bar(labels, rates, color=colors, width=0.55)
for b, r in zip(bars, rates, strict=True):
    ax.text(b.get_x() + b.get_width() / 2, r + 1, f"{r:.0f}%", ha="center", fontsize=10, fontweight="bold")
ax.set_ylabel("Invention rate (% of runs)")
ax.set_ylim(0, max(rates) + 15)
ax.set_title("Faithfulness-gate invention rate\n(15 repeats per arm, reported 2026-08-22, SS4.7)")
ax.grid(alpha=0.25, axis="y")
fig.tight_layout()
fig.savefig(OUT / "fig_faithfulness_bars.png", dpi=160)
plt.close(fig)
print("  saved fig_faithfulness_bars.png (plots the SS4.7 reported figures, not a fresh run)")

print("Done. Figures in", OUT)
