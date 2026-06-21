# Notebooks

All notebooks for v1, in reading order. They **import the `src` modules** and run them on the Jena
Climate data, so they're the documented, runnable view of the system.

## How to run them
- Use the **Python 3.12** kernel (the dependencies + GPU live there). In VS Code / Jupyter pick
  **"Python 3.12"** (or the `tshci` kernel), then **Restart & Run All**.
- Each notebook has a self-bootstrap cell, so it finds the project's packages even if the kernel
  isn't perfectly set up. It also auto-locates the dataset, so it works from any folder.

## Reading order

| # | Notebook | What it teaches |
| --- | --- | --- |
| 00 | **code_walkthrough** | Tours every `src` module with short explanations + live examples. **Start here to understand the code.** |
| 01 | **data_exploration** | Loads and documents the dataset: columns, missing data, seasonality, the temperature curve, sensor correlations. |
| 02 | **evaluation** | Proves the query engine matches pandas exactly; sanity-checks anomalies and clusters. |
| 03 | **methods_demo** | Explains and runs the three core analytics: query engine, anomaly detection, clustering. |
| 04 | **visualizations** | Nine plots: trend, STL decomposition, monthly/daily patterns, heatmap, anomalies, distributions, clusters, correlations. |
| 05 | **interface_demo** | The natural-language interface (trust-aware answers), multi-sensor reasoning, and labeled evaluation. |
| 06 | **deep_model** | Trains the 1D CNN autoencoder and compares it with the statistical and ML detectors. |
| 07 | **deep_comparison** | Compares the three deep techniques (CNN-AE, LSTM-AE, LSTM-Forecaster). |
| 08 | **model_comparison** | All six detectors side by side: counts, timing, precision/recall, confusion matrix. |

## Tips
- Notebooks 06–08 use deep models — they run on the **GPU** automatically. If you also have a large
  LLM loaded on the GPU at the same time, you may hit an out-of-memory error (the 8 GB card is
  shared). Restart the kernel to free GPU memory.
- The notebooks are **executed with outputs saved**, so you can read them without running anything.
