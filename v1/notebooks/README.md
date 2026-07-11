# Notebooks

All notebooks for v1, in reading order. They **import the `src` modules** and run them on the Jena
Climate data, so they're the documented, runnable view of the system.

## How to run them
- Use the **Python 3.12 (tshci)** kernel `py312-tshci` (the dependencies + GPU torch live there).
  In VS Code / Jupyter pick **"Python 3.12 (tshci)"**, then **Restart & Run All**.
- Each notebook has a self-bootstrap cell, so it finds the project's packages even if the kernel
  isn't perfectly set up. It also auto-locates the dataset, so it works from any folder.

## Reading order (6 notebooks)

| # | Notebook | What it teaches |
| --- | --- | --- |
| 00 | **code_walkthrough** | Tours every `src` module with short explanations + live examples. **Start here to understand the code.** |
| 01 | **data_exploration_and_visualization** | Part 1 documents the dataset (structure, missing/sentinels, stats, seasonality, correlations); Part 2 is the full visual gallery (trend, STL, monthly/diurnal cycles, heatmap, anomalies, distributions, clusters). |
| 02 | **methods_and_evaluation** | Part 1 demonstrates the three core analytics (query engine, anomaly detection, clustering); Part 2 verifies them against pandas ground truth and checks offline operation + limitations. |
| 03 | **interface_reasoning** | The natural-language interface (trust-aware answers), multi-sensor reasoning, and labeled evaluation. |
| 04 | **deep_models_comparison** | Part A trains the 1D CNN autoencoder; Part B compares the three deep techniques (CNN-AE, LSTM-AE, LSTM-Forecaster); Part C puts all six detectors side by side (counts, timing, precision/recall, confusion matrix). |
| 05 | **deep_training_techniques** | Self-contained (models inline, **no LLM**), GPU. Treats the station as a **14-sensor IoT array**: documents every sensor + faults + cross-sensor correlations, then trains **RNN·GRU·LSTM·CNN·Transformer** as **multivariate** autoencoders and compares them on **single-sensor fault** detection + **which-sensor attribution**; Sec C varies window/epochs, Sec D ranks per-sensor modelability. Headline: on the multivariate array the **Transformer wins** (cross-sensor consistency), reversing the univariate CNN result. |

> *Consolidated from the original 9 notebooks: old **01+04 → 01**, **02+03 → 02**, **06+07+08 → 04** (old 05 → 03).*

## Tips
- Notebooks **04 and 05** use deep models — they run on the **GPU** automatically. If you also have a
  large LLM loaded on the GPU at the same time, you may hit an out-of-memory error (the 8 GB card is
  shared). Restart the kernel to free GPU memory.
- Pick the **Python 3.12 (tshci)** kernel (`py312-tshci`) — it has the deps + GPU torch. Then
  **Restart & Run All**.
- The notebooks are **executed with outputs saved**, so you can read them without running anything.
