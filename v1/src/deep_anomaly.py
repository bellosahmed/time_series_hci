"""Deep anomaly detection: a 1D Convolutional Autoencoder (unsupervised).

Idea: train a small 1D CNN to *reconstruct* short windows of the (normalised)
signal. Having learned the normal daily/seasonal shape, it reconstructs normal
windows well and abnormal ones poorly — so a high **reconstruction error** marks
an anomaly. No labels are needed, and it runs locally on CPU in seconds.

This complements the statistical (z-score) and classical-ML (Isolation Forest)
detectors for a three-way comparison. PyTorch is imported lazily so the rest of
the package works without it installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _windows_from_array(values: np.ndarray, window: int, step: int):
    """Return (X[N,1,window], start_positions) from a 1-D array."""
    starts = list(range(0, len(values) - window + 1, step))
    X = np.stack([values[s:s + window] for s in starts])[:, None, :].astype("float32")
    return X, starts


def make_windows(series: pd.Series, window: int = 24, step: int | None = None):
    """Slice a series into windows. Returns (X[N,1,window], list-of-index-arrays).

    Non-overlapping by default (step=window); each window keeps the timestamps it
    covers so errors can be mapped back to the original index.
    """
    step = step or window
    X, starts = _windows_from_array(series.to_numpy(dtype="float32"), window, step)
    spans = [series.index[s:s + window] for s in starts]
    return X, spans


def _build_model(window: int):
    import torch.nn as nn

    class CNNAutoencoder(nn.Module):
        def __init__(self, w: int, latent: int = 16):
            super().__init__()
            # conv with padding=(k-1)//2 keeps length = w throughout
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 16, 7, padding=3), nn.ReLU(),
                nn.Conv1d(16, 8, 7, padding=3), nn.ReLU(),
                nn.Flatten(),
                nn.Linear(8 * w, latent), nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent, 8 * w), nn.ReLU(),
                nn.Unflatten(1, (8, w)),
                nn.Conv1d(8, 16, 7, padding=3), nn.ReLU(),
                nn.Conv1d(16, 1, 7, padding=3),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    return CNNAutoencoder(window)


def train_autoencoder(X: np.ndarray, epochs: int = 20, lr: float = 1e-3,
                      batch_size: int = 128, seed: int = 42, verbose: bool = False):
    """Train the autoencoder on windows X and return the fitted model."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    window = X.shape[-1]
    model = _build_model(window)
    tensor = torch.from_numpy(X.astype("float32"))
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total = 0.0
        for (batch,) in loader:
            opt.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)
        if verbose:
            print(f"epoch {epoch + 1:>2}/{epochs}  loss={total / len(tensor):.4f}")
    return model


def reconstruction_errors(model, X: np.ndarray) -> np.ndarray:
    """Per-window mean-squared reconstruction error (one number per window)."""
    return _squared_errors(model, X).mean(axis=1)


def _squared_errors(model, X: np.ndarray) -> np.ndarray:
    """Element-wise squared error per window: returns [N, window]."""
    import torch

    model.eval()
    with torch.no_grad():
        recon = model(torch.from_numpy(X.astype("float32"))).numpy()
    return ((X - recon) ** 2)[:, 0, :]


def cnn_ae_scores(series: pd.Series, window: int = 24, epochs: int = 20,
                  score_step: int = 1, seed: int = 42) -> pd.Series:
    """Per-timestamp reconstruction-error score (higher = more anomalous).

    Trains on non-overlapping windows (fast), then scores with overlapping windows
    and averages each timestamp's error across all windows covering it — which
    localises anomalies far better than one error per window.
    """
    s = series.astype(float)
    z = ((s - s.mean()) / s.std(ddof=0)).to_numpy("float32")

    X_train, _ = _windows_from_array(z, window, step=window)
    model = train_autoencoder(X_train, epochs=epochs, seed=seed)

    X_score, starts = _windows_from_array(z, window, step=score_step)
    sq = _squared_errors(model, X_score)                  # [N, window]
    err_sum = np.zeros(len(z))
    err_cnt = np.zeros(len(z))
    for i, start in enumerate(starts):
        err_sum[start:start + window] += sq[i]
        err_cnt[start:start + window] += 1
    scores = err_sum / np.maximum(err_cnt, 1)
    return pd.Series(scores, index=series.index, name="cnn_ae_score")


def cnn_ae_anomalies(series: pd.Series, window: int = 24, epochs: int = 20,
                     quantile: float = 0.99, score_step: int = 1,
                     seed: int = 42) -> pd.Series:
    """Flag timestamps whose per-point reconstruction error exceeds `quantile`."""
    scores = cnn_ae_scores(series, window=window, epochs=epochs,
                           score_step=score_step, seed=seed)
    threshold = np.quantile(scores, quantile)
    return (scores > threshold).rename("cnn_ae_anomaly")
