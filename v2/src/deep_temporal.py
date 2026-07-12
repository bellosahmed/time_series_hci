"""v2 temporal deep detectors — LSTM-AE, CNN-AE, TCN-AE (multivariate, sequence-aware).

`deep_anomaly.AE` treats each month as an independent 54-dim vector: it can flag a month whose
cross-country pattern is unusual, but it has no notion of *order* — a slow six-month slide into a
crisis looks identical to that same vector appearing in isolation. These three architectures instead
window several consecutive months into a short sequence and reconstruct the *whole window*, so a
temporal shape (not just a single month's snapshot) can drive the anomaly score:

1. **LSTM-AE**  — recurrent encoder/decoder, the direct multivariate port of v1's `deep.py` LSTM-AE.
2. **CNN-AE**   — 1D convolution over the time axis, multivariate port of v1's CNN-AE.
3. **TCN-AE**   — dilated causal convolutions (a Temporal Convolutional Network), not used anywhere
   else in this project; included because it is the architecture family explicitly named in v1's
   literature review but never implemented.

All three share one training/scoring harness (`temporal_ae_scores` / `temporal_ae_anomalies`) so they
are directly comparable to each other and to `deep_anomaly.AE` / `anomaly.isoforest_anomalies` on the
exact same `systemic_features` input. Kept small and seeded — 292 months is a hard regime for any
sequence model; see `report/EVALUATION.md` for the honest result of trying anyway.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def get_device(device: str | None = None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- windowing
def _windows_from_array(X: np.ndarray, window: int, step: int):
    """[T, F] -> (Xw[N, window, F], start_positions)."""
    starts = list(range(0, len(X) - window + 1, step))
    Xw = np.stack([X[s:s + window] for s in starts]).astype("float32")
    return Xw, starts


def _average_overlapping(sq: np.ndarray, starts: list[int], window: int, n: int) -> np.ndarray:
    """Map per-window-per-timestep squared error [N, window] back to a per-timestep score [n]."""
    err_sum = np.zeros(n)
    err_cnt = np.zeros(n)
    for i, s in enumerate(starts):
        err_sum[s:s + window] += sq[i]
        err_cnt[s:s + window] += 1
    return err_sum / np.maximum(err_cnt, 1)


# --------------------------------------------------------------------------- architectures
class LSTMAutoencoder(nn.Module):
    """Recurrent encoder/decoder over an [B, window, n_features] sequence."""

    def __init__(self, n_features: int, hidden: int = 32, latent: int = 16):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden, batch_first=True)
        self.to_latent = nn.Linear(hidden, latent)
        self.from_latent = nn.Linear(latent, hidden)
        self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_features)

    def forward(self, x):                                  # x: [B, window, n_features]
        _, (h, _) = self.encoder(x)
        z = self.to_latent(h[-1])
        d = self.from_latent(z).unsqueeze(1).repeat(1, x.size(1), 1)
        seq, _ = self.decoder(d)
        return self.out(seq)


class CNNAutoencoder(nn.Module):
    """1D convolution over the time axis, `n_features` input/output channels."""

    def __init__(self, n_features: int, window: int, latent: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_features, 32, 3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 16, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * window, latent), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, 16 * window), nn.ReLU(),
            nn.Unflatten(1, (16, window)),
            nn.Conv1d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Conv1d(32, n_features, 3, padding=1),
        )

    def forward(self, x):                                  # x: [B, window, n_features]
        xc = x.transpose(1, 2)                              # -> [B, n_features, window]
        rec = self.decoder(self.encoder(xc))
        return rec.transpose(1, 2)                          # -> [B, window, n_features]


class _TCNBlock(nn.Module):
    """One dilated-causal-conv residual block: pad on the left only, so output[t] never sees input[>t]."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation)
        self.relu = nn.ReLU()
        self.proj = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        out = self.relu(self.conv(nn.functional.pad(x, (self.pad, 0))))
        return out + self.proj(x)


class TCNAutoencoder(nn.Module):
    """Temporal Convolutional Network autoencoder: causal dilated convs (dilation 1, 2, 4).

    Named in v1's literature review (Bai, Kolter & Koltun, 2018) but never implemented anywhere in
    the project until now. Causal + dilated convolutions grow the receptive field exponentially with
    depth while keeping the model small — the natural "give it more temporal context per parameter"
    alternative to a recurrent (LSTM) or windowed (CNN) autoencoder.
    """

    def __init__(self, n_features: int, window: int, hidden: int = 16, latent: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            _TCNBlock(n_features, hidden, kernel=3, dilation=1),
            _TCNBlock(hidden, hidden, kernel=3, dilation=2),
            _TCNBlock(hidden, hidden, kernel=3, dilation=4),
        )
        self.to_latent = nn.Conv1d(hidden, latent, 1)
        self.decoder = nn.Sequential(
            _TCNBlock(latent, hidden, kernel=3, dilation=1),
            _TCNBlock(hidden, hidden, kernel=3, dilation=2),
        )
        self.out = nn.Conv1d(hidden, n_features, 1)

    def forward(self, x):                                  # x: [B, window, n_features]
        xc = x.transpose(1, 2)                              # -> [B, n_features, window]
        z = self.to_latent(self.encoder(xc))
        rec = self.out(self.decoder(z))
        return rec.transpose(1, 2)


_ARCHS = {"lstm_ae": LSTMAutoencoder, "cnn_ae": CNNAutoencoder, "tcn_ae": TCNAutoencoder}


def _build(arch: str, n_features: int, window: int):
    if arch not in _ARCHS:
        raise ValueError(f"Unknown arch {arch!r}. Choose one of: {', '.join(_ARCHS)}.")
    cls = _ARCHS[arch]
    return cls(n_features, window) if arch != "lstm_ae" else cls(n_features)


# --------------------------------------------------------------------------- train / score
def _standardize(X: np.ndarray):
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-9
    return (X - mu) / sd, mu, sd


def temporal_ae_scores(features: pd.DataFrame, arch: str = "lstm_ae", window: int = 6,
                       epochs: int = 100, lr: float = 1e-3, batch_size: int = 16,
                       seed: int = 0, device: str | None = None) -> pd.Series:
    """Per-month reconstruction-error score from a temporal (sequence) autoencoder.

    Trains on non-overlapping `window`-month sequences (fast, avoids near-duplicate windows eating
    the already-small training set), scores on overlapping windows, and averages each month's error
    across every window that covers it — the same localisation trick as v1's `deep.py`.
    """
    device = get_device(device)
    torch.manual_seed(seed)
    valid = features.dropna()
    X = valid.to_numpy(dtype="float32")
    Xs, _, _ = _standardize(X)
    n, n_features = Xs.shape
    if n < window:
        raise ValueError(f"Only {n} valid rows but window={window}.")

    Xtr, _ = _windows_from_array(Xs, window, step=window)
    model = _build(arch, n_features, window).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    tensor = torch.from_numpy(Xtr).to(device)
    model.train()
    for _ in range(epochs):
        for i in range(0, len(tensor), batch_size):
            batch = tensor[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            opt.step()

    Xsc, starts = _windows_from_array(Xs, window, step=1)
    model.eval()
    with torch.no_grad():
        recon = model(torch.from_numpy(Xsc).to(device)).cpu().numpy()
    sq = ((Xsc - recon) ** 2).mean(axis=2)                  # [N, window] — mean over features
    per_row = _average_overlapping(sq, starts, window, n)
    return pd.Series(per_row, index=valid.index, name=f"{arch}_score").reindex(features.index)


def temporal_ae_anomalies(features: pd.DataFrame, arch: str = "lstm_ae", window: int = 6,
                          epochs: int = 100, quantile: float = 0.95, seed: int = 0,
                          device: str | None = None) -> pd.Series:
    """Boolean mask: months whose temporal reconstruction error is in the top `1-quantile` fraction."""
    s = temporal_ae_scores(features, arch=arch, window=window, epochs=epochs, seed=seed, device=device)
    thr = s.quantile(quantile)
    return (s > thr).fillna(False)
