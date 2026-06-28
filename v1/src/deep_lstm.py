"""LSTM-based deep anomaly detection (two techniques).

Complements the 1D CNN autoencoder in `deep_anomaly.py`:

1. **LSTM Autoencoder** — a recurrent encoder compresses a window to a latent
   vector, a recurrent decoder rebuilds it. Like the CNN-AE, high reconstruction
   error = anomaly, but the recurrence captures *temporal order* explicitly.
2. **LSTM Forecaster** — given the past window it predicts the next value; a large
   gap between predicted and actual = anomaly. A forecasting (not reconstruction)
   view of the same idea.

Both are unsupervised, run locally on CPU, and return a boolean Series aligned to
the input. PyTorch is imported lazily. Windowing is shared with `deep_anomaly`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .deep_anomaly import _free_cuda, _windows_from_array


def _standardise(series: pd.Series):
    s = series.astype(float)
    return ((s - s.mean()) / s.std(ddof=0)).to_numpy("float32")


def _predict_batched(model, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """Run model.forward over X in mini-batches and return stacked CPU output.

    LSTMs build large intermediate state, so scoring the whole series in one
    forward pass is the main CUDA-OOM risk; batching bounds peak GPU memory.
    """
    import torch

    device = next(model.parameters()).device
    model.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i + batch_size].astype("float32")).to(device)
            chunks.append(model(xb).cpu().numpy())
            del xb
    return np.concatenate(chunks, axis=0)


# --------------------------------------------------------------------------- #
# LSTM Autoencoder
# --------------------------------------------------------------------------- #
def _build_lstm_ae(hidden: int = 32, latent: int = 16):
    import torch.nn as nn

    class LSTMAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.LSTM(1, hidden, batch_first=True)
            self.to_latent = nn.Linear(hidden, latent)
            self.from_latent = nn.Linear(latent, hidden)
            self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
            self.out = nn.Linear(hidden, 1)

        def forward(self, x):                      # x: [B, window, 1]
            _, (h, _) = self.encoder(x)
            z = self.to_latent(h[-1])              # [B, latent]
            d = self.from_latent(z).unsqueeze(1).repeat(1, x.size(1), 1)
            seq, _ = self.decoder(d)
            return self.out(seq)                   # [B, window, 1]

    return LSTMAutoencoder()


def _train(model, X_seq, epochs, lr, batch_size, seed, predict_y=None, device=None):
    """Generic training loop. X_seq: [N, window, 1]. If predict_y given, supervised.

    Uses the GPU automatically when available.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from .deep_anomaly import get_device
    device = device or get_device()
    model = model.to(device)
    torch.manual_seed(seed)
    xt = torch.from_numpy(X_seq.astype("float32"))
    ds = TensorDataset(xt, torch.from_numpy(predict_y.astype("float32"))) if predict_y is not None \
        else TensorDataset(xt)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for batch in loader:
            opt.zero_grad()
            if predict_y is not None:
                xb, yb = batch[0].to(device), batch[1].to(device)
                loss = loss_fn(model(xb), yb)
            else:
                xb = batch[0].to(device)
                loss = loss_fn(model(xb), xb)
            loss.backward()
            opt.step()
    return model


def _to_seq(X):
    """[N,1,window] -> [N,window,1] for LSTM input."""
    return np.transpose(X, (0, 2, 1))


def lstm_ae_scores(series: pd.Series, window: int = 24, epochs: int = 10,
                   score_step: int = 4, seed: int = 42) -> pd.Series:
    """Per-timestamp reconstruction error from an LSTM autoencoder."""
    z = _standardise(series)
    Xtr, _ = _windows_from_array(z, window, step=window)
    model = _build_lstm_ae()
    _train(model, _to_seq(Xtr), epochs, 1e-3, 128, seed)

    Xsc, starts = _windows_from_array(z, window, step=score_step)
    Xseq = _to_seq(Xsc)
    recon = _predict_batched(model, Xseq)[:, :, 0]
    _free_cuda(model)
    sq = (Xseq[:, :, 0] - recon) ** 2                  # [N, window]

    err_sum = np.zeros(len(z)); err_cnt = np.zeros(len(z))
    for i, s in enumerate(starts):
        err_sum[s:s + window] += sq[i]
        err_cnt[s:s + window] += 1
    return pd.Series(err_sum / np.maximum(err_cnt, 1), index=series.index, name="lstm_ae_score")


def lstm_ae_anomalies(series: pd.Series, window: int = 24, epochs: int = 10,
                      quantile: float = 0.99, score_step: int = 4, seed: int = 42) -> pd.Series:
    scores = lstm_ae_scores(series, window, epochs, score_step, seed)
    return (scores > np.quantile(scores, quantile)).rename("lstm_ae_anomaly")


# --------------------------------------------------------------------------- #
# LSTM Forecaster
# --------------------------------------------------------------------------- #
def _build_lstm_forecaster(hidden: int = 32):
    import torch.nn as nn

    class LSTMForecaster(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, hidden, batch_first=True)
            self.out = nn.Linear(hidden, 1)

        def forward(self, x):                      # x: [B, window, 1]
            _, (h, _) = self.lstm(x)
            return self.out(h[-1])                 # [B, 1] next-step prediction

    return LSTMForecaster()


def _forecast_pairs(z: np.ndarray, window: int, step: int):
    """Build (X[N,window,1], y[N,1], target_positions) for next-step forecasting."""
    starts = list(range(0, len(z) - window, step))
    X = np.stack([z[s:s + window] for s in starts])[:, :, None]
    y = np.array([z[s + window] for s in starts])[:, None]
    targets = [s + window for s in starts]
    return X.astype("float32"), y.astype("float32"), targets


def lstm_forecast_scores(series: pd.Series, window: int = 24, epochs: int = 10,
                         train_step: int = 4, seed: int = 42) -> pd.Series:
    """Per-timestamp absolute forecast error (|actual - predicted|)."""
    z = _standardise(series)
    Xtr, ytr, _ = _forecast_pairs(z, window, step=train_step)
    model = _build_lstm_forecaster()
    _train(model, Xtr, epochs, 1e-3, 128, seed, predict_y=ytr)

    Xall, yall, targets = _forecast_pairs(z, window, step=1)
    pred = _predict_batched(model, Xall).ravel()
    _free_cuda(model)
    err = np.abs(yall.ravel() - pred)

    scores = pd.Series(0.0, index=series.index, name="lstm_forecast_score")
    scores.iloc[targets] = err
    return scores


def lstm_forecast_anomalies(series: pd.Series, window: int = 24, epochs: int = 10,
                            quantile: float = 0.99, train_step: int = 4, seed: int = 42) -> pd.Series:
    scores = lstm_forecast_scores(series, window, epochs, train_step, seed)
    return (scores > np.quantile(scores, quantile)).rename("lstm_forecast_anomaly")
