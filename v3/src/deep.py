"""v3 deep detectors — never tried in v3 before this experiment (see report/EVALUATION.md).

Two candidates, one per channel:

1. **`flow_lstm_ae_score`** — a multivariate LSTM autoencoder over short sequences of consecutive
   flow-feature windows (msg_rate, iat_mean/std, dup_msgid_ratio), replacing the flow channel's
   Isolation Forest with a sequence-aware reconstruction model. Tests whether a *temporal* duplication
   pattern is visible to a sequence model even where hand-aggregated per-window features wash it out.
2. **`value_lstm_forecast_score`** — v1's univariate `LSTMForecaster` (`v1/src/deep.py`), applied
   independently per sensor and combined via max-|z| across sensors (the same reduction
   `channels.value_score` already uses for the plain z-score value channel). Tests whether a learned
   per-sensor temporal model beats a marginal z-score for the *modification* attack — the case v1's
   notebook 05 predicts it should not (a single-channel value fault is a covariance/threshold problem,
   not a sequence problem), so this experiment doubles as a cross-domain check of that finding.

Both are small and seeded; DAD is only seven days, so any temporal model here has at most one example
of a weekly cycle to learn from — an honest limitation stated up front, not after the result.
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


# --------------------------------------------------------------------------- flow channel: LSTM-AE
class _LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden: int = 16, latent: int = 8):
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


def _windows(X: np.ndarray, window: int, step: int):
    starts = list(range(0, len(X) - window + 1, step))
    return np.stack([X[s:s + window] for s in starts]).astype("float32"), starts


def _average_overlapping(sq: np.ndarray, starts: list[int], window: int, n: int) -> np.ndarray:
    err_sum, err_cnt = np.zeros(n), np.zeros(n)
    for i, s in enumerate(starts):
        err_sum[s:s + window] += sq[i]
        err_cnt[s:s + window] += 1
    return err_sum / np.maximum(err_cnt, 1)


def flow_lstm_ae_score(flow_df: pd.DataFrame, window: int = 6, epochs: int = 100,
                       lr: float = 1e-3, batch_size: int = 32, seed: int = 0,
                       device: str | None = None) -> pd.Series:
    """Per-window reconstruction-error score from an LSTM autoencoder over the flow features.

    Trains on non-overlapping `window`-length sequences of consecutive flow-feature rows, scores on
    overlapping sequences, and averages each window's error across every sequence that covers it.
    """
    device = get_device(device)
    torch.manual_seed(seed)
    X = flow_df.to_numpy(dtype="float32")
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    Xs = (X - mu) / sd
    n, n_features = Xs.shape

    Xtr, _ = _windows(Xs, window, step=window)
    model = _LSTMAutoencoder(n_features).to(device)
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

    Xsc, starts = _windows(Xs, window, step=1)
    model.eval()
    with torch.no_grad():
        recon = model(torch.from_numpy(Xsc).to(device)).cpu().numpy()
    sq = ((Xsc - recon) ** 2).mean(axis=2)
    per_row = _average_overlapping(sq, starts, window, n)
    return pd.Series(per_row, index=flow_df.index, name="flow_lstm_ae_score")


# --------------------------------------------------------------------------- value channel: per-sensor forecaster
class _LSTMForecaster(nn.Module):
    def __init__(self, hidden: int = 16):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):                                  # x: [B, window, 1]
        _, (h, _) = self.lstm(x)
        return self.out(h[-1])


def _forecast_pairs(z: np.ndarray, window: int, step: int):
    starts = list(range(0, len(z) - window, step))
    X = np.stack([z[s:s + window] for s in starts])[:, :, None]
    y = np.array([z[s + window] for s in starts])[:, None]
    return X.astype("float32"), y.astype("float32"), [s + window for s in starts]


def _one_sensor_forecast_z(series: pd.Series, window: int, epochs: int, lr: float,
                           batch_size: int, seed: int, device: str) -> np.ndarray:
    """Standardised |forecast error| for one sensor's series -> a per-timestep z-like score."""
    s = series.astype(float)
    valid = s.dropna()
    if len(valid) < window + 5:                            # too short to forecast meaningfully
        return np.zeros(len(s))
    z = ((valid - valid.mean()) / (valid.std(ddof=0) + 1e-9)).to_numpy("float32")

    torch.manual_seed(seed)
    Xtr, ytr, _ = _forecast_pairs(z, window, step=max(1, window // 4))
    model = _LSTMForecaster().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    xt, yt = torch.from_numpy(Xtr).to(device), torch.from_numpy(ytr).to(device)
    model.train()
    for _ in range(epochs):
        for i in range(0, len(xt), batch_size):
            opt.zero_grad()
            loss = loss_fn(model(xt[i:i + batch_size]), yt[i:i + batch_size])
            loss.backward()
            opt.step()

    Xall, yall, targets = _forecast_pairs(z, window, step=1)
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(Xall).to(device)).cpu().numpy().ravel()
    err = np.abs(yall.ravel() - pred)
    err_z = (err - err.mean()) / (err.std(ddof=0) + 1e-9)  # comparable scale across sensors

    out = pd.Series(0.0, index=valid.index)
    out.iloc[targets] = err_z
    return out.reindex(series.index).fillna(0.0).to_numpy()


def value_lstm_forecast_score(sensor_df: pd.DataFrame, window: int = 24, epochs: int = 20,
                              lr: float = 1e-3, batch_size: int = 32, seed: int = 0,
                              device: str | None = None) -> pd.Series:
    """VALUE channel score: max |forecast-error z| across sensors, one small LSTM per sensor.

    Mirrors `channels.value_score`'s reduction (largest absolute per-sensor deviation in each window),
    but the per-sensor deviation comes from a learned next-step forecast instead of a marginal z-score.
    """
    device = get_device(device)
    cols = [_one_sensor_forecast_z(sensor_df[c], window, epochs, lr, batch_size, seed, device)
            for c in sensor_df.columns]
    Z = np.abs(np.stack(cols, axis=1))
    return pd.Series(Z.max(axis=1), index=sensor_df.index, name="value_lstm_forecast_score")
