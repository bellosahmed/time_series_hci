"""
v2 deep anomaly detection — a multivariate Autoencoder (the recommended v2 deep model).

Idea: each month is a 54-dim vector (all countries' year-on-year growth). A small autoencoder learns
to reconstruct the *normal* cross-country pattern; months it reconstructs poorly (high error) are
systemic anomalies. This is the deep analogue of the Isolation Forest detector and the natural fit
for a short-but-wide series (it exploits the 54-channel width).

Kept deliberately small + seeded (292 points overfit easily — see DEEP_MODELS.md). Device-aware
(GPU if available). Importable; no plotting.
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


class AE(nn.Module):
    """Tiny under-complete autoencoder: n -> hidden -> latent -> hidden -> n."""

    def __init__(self, n: int, hidden: int = 16, latent: int = 6):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n, hidden), nn.ReLU(), nn.Linear(hidden, latent))
        self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, n))

    def forward(self, x):
        return self.dec(self.enc(x))


def _standardize(X: np.ndarray):
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-9
    return (X - mu) / sd, mu, sd


def train_autoencoder(features: pd.DataFrame, epochs: int = 400, lr: float = 1e-2,
                      weight_decay: float = 1e-4, seed: int = 0, device: str | None = None):
    """Fit the AE on the feature matrix. Returns (model, (mu, sd), device)."""
    device = get_device(device)
    torch.manual_seed(seed)
    X = features.dropna().values.astype("float32")
    Xs, mu, sd = _standardize(X)
    t = torch.tensor(Xs, device=device)
    model = AE(X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # decay = regularise
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(t), t)
        loss.backward()
        opt.step()
    return model, (mu, sd), device


def ae_scores(features: pd.DataFrame, trained=None, device: str | None = None) -> pd.Series:
    """Per-month reconstruction error (higher = more anomalous)."""
    if trained is None:
        trained = train_autoencoder(features, device=device)
    model, (mu, sd), device = trained
    valid = features.dropna()
    Xs = (valid.values.astype("float32") - mu) / sd
    model.eval()
    with torch.no_grad():
        rec = model(torch.tensor(Xs, device=device)).cpu().numpy()
    err = ((Xs - rec) ** 2).mean(axis=1)
    return pd.Series(err, index=valid.index).reindex(features.index)


def ae_anomalies(features: pd.DataFrame, contamination: float = 0.05, trained=None,
                 device: str | None = None) -> pd.Series:
    """Boolean mask: months whose reconstruction error is in the top `contamination` fraction."""
    s = ae_scores(features, trained=trained, device=device)
    thr = s.quantile(1 - contamination)
    return (s > thr).fillna(False)
