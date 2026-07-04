"""Deep-learning anomaly detectors (all in one place).

Three unsupervised neural detectors, for comparison against the statistical (z-score),
classical-ML (Isolation Forest) and forecasting (ARIMA) detectors elsewhere in the pipeline:

1. **CNN autoencoder** (`cnn_ae_*`) — a 1D convolutional net reconstructs short windows of the
   signal. Having learned the normal daily shape, it rebuilds normal windows well and abnormal
   ones poorly, so a high **reconstruction error** marks an anomaly.
2. **LSTM autoencoder** (`lstm_ae_*`) — the same reconstruction idea, but a recurrent encoder/decoder
   captures *temporal order* explicitly.
3. **LSTM forecaster** (`lstm_forecast_*`) — predicts the next value from the past window; a large
   gap between predicted and actual = anomaly (a forecasting, rather than reconstruction, view).

All three need no labels, run locally, and use the GPU automatically when one is present. PyTorch is
imported lazily inside the functions, so the rest of the package works even without it installed.
Each detector exposes a `*_scores` function (a per-timestamp anomaly score) and a `*_anomalies`
function (a boolean flag Series thresholded at a quantile).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Shared utilities (device handling, windowing, scoring)
# --------------------------------------------------------------------------- #
def get_device():
    """Return 'cuda' if a GPU is available, else 'cpu'."""
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _seed_torch(seed: int) -> None:
    """Seed torch's global RNG for reproducibility.

    Weight initialisation happens at model *construction*, so the seed must be set
    **before** the model is built. The CNN path already does this inside
    ``train_autoencoder`` (which builds the model itself); the LSTM detectors build
    their model in the caller, so they must seed here first — otherwise the initial
    weights are unseeded and scores drift run-to-run despite a fixed ``seed``.
    """
    import torch
    torch.manual_seed(seed)


def _free_cuda(model=None) -> None:
    """Release a model's GPU memory back to the driver (no-op on CPU).

    Scoring the full multi-year series in one forward pass is what exhausts an
    8 GB card; we already batch inference, and this returns the freed blocks so
    training several deep models in one session (notebook 04) doesn't accumulate.
    """
    import torch
    if model is not None:
        model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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


def standardise(series: pd.Series) -> np.ndarray:
    """Z-standardise a series to a float32 array (mean 0, std 1).

    Shared by every deep detector so they all normalise the signal identically before windowing.
    """
    s = series.astype(float)
    return ((s - s.mean()) / s.std(ddof=0)).to_numpy("float32")


def average_overlapping_windows(sq: np.ndarray, starts, window: int, n: int,
                                index, name: str) -> pd.Series:
    """Map per-window errors back to a per-timestamp score.

    Overlapping windows each cover a timestamp; averaging their errors at every point
    localises an anomaly far better than one error per window. Shared by the CNN- and
    LSTM-autoencoder scorers. `sq` is [N_windows, window]; returns a Series over `index`.
    """
    err_sum = np.zeros(n)
    err_cnt = np.zeros(n)
    for i, start in enumerate(starts):
        err_sum[start:start + window] += sq[i]
        err_cnt[start:start + window] += 1
    return pd.Series(err_sum / np.maximum(err_cnt, 1), index=index, name=name)


# --------------------------------------------------------------------------- #
# 1. CNN Autoencoder
# --------------------------------------------------------------------------- #
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
                      batch_size: int = 128, seed: int = 42, verbose: bool = False,
                      device: str | None = None):
    """Train the autoencoder on windows X and return the fitted model.

    Uses the GPU automatically when available (`device=None` -> auto).
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    device = device or get_device()
    torch.manual_seed(seed)
    window = X.shape[-1]
    model = _build_model(window).to(device)
    tensor = torch.from_numpy(X.astype("float32"))
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
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


def _squared_errors(model, X: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    """Element-wise squared error per window: returns [N, window].

    Inference runs in mini-batches so the whole (overlapping-window) scoring set
    is never resident on the GPU at once — this is what prevents CUDA out-of-memory
    on the long Jena series.
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
    recon = np.concatenate(chunks, axis=0)
    return ((X - recon) ** 2)[:, 0, :]


def cnn_ae_scores(series: pd.Series, window: int = 24, epochs: int = 20,
                  score_step: int = 1, seed: int = 42, device: str | None = None) -> pd.Series:
    """Per-timestamp reconstruction-error score (higher = more anomalous).

    Trains on non-overlapping windows (fast), then scores with overlapping windows
    and averages each timestamp's error across all windows covering it — which
    localises anomalies far better than one error per window.
    """
    z = standardise(series)

    X_train, _ = _windows_from_array(z, window, step=window)
    model = train_autoencoder(X_train, epochs=epochs, seed=seed, device=device)

    X_score, starts = _windows_from_array(z, window, step=score_step)
    sq = _squared_errors(model, X_score)                  # [N, window]
    _free_cuda(model)                                     # release GPU before returning
    return average_overlapping_windows(sq, starts, window, len(z), series.index, "cnn_ae_score")


def cnn_ae_anomalies(series: pd.Series, window: int = 24, epochs: int = 20,
                     quantile: float = 0.99, score_step: int = 1,
                     seed: int = 42, device: str | None = None) -> pd.Series:
    """Flag timestamps whose per-point reconstruction error exceeds `quantile`."""
    scores = cnn_ae_scores(series, window=window, epochs=epochs,
                           score_step=score_step, seed=seed, device=device)
    threshold = np.quantile(scores, quantile)
    return (scores > threshold).rename("cnn_ae_anomaly")


# --------------------------------------------------------------------------- #
# Batched inference shared by the LSTM models
# --------------------------------------------------------------------------- #
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


def _train(model, X_seq, epochs, lr, batch_size, seed, predict_y=None, device=None):
    """Generic training loop. X_seq: [N, window, 1]. If predict_y given, supervised.

    Uses the GPU automatically when available.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

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


# --------------------------------------------------------------------------- #
# 2. LSTM Autoencoder
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


def lstm_ae_scores(series: pd.Series, window: int = 24, epochs: int = 10,
                   score_step: int = 4, seed: int = 42) -> pd.Series:
    """Per-timestamp reconstruction error from an LSTM autoencoder."""
    z = standardise(series)
    Xtr, _ = _windows_from_array(z, window, step=window)
    _seed_torch(seed)                                  # seed before building (weight init)
    model = _build_lstm_ae()
    _train(model, _to_seq(Xtr), epochs, 1e-3, 128, seed)

    Xsc, starts = _windows_from_array(z, window, step=score_step)
    Xseq = _to_seq(Xsc)
    recon = _predict_batched(model, Xseq)[:, :, 0]
    _free_cuda(model)
    sq = (Xseq[:, :, 0] - recon) ** 2                  # [N, window]
    return average_overlapping_windows(sq, starts, window, len(z), series.index, "lstm_ae_score")


def lstm_ae_anomalies(series: pd.Series, window: int = 24, epochs: int = 10,
                      quantile: float = 0.99, score_step: int = 4, seed: int = 42) -> pd.Series:
    scores = lstm_ae_scores(series, window, epochs, score_step, seed)
    return (scores > np.quantile(scores, quantile)).rename("lstm_ae_anomaly")


# --------------------------------------------------------------------------- #
# 3. LSTM Forecaster
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
    z = standardise(series)
    Xtr, ytr, _ = _forecast_pairs(z, window, step=train_step)
    _seed_torch(seed)                                  # seed before building (weight init)
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
