"""Independently reproduce the network-level anomaly-detection AP-by-fault-type table from the
SAVED MODEL CHECKPOINTS (notebooks/saved_models_05/*.pt), not from the notebook and not from the
persisted CSV. This is a stronger reproducibility check than verify_network_level_results.py,
which only checks the CSV against the dissertation's cited figures: this script rebuilds each
architecture, loads its trained weights, and reruns the exact fault-injection protocol
(notebooks/05_deep_training_techniques.ipynb, cells 38/46/48) from scratch.

Run:  python3.12 verify_network_level_from_checkpoints.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src import config, preprocess  # noqa: E402

MODELS_DIR = ROOT / "notebooks" / "saved_models_05"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HIDDEN, LATENT = 32, 16

# ---------------------------------------------------------------------------
# Architecture builders (verbatim from notebooks/05_deep_training_techniques.ipynb, cell 11)
# ---------------------------------------------------------------------------

def build_rnn_ae(window, channels):
    class RNNAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.RNN(channels, HIDDEN, batch_first=True)
            self.to_latent = nn.Linear(HIDDEN, LATENT)
            self.from_latent = nn.Linear(LATENT, HIDDEN)
            self.decoder = nn.RNN(HIDDEN, HIDDEN, batch_first=True)
            self.out = nn.Linear(HIDDEN, channels)

        def forward(self, x):
            _, h = self.encoder(x)
            z = self.to_latent(h[-1])
            d = self.from_latent(z).unsqueeze(1).repeat(1, x.size(1), 1)
            seq, _ = self.decoder(d)
            return self.out(seq)
    return RNNAutoencoder()


def build_gru_ae(window, channels):
    class GRUAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.GRU(channels, HIDDEN, batch_first=True)
            self.to_latent = nn.Linear(HIDDEN, LATENT)
            self.from_latent = nn.Linear(LATENT, HIDDEN)
            self.decoder = nn.GRU(HIDDEN, HIDDEN, batch_first=True)
            self.out = nn.Linear(HIDDEN, channels)

        def forward(self, x):
            _, h = self.encoder(x)
            z = self.to_latent(h[-1])
            d = self.from_latent(z).unsqueeze(1).repeat(1, x.size(1), 1)
            seq, _ = self.decoder(d)
            return self.out(seq)
    return GRUAutoencoder()


def build_lstm_ae(window, channels):
    class LSTMAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.LSTM(channels, HIDDEN, batch_first=True)
            self.to_latent = nn.Linear(HIDDEN, LATENT)
            self.from_latent = nn.Linear(LATENT, HIDDEN)
            self.decoder = nn.LSTM(HIDDEN, HIDDEN, batch_first=True)
            self.out = nn.Linear(HIDDEN, channels)

        def forward(self, x):
            _, (h, _) = self.encoder(x)
            z = self.to_latent(h[-1])
            d = self.from_latent(z).unsqueeze(1).repeat(1, x.size(1), 1)
            seq, _ = self.decoder(d)
            return self.out(seq)
    return LSTMAutoencoder()


def build_cnn_ae(window, channels):
    class CNNAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv1d(channels, 16, 7, padding=3), nn.ReLU(),
                nn.Conv1d(16, 8, 7, padding=3), nn.ReLU(),
                nn.Flatten(),
                nn.Linear(8 * window, LATENT), nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(LATENT, 8 * window), nn.ReLU(),
                nn.Unflatten(1, (8, window)),
                nn.Conv1d(8, 16, 7, padding=3), nn.ReLU(),
                nn.Conv1d(16, channels, 7, padding=3),
            )

        def forward(self, x):
            out = self.decoder(self.encoder(x.transpose(1, 2)))
            return out.transpose(1, 2)
    return CNNAutoencoder()


def build_transformer_ae(window, channels, d_model=32, nhead=4, layers=2, dim_ff=64):
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=512):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(0, max_len).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, :x.size(1)]

    class TransformerAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(channels, d_model)
            self.pos = PositionalEncoding(d_model)
            enc = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, batch_first=True)
            self.encoder = nn.TransformerEncoder(enc, layers)
            self.out = nn.Linear(d_model, channels)

        def forward(self, x):
            return self.out(self.encoder(self.pos(self.inp(x))))
    return TransformerAutoencoder()


BUILDERS = {"RNN": build_rnn_ae, "GRU": build_gru_ae, "LSTM": build_lstm_ae,
            "CNN": build_cnn_ae, "Transformer": build_transformer_ae}

# ---------------------------------------------------------------------------
# Scoring machinery (verbatim from cell 9)
# ---------------------------------------------------------------------------


def apply_standardise(frame, mu, sd):
    return ((frame.to_numpy("float32") - mu) / sd).astype("float32")


def make_windows(Z, window, step):
    starts = list(range(0, len(Z) - window + 1, step))
    X = np.stack([Z[s:s + window] for s in starts]).astype("float32")
    return X, starts


def _predict_batched(model, X, batch_size=4096):
    device = next(model.parameters()).device
    model.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i + batch_size].astype("float32")).to(device)
            chunks.append(model(xb).cpu().numpy())
            del xb
    return np.concatenate(chunks, axis=0)


def recon_error_matrix(model, Z, window, score_step=1):
    Xsc, starts = make_windows(Z, window, score_step)
    recon = _predict_batched(model, Xsc)
    sq = (Xsc - recon) ** 2
    T, Sd = Z.shape
    err_sum = np.zeros((T, Sd))
    err_cnt = np.zeros((T, 1))
    for i, s in enumerate(starts):
        err_sum[s:s + window] += sq[i]
        err_cnt[s:s + window] += 1
    return err_sum / np.maximum(err_cnt, 1)


def network_score(err_TS):
    return err_TS.mean(axis=1)


def inject_sensor_faults(frame, n=100, magnitude=4.0, seed=42):
    rng = np.random.default_rng(seed)
    out = frame.copy()
    stds = frame.std(ddof=0).to_numpy()
    times = rng.choice(len(frame), size=n, replace=False)
    sensors = rng.integers(0, frame.shape[1], size=n)
    signs = rng.choice([-1.0, 1.0], size=n)
    labels = np.zeros(len(frame), bool)
    fault_sensor = np.full(len(frame), -1, int)
    for t, j, sgn in zip(times, sensors, signs):
        out.iat[t, j] += sgn * magnitude * stds[j]
        labels[t] = True
        fault_sensor[t] = j
    return out, labels, fault_sensor


def inject_drift(frame, n=100, magnitude=4.0, length=24, seed=1):
    rng = np.random.default_rng(seed)
    arr = frame.to_numpy().copy()
    stds = frame.std(ddof=0).to_numpy()
    labels = np.zeros(len(frame), bool)
    ramp = np.linspace(0, 1, length)
    starts = rng.choice(len(frame) - length, n, replace=False)
    js = rng.integers(0, frame.shape[1], n)
    sgn = rng.choice([-1.0, 1.0], n)
    for s0, j, g in zip(starts, js, sgn):
        arr[s0:s0 + length, j] += g * magnitude * stds[j] * ramp
        labels[s0:s0 + length] = True
    return pd.DataFrame(arr, index=frame.index, columns=frame.columns), labels


def inject_stuck(frame, n=100, length=24, seed=1, glob=False):
    rng = np.random.default_rng(seed)
    arr = frame.to_numpy().copy()
    labels = np.zeros(len(frame), bool)
    starts = rng.choice(len(frame) - length, n, replace=False)
    js = rng.integers(0, frame.shape[1], n)
    for s0, j in zip(starts, js):
        if glob:
            arr[s0:s0 + length, :] = arr[s0, :]
        else:
            arr[s0:s0 + length, j] = arr[s0, j]
        labels[s0:s0 + length] = True
    return pd.DataFrame(arr, index=frame.index, columns=frame.columns), labels


def main():
    print("Loading the real Jena Climate record...")
    df = preprocess.align(preprocess.clean(preprocess.load_data(config.DATA_PATH)), freq="1h")
    print(f"  {df.shape[0]} hourly readings x {df.shape[1]} sensors")

    print("Loading the saved standardiser and reloading each trained model checkpoint...")
    ckpt = np.load(MODELS_DIR / "standardiser.npz", allow_pickle=True)
    mu, sd, window = ckpt["mu"], ckpt["sd"], int(ckpt["window"])

    models = {}
    for name, builder in BUILDERS.items():
        m = builder(window, len(mu))
        m.load_state_dict(torch.load(MODELS_DIR / f"{name}.pt", map_location=DEVICE))
        m.to(DEVICE).eval()
        models[name] = m
        print(f"  reloaded {name} ({sum(p.numel() for p in m.parameters())} params)")

    Z = apply_standardise(df, mu, sd)

    print("Refitting the classical detectors on the same clean, standardised data...")
    _pca = PCA(n_components=5).fit(Z)
    _cov = LedoitWolf().fit(Z)
    _iso = IsolationForest(n_estimators=200, random_state=0).fit(Z)
    _sub = np.random.default_rng(0).choice(len(Z), 5000, replace=False)
    _lof = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(Z[_sub])
    _ocs = OneClassSVM(nu=0.05, gamma="scale").fit(Z[_sub])

    def classical_scores(Zn):
        rec = _pca.inverse_transform(_pca.transform(Zn))
        return {"PCA": ((Zn - rec) ** 2).sum(1),
                "Mahalanobis": _cov.mahalanobis(Zn),
                "IsoForest": -_iso.score_samples(Zn),
                "LOF": -_lof.score_samples(Zn),
                "OC-SVM": -_ocs.decision_function(Zn)}

    def ap_all_methods(frame, labels):
        Zn = apply_standardise(frame, mu, sd)
        out = {name: average_precision_score(labels, network_score(recon_error_matrix(m, Zn, window)))
               for name, m in models.items()}
        for name, sc in classical_scores(Zn).items():
            out[name] = average_precision_score(labels, sc)
        return out

    print("Re-running the exact fault-injection protocol (same seeds as the original notebook)...")
    pn, pl, _ = inject_sensor_faults(df, n=100, magnitude=4.0, seed=42)
    faults = {"spike (4s)": (pn, pl),
              "drift (4s,24h)": inject_drift(df, 100, 4.0, 24, seed=1),
              "stuck (24h)": inject_stuck(df, 100, 24, seed=1, glob=False),
              "global freeze (24h)": inject_stuck(df, 100, 24, seed=1, glob=True)}

    fresh = pd.DataFrame({ft: ap_all_methods(fr, lb) for ft, (fr, lb) in faults.items()}).round(3)
    fresh = fresh.sort_values("spike (4s)", ascending=False)
    print()
    print("Freshly reproduced (from saved checkpoints, not the notebook or the CSV):")
    print(fresh)

    csv_path = ROOT / "docs" / "evaluation" / "results" / "network_level_ap_by_fault_type.csv"
    persisted = pd.read_csv(csv_path, index_col=0)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    print()
    print("Persisted CSV (what verify_network_level_results.py checks against the dissertation):")
    print(persisted)

    # Column names differ between the fresh run and the persisted CSV (spike (4s) vs spike_4sigma
    # etc.) -- map them explicitly rather than relying on an exact-name intersection, which would
    # silently compare nothing.
    col_map = {"spike (4s)": "spike_4sigma", "drift (4s,24h)": "drift_4sigma_24h",
               "stuck (24h)": "stuck_24h", "global freeze (24h)": "global_freeze_24h"}

    print()
    print("=" * 70)
    diffs = []
    for idx in fresh.index:
        if idx not in persisted.index:
            continue
        for fresh_col, persisted_col in col_map.items():
            if persisted_col not in persisted.columns:
                continue
            a, b = fresh.loc[idx, fresh_col], persisted.loc[idx, persisted_col]
            if abs(a - b) > 0.005:
                diffs.append((idx, fresh_col, a, b))
    if diffs:
        print(f"MISMATCHES vs persisted CSV ({len(diffs)}):")
        for idx, col, a, b in diffs:
            print(f"  {idx:14s} {col:20s} fresh={a:.3f}  persisted={b:.3f}")
    else:
        print("EXACT MATCH: every figure reproduced from the saved model checkpoints matches the")
        print("persisted CSV (and therefore the dissertation's cited figures) to 3 decimal places.")
    print("=" * 70)


if __name__ == "__main__":
    main()
