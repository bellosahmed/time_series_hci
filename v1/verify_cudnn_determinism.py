"""Reverify the root cause found in verify_network_level_from_checkpoints.py: does training the
CNN autoencoder twice from the same seed give different results (as observed between the
persisted CSV and the saved checkpoint), and does setting the standard cuDNN determinism flags
fix it? Trains CNN four times total (2x default settings, 2x deterministic settings) and compares
the resulting spike-fault average precision each time. Does not touch any dissertation-cited
figure -- this is a small, contained reverification of the mechanism only.

Run:  python3.12 verify_cudnn_determinism.py
"""
from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src import config, preprocess  # noqa: E402
from verify_network_level_from_checkpoints import (  # noqa: E402
    apply_standardise,
    build_cnn_ae,
    inject_sensor_faults,
    network_score,
    recon_error_matrix,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW, MAX_EPOCHS, PATIENCE, SEED = 24, 50, 5, 42


def make_windows(Z, window, step):
    starts = list(range(0, len(Z) - window + 1, step))
    X = np.stack([Z[s:s + window] for s in starts]).astype("float32")
    return X, starts


def _mean_loss(model, X, batch_size=4096):
    device = next(model.parameters()).device
    model.eval()
    tot, n = 0.0, 0
    loss_sum = nn.MSELoss(reduction="sum")
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i + batch_size].astype("float32")).to(device)
            tot += loss_sum(model(xb), xb).item()
            n += xb.numel()
            del xb
    return tot / n


def train_es(model, X_train, X_val, max_epochs=50, patience=5, min_delta=1e-5,
             lr=1e-3, batch_size=128, seed=42, device=None):
    device = device or DEVICE
    model = model.to(device)
    torch.manual_seed(seed)
    xt = torch.from_numpy(X_train.astype("float32"))
    loader = DataLoader(TensorDataset(xt), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best, best_state, best_ep, wait = float("inf"), None, 0, 0
    for ep in range(1, max_epochs + 1):
        model.train()
        for (batch,) in loader:
            batch = batch.to(device)
            opt.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            opt.step()
        va = _mean_loss(model, X_val)
        if va < best - min_delta:
            best, best_state, best_ep, wait = va, copy.deepcopy(model.state_dict()), ep, 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_ep


def run_one(df, mu, sd, Xtr, Xval, deterministic: bool) -> tuple[float, int, float]:
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.manual_seed(SEED)
    model = build_cnn_ae(WINDOW, len(mu))
    t0 = time.time()
    model, best_ep = train_es(model, Xtr, Xval, max_epochs=MAX_EPOCHS, patience=PATIENCE, seed=SEED)
    dt = time.time() - t0
    pn, pl, _ = inject_sensor_faults(df, n=100, magnitude=4.0, seed=42)
    Zn = apply_standardise(pn, mu, sd)
    sc = network_score(recon_error_matrix(model, Zn, WINDOW))
    ap = average_precision_score(pl, sc)
    return ap, best_ep, dt


def main():
    print(f"Device: {DEVICE}")
    df = preprocess.align(preprocess.clean(preprocess.load_data(config.DATA_PATH)), freq="1h")
    mu = df.mean().to_numpy("float32")
    sd = df.std(ddof=0).to_numpy("float32")
    Z = apply_standardise(df, mu, sd)
    Xall, _ = make_windows(Z, WINDOW, step=WINDOW)
    n_val = int(len(Xall) * 0.2)
    Xtr, Xval = Xall[:-n_val], Xall[-n_val:]
    print(f"train windows: {len(Xtr)}  val windows: {len(Xval)}")

    print("\n--- DEFAULT cuDNN settings (deterministic=False, benchmark=True) ---")
    ap_a, ep_a, dt_a = run_one(df, mu, sd, Xtr, Xval, deterministic=False)
    print(f"run A: AP={ap_a:.4f}  best_epoch={ep_a}  ({dt_a:.1f}s)")
    ap_b, ep_b, dt_b = run_one(df, mu, sd, Xtr, Xval, deterministic=False)
    print(f"run B: AP={ap_b:.4f}  best_epoch={ep_b}  ({dt_b:.1f}s)")
    print(f"difference: {abs(ap_a - ap_b):.4f}")

    print("\n--- DETERMINISTIC cuDNN settings (deterministic=True, benchmark=False) ---")
    ap_c, ep_c, dt_c = run_one(df, mu, sd, Xtr, Xval, deterministic=True)
    print(f"run C: AP={ap_c:.4f}  best_epoch={ep_c}  ({dt_c:.1f}s)")
    ap_d, ep_d, dt_d = run_one(df, mu, sd, Xtr, Xval, deterministic=True)
    print(f"run D: AP={ap_d:.4f}  best_epoch={ep_d}  ({dt_d:.1f}s)")
    print(f"difference: {abs(ap_c - ap_d):.6f}")

    print("\n" + "=" * 70)
    print(f"Default settings,      run-to-run |AP diff| = {abs(ap_a - ap_b):.4f}")
    print(f"Deterministic settings, run-to-run |AP diff| = {abs(ap_c - ap_d):.6f}")
    if abs(ap_c - ap_d) < 1e-4 and abs(ap_a - ap_b) > 1e-3:
        print("CONFIRMED: cuDNN determinism flags eliminate the run-to-run training variance.")
    elif abs(ap_c - ap_d) < 1e-4:
        print("Deterministic runs matched; default runs happened to match too on this attempt")
        print("(the original discrepancy was real but is not guaranteed to reproduce every attempt).")
    else:
        print("NOT CONFIRMED: deterministic settings did not eliminate the variance on this run.")
    print("=" * 70)


if __name__ == "__main__":
    main()
