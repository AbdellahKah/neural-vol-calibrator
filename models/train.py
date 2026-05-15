"""
models/train.py
---------------
Training loop for HestonCalibrator.

Features:
  - Loads pre-generated dataset (X.npy, y.npy) or generates on the fly
  - Train / val / test split
  - MSE loss on normalised parameters
  - Adam + CosineAnnealingLR scheduler
  - Early stopping (patience-based)
  - Saves best checkpoint + normalisation stats
  - Prints per-epoch metrics; final test RMSE per parameter

Author: Abdellah Kahlaoui
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Optional

# Adjust import paths when running as module vs script
try:
    from models.network import HestonCalibrator
    from data.surfaces import normalize_surfaces, normalize_params, GRID_SIZE
    from data.heston import PARAM_BOUNDS
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from models.network import HestonCalibrator
    from data.surfaces import normalize_surfaces, normalize_params, GRID_SIZE
    from data.heston import PARAM_BOUNDS


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HestonSurfaceDataset(Dataset):
    """
    Wraps (X, y) numpy arrays.

    X : (N, grid_size)  — z-scored IV surfaces
    y : (N, 5)          — [0,1] normalised Heston params
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Early stopping helper
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 1e-6):
        self.patience  = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter   = 0
        self.triggered = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train(
    # --- data ---
    X: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    data_dir: str = "data/saved",
    # --- model ---
    hidden_dims: tuple = (256, 128, 64),
    dropout: float = 0.1,
    # --- training ---
    n_epochs: int = 200,
    batch_size: int = 512,
    lr: float = 3e-4,
    weight_decay: float = 1e-5,
    patience: int = 25,
    # --- splits ---
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    # --- output ---
    checkpoint_path: str = "models/best_model.pt",
    seed: int = 42,
    device: Optional[str] = None,
) -> dict:
    """
    Train the HestonCalibrator.

    Data priority:
      1. X, y passed directly as numpy arrays
      2. Loaded from data_dir/X.npy and data_dir/y.npy

    Returns a dict with training history and test metrics.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ------------------------------------------------------------------ #
    # 1. Load / validate data
    # ------------------------------------------------------------------ #
    if X is None or y is None:
        x_path = os.path.join(data_dir, "X.npy")
        y_path = os.path.join(data_dir, "y.npy")
        if not os.path.exists(x_path):
            raise FileNotFoundError(
                f"No data found at {x_path}. "
                "Run data/surface.py generate_dataset() first."
            )
        X = np.load(x_path)
        y = np.load(y_path)
        print(f"Loaded dataset: X={X.shape}, y={y.shape}")

    N = len(X)
    print(f"Dataset: {N:,} surfaces  |  grid_size={X.shape[1]}  |  params={y.shape[1]}")

    # ------------------------------------------------------------------ #
    # 2. Normalise
    # ------------------------------------------------------------------ #
    # Split indices first so normalisation stats come from train only
    n_test = max(1, int(N * test_frac))
    n_val  = max(1, int(N * val_frac))
    n_train = N - n_val - n_test

    idx = np.random.permutation(N)
    idx_train = idx[:n_train]
    idx_val   = idx[n_train:n_train + n_val]
    idx_test  = idx[n_train + n_val:]

    # Z-score surfaces (fit on train, apply to all)
    X_train_raw = X[idx_train]
    X_norm, surf_mean, surf_std = normalize_surfaces(X_train_raw)
    X_norm_full, _, _ = normalize_surfaces(X, surf_mean, surf_std)

    # [0,1] params
    y_norm = normalize_params(y)

    X_train = X_norm_full[idx_train]
    X_val   = X_norm_full[idx_val]
    X_test  = X_norm_full[idx_test]
    y_train = y_norm[idx_train]
    y_val   = y_norm[idx_val]
    y_test  = y_norm[idx_test]

    print(f"Split — train: {n_train:,}  val: {n_val:,}  test: {n_test:,}")

    # ------------------------------------------------------------------ #
    # 3. DataLoaders
    # ------------------------------------------------------------------ #
    train_ds = HestonSurfaceDataset(X_train, y_train)
    val_ds   = HestonSurfaceDataset(X_val,   y_val)
    test_ds  = HestonSurfaceDataset(X_test,  y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)

    # ------------------------------------------------------------------ #
    # 4. Model, optimiser, scheduler
    # ------------------------------------------------------------------ #
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    print(f"Device: {dev}\n")

    model = HestonCalibrator(
        input_dim=X.shape[1],
        hidden_dims=hidden_dims,
        output_dim=y.shape[1],
        dropout=dropout,
    ).to(dev)
    print(f"Model params: {model.count_params():,}\n")

    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=n_epochs, eta_min=lr * 0.01
    )
    stopper = EarlyStopping(patience=patience)

    # ------------------------------------------------------------------ #
    # 5. Training loop
    # ------------------------------------------------------------------ #
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)

    print(f"{'Epoch':>6}  {'Train MSE':>10}  {'Val MSE':>10}  "
          f"{'LR':>8}  {'Time':>6}")
    print("-" * 52)

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        # -- train --
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            optimiser.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            train_loss += loss.item() * len(xb)
        train_loss /= n_train

        # -- validate --
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= n_val

        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        lr_now = optimiser.param_groups[0]["lr"]

        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:>6}  {train_loss:>10.6f}  {val_loss:>10.6f}  "
                  f"{lr_now:>8.2e}  {elapsed:>5.1f}s")

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "val_loss":   val_loss,
                "surf_mean":  surf_mean,
                "surf_std":   surf_std,
                "hidden_dims": hidden_dims,
                "dropout":    dropout,
                "input_dim":  X.shape[1],
                "output_dim": y.shape[1],
            }, checkpoint_path)

        if stopper.step(val_loss):
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {patience} epochs).")
            break

    print(f"\nBest val MSE: {best_val_loss:.6f}")

    # ------------------------------------------------------------------ #
    # 6. Test evaluation
    # ------------------------------------------------------------------ #
    ckpt = torch.load(checkpoint_path, map_location=dev)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_pred, all_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(dev)
            all_pred.append(model(xb).cpu().numpy())
            all_true.append(yb.numpy())

    pred = np.vstack(all_pred)   # (n_test, 5)  — normalised
    true = np.vstack(all_true)

    # Per-parameter RMSE in normalised space
    param_names = ["v0", "kappa", "theta", "sigma", "rho"]
    rmse_norm = np.sqrt(((pred - true) ** 2).mean(axis=0))

    # Denormalise to real-world values for interpretable error
    from data.surfaces import denormalize_params
    pred_real = denormalize_params(pred)
    true_real = denormalize_params(true)
    rmse_real = np.sqrt(((pred_real - true_real) ** 2).mean(axis=0))
    mae_real  = np.abs(pred_real - true_real).mean(axis=0)

    print("\n=== Test Results ===")
    print(f"{'Param':>6}  {'RMSE (norm)':>12}  {'RMSE (real)':>12}  {'MAE (real)':>10}")
    print("-" * 48)
    for name, rn, rr, mr in zip(param_names, rmse_norm, rmse_real, mae_real):
        print(f"{name:>6}  {rn:>12.6f}  {rr:>12.6f}  {mr:>10.6f}")

    results = {
        "history":    history,
        "best_val_loss": best_val_loss,
        "test_rmse_norm": dict(zip(param_names, rmse_norm.tolist())),
        "test_rmse_real": dict(zip(param_names, rmse_real.tolist())),
        "test_mae_real":  dict(zip(param_names, mae_real.tolist())),
        "checkpoint":    checkpoint_path,
        "surf_mean":     surf_mean,
        "surf_std":      surf_std,
    }
    return results


# ---------------------------------------------------------------------------
# Checkpoint loader (for inference)
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: str = "cpu") -> tuple:
    """
    Load a saved HestonCalibrator from a checkpoint.

    Returns (model, surf_mean, surf_std).
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = HestonCalibrator(
        input_dim=ckpt["input_dim"],
        hidden_dims=ckpt["hidden_dims"],
        output_dim=ckpt["output_dim"],
        dropout=ckpt["dropout"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["surf_mean"], ckpt["surf_std"]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train HestonCalibrator")
    parser.add_argument("--data-dir",   default="data/saved",
                        help="Directory with X.npy and y.npy")
    parser.add_argument("--epochs",     type=int,   default=200)
    parser.add_argument("--batch-size", type=int,   default=512)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--patience",   type=int,   default=25)
    parser.add_argument("--checkpoint", default="models/best_model.pt")
    parser.add_argument("--device",     default=None)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        checkpoint_path=args.checkpoint,
        device=args.device,
        seed=args.seed,
    )