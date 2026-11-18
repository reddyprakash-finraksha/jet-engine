"""Train the LSTM RUL model.

Usage:  python -m src.train [--epochs 100] [--device auto|cpu|mps|cuda]
"""
from __future__ import annotations

import argparse
import copy
import json
import time

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .config import ARTIFACT_DIR, Config
from .data import prepare
from .model import LSTMRegressor, get_device


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


@torch.no_grad()
def predict_cycles(model: nn.Module, X: np.ndarray, cap: int, device, bs: int = 2048) -> np.ndarray:
    """Model works on RUL/cap; return predictions in cycles, clipped to [0, cap]."""
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(device)
        out.append(model(xb).cpu().numpy())
    return np.clip(np.concatenate(out) * cap, 0, cap)


def train(cfg: Config, verbose: bool = True) -> dict:
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    d = prepare(cfg)
    n_feat = d["X_train"].shape[2]
    if verbose:
        print(f"device={device} | features={n_feat} | train windows={len(d['X_train'])} "
              f"| val windows={len(d['X_val'])} | test engines={len(d['X_test'])}")
        print(f"constant columns dropped: {d['constant_cols']}")
        print(f"features used: {d['features']}")

    cap = cfg.rul_cap
    ytr = torch.from_numpy(d["y_train"] / cap)
    xtr = torch.from_numpy(d["X_train"])
    loader = DataLoader(TensorDataset(xtr, ytr), batch_size=cfg.batch_size, shuffle=True)

    model = LSTMRegressor(n_feat, cfg.hidden_size, cfg.num_layers, cfg.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
    loss_fn = nn.MSELoss()

    history = {"train_rmse": [], "val_rmse": [], "lr": []}
    best_val, best_state, bad = float("inf"), None, 0
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        sq, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sq += loss.item() * len(xb)
            n += len(xb)
        train_rmse = float(np.sqrt(sq / n)) * cap
        val_rmse = rmse(predict_cycles(model, d["X_val"], cap, device), d["y_val"])
        sched.step(val_rmse)
        history["train_rmse"].append(train_rmse)
        history["val_rmse"].append(val_rmse)
        history["lr"].append(opt.param_groups[0]["lr"])
        improved = val_rmse < best_val - 1e-4
        if improved:
            best_val, best_state, bad = val_rmse, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        if verbose and (epoch == 1 or epoch % 5 == 0 or improved and epoch < 10):
            print(f"epoch {epoch:3d} | train RMSE {train_rmse:6.2f} | val RMSE {val_rmse:6.2f}"
                  f"{' *' if improved else ''}")
        if bad >= cfg.patience:
            if verbose:
                print(f"early stopping at epoch {epoch} (best val RMSE {best_val:.3f})")
            break

    model.load_state_dict(best_state)
    ARTIFACT_DIR.mkdir(exist_ok=True)
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "n_features": n_feat, "config": cfg.to_dict()}, ARTIFACT_DIR / "model.pt")
    joblib.dump(d["scaler"], ARTIFACT_DIR / "scaler.joblib")
    meta = {"features": d["features"], "constant_cols": d["constant_cols"],
            "train_units": d["train_units"].tolist(), "val_units": d["val_units"].tolist(),
            "best_val_rmse": best_val, "epochs_run": len(history["val_rmse"]),
            "train_seconds": round(time.time() - t0, 1), "device": str(device),
            "config": cfg.to_dict()}
    (ARTIFACT_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    (ARTIFACT_DIR / "history.json").write_text(json.dumps(history))
    if verbose:
        print(f"saved model + scaler + metadata to {ARTIFACT_DIR}")
    return {"model": model, "data": d, "history": history, "meta": meta, "device": device}


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    base = Config()
    for k, v in base.to_dict().items():
        if isinstance(v, bool):
            p.add_argument(f"--{k.replace('_', '-')}", action=argparse.BooleanOptionalAction, default=v)
        else:
            p.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    return Config(**{k.replace("-", "_"): v for k, v in vars(p.parse_args()).items()})


if __name__ == "__main__":
    train(parse_args())
