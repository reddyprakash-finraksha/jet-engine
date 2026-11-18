"""Inference helpers shared by evaluate.py, the notebooks and the Streamlit app."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from .config import ALL_COLS, ARTIFACT_DIR, Config
from .data import last_window, read_cmapss
from .model import LSTMRegressor, get_device


@dataclass
class Predictor:
    model: LSTMRegressor
    scaler: object
    features: list
    window: int
    cap: int
    device: torch.device
    meta: dict

    # ---- single engine: RUL after its last observed cycle --------------------
    def predict_last(self, engine_df: pd.DataFrame) -> float:
        arr = self.scaler.transform(engine_df.sort_values("cycle")[self.features].to_numpy())
        x = torch.from_numpy(last_window(arr, self.window).astype(np.float32))[None].to(self.device)
        with torch.no_grad():
            return float(np.clip(self.model(x).item() * self.cap, 0, self.cap))

    # ---- rolling prediction at every cycle >= window --------------------------
    def predict_rolling(self, engine_df: pd.DataFrame) -> pd.DataFrame:
        g = engine_df.sort_values("cycle")
        arr = self.scaler.transform(g[self.features].to_numpy()).astype(np.float32)
        cycles = g["cycle"].to_numpy()
        wins = np.stack([last_window(arr[:i + 1], self.window) for i in range(len(arr))])
        with torch.no_grad():
            p = self.model(torch.from_numpy(wins).to(self.device)).cpu().numpy()
        return pd.DataFrame({"cycle": cycles, "pred_rul": np.clip(p * self.cap, 0, self.cap)})


def load_predictor(artifact_dir: Path | str = ARTIFACT_DIR, device: str = "cpu") -> Predictor:
    artifact_dir = Path(artifact_dir)
    ckpt = torch.load(artifact_dir / "model.pt", map_location="cpu", weights_only=False)
    cfg = Config(**ckpt["config"])
    model = LSTMRegressor(ckpt["n_features"], cfg.hidden_size, cfg.num_layers, cfg.dropout)
    model.load_state_dict(ckpt["state_dict"])
    dev = get_device(device)
    model.to(dev).eval()
    meta = json.loads((artifact_dir / "metadata.json").read_text())
    return Predictor(model, joblib.load(artifact_dir / "scaler.joblib"), meta["features"],
                     cfg.window, cfg.rul_cap, dev, meta)


def parse_uploaded_engine(file_like) -> pd.DataFrame:
    """Parse an uploaded C-MAPSS style text/CSV (26 cols, whitespace or comma separated).
    Missing unit column (25 cols) is tolerated."""
    df = pd.read_csv(file_like, sep=r"[\s,]+", header=None, engine="python").dropna(axis=1, how="all")
    if df.shape[1] == len(ALL_COLS) - 1:
        df.insert(0, "unit", 1)
    if df.shape[1] != len(ALL_COLS):
        raise ValueError(f"Expected 26 columns (unit, cycle, 3 op settings, 21 sensors); got {df.shape[1]}.")
    df.columns = ALL_COLS
    return df
