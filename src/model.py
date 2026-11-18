"""PyTorch LSTM regressor and device helpers."""
from __future__ import annotations

import torch
import torch.nn as nn


def get_device(pref: str = "auto") -> torch.device:
    """CPU / CUDA / Apple-Silicon MPS selection."""
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class LSTMRegressor(nn.Module):
    """2-layer LSTM with inter-layer dropout -> dropout -> MLP head -> scalar RUL."""

    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)           # (B, T, H)
        return self.head(out[:, -1, :]).squeeze(-1)   # use last time-step
