"""LSTM sequence-to-sequence forecaster (direct multi-step). RevIN-normalised."""
from __future__ import annotations

import torch
import torch.nn as nn

from ..revin import RevIN


class LSTMForecaster(nn.Module):
    def __init__(self, seq_len, pred_len, n_vars, hidden_size=128, num_layers=2,
                 dropout=0.1, use_revin=True, **kwargs):
        super().__init__()
        self.H, self.N = pred_len, n_vars
        self.revin = RevIN(n_vars) if use_revin else None
        self.lstm = nn.LSTM(
            input_size=n_vars, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(hidden_size, pred_len * n_vars)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.revin is not None:
            x = self.revin(x, "norm")
        out, _ = self.lstm(x)
        last = out[:, -1]                       # (B, hidden)
        y = self.proj(last).view(x.shape[0], self.H, self.N)
        if self.revin is not None:
            y = self.revin(y, "denorm")
        return y
