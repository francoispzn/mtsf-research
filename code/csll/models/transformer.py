"""Vanilla Transformer encoder forecaster (direct multi-step), RevIN-normalised.

Embeds each timestep (N->d_model), self-attends over time, then projects the time axis
L->H and the feature axis d_model->N. Scales to high-dimensional datasets (no giant
flatten head)."""
from __future__ import annotations

import torch
import torch.nn as nn

from ..revin import RevIN


class VanillaTransformer(nn.Module):
    def __init__(self, seq_len, pred_len, n_vars, d_model=64, nhead=4, num_layers=2,
                 dim_ff=128, dropout=0.1, use_revin=True, **kwargs):
        super().__init__()
        self.revin = RevIN(n_vars) if use_revin else None
        self.embed = nn.Linear(n_vars, d_model)
        self.pos = nn.Parameter(0.02 * torch.randn(1, seq_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.time_proj = nn.Linear(seq_len, pred_len)
        self.out_proj = nn.Linear(d_model, n_vars)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.revin is not None:
            x = self.revin(x, "norm")
        h = self.embed(x) + self.pos                 # (B,L,d_model)
        h = self.encoder(h)
        h = self.time_proj(h.transpose(1, 2)).transpose(1, 2)   # (B,H,d_model)
        y = self.out_proj(h)                         # (B,H,N)
        if self.revin is not None:
            y = self.revin(y, "denorm")
        return y
