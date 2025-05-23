"""iTransformer (Liu et al., ICLR 2024): attention over variate tokens, RevIN.

Each variate's whole look-back is embedded into one token; self-attention runs ACROSS
variates to capture cross-variate correlations; a shared FFN maps each token to the horizon."""
from __future__ import annotations

import torch
import torch.nn as nn

from ..revin import RevIN


class ITransformer(nn.Module):
    def __init__(self, seq_len, pred_len, n_vars, d_model=128, nhead=8, num_layers=3,
                 dim_ff=256, dropout=0.1, use_revin=True, **kwargs):
        super().__init__()
        self.revin = RevIN(n_vars) if use_revin else None
        self.embed = nn.Linear(seq_len, d_model)          # variate-token embedding
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.proj = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.revin is not None:
            x = self.revin(x, "norm")
        tokens = self.embed(x.transpose(1, 2))            # (B,N,d_model)
        tokens = self.encoder(tokens)                     # attention across N variates
        y = self.proj(tokens).transpose(1, 2)             # (B,H,N)
        if self.revin is not None:
            y = self.revin(y, "denorm")
        return y
