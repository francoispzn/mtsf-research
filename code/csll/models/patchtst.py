"""PatchTST (Nie et al., ICLR 2023): channel-independent patched Transformer, RevIN."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..revin import RevIN


class PatchTST(nn.Module):
    def __init__(self, seq_len, pred_len, n_vars, patch_len=16, stride=8, d_model=64,
                 nhead=4, num_layers=3, dim_ff=128, dropout=0.1, use_revin=True, **kwargs):
        super().__init__()
        self.H, self.N = pred_len, n_vars
        self.patch_len, self.stride, self.pad = patch_len, stride, stride
        self.num_patches = (seq_len + self.pad - patch_len) // stride + 1
        self.revin = RevIN(n_vars) if use_revin else None
        self.embed = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(0.02 * torch.randn(1, self.num_patches, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(self.num_patches * d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.revin is not None:
            x = self.revin(x, "norm")
        B, L, N = x.shape
        xt = x.transpose(1, 2)                                   # (B,N,L)
        xt = F.pad(xt, (0, self.pad), mode="replicate")
        patches = xt.unfold(dimension=2, size=self.patch_len, step=self.stride)  # (B,N,P,patch_len)
        p = patches.reshape(B * N, self.num_patches, self.patch_len)
        h = self.embed(p) + self.pos                            # (B*N,P,d_model)
        h = self.encoder(h)
        h = h.reshape(B * N, -1)
        y = self.head(h).reshape(B, N, self.H).transpose(1, 2)  # (B,H,N)
        if self.revin is not None:
            y = self.revin(y, "denorm")
        return y
