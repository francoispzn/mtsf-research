"""Seasonal-naive baseline: repeat the last observed season. No trainable parameters."""
from __future__ import annotations

import torch
import torch.nn as nn


class SeasonalNaive(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, n_vars: int, period: int = 24, **kwargs):
        super().__init__()
        self.L, self.H, self.N = seq_len, pred_len, n_vars
        self.period = max(1, int(period))
        # dummy parameter so optimiser/.to(device) behave uniformly (unused).
        self._noop = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L, H = x.shape[1], self.H
        m = self.period if self.period <= L else 1
        idxs = [max(0, min(L - 1, L - m + (k % m))) for k in range(H)]
        idx = torch.tensor(idxs, device=x.device, dtype=torch.long)
        return x.index_select(1, idx)  # (B,H,N)
