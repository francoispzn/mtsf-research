"""Linear baselines: DLinear and NLinear (Zeng et al., AAAI 2023). Channel-independent
(weights shared across channels), which is their standard strong configuration."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def series_decomp(x: torch.Tensor, kernel_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    pad = (kernel_size - 1) // 2
    xt = x.transpose(1, 2)
    left = xt[:, :, :1].repeat(1, 1, pad)
    right = xt[:, :, -1:].repeat(1, 1, kernel_size - 1 - pad)
    xp = torch.cat([left, xt, right], dim=2)
    trend = F.avg_pool1d(xp, kernel_size=kernel_size, stride=1).transpose(1, 2)
    return trend, x - trend


class DLinear(nn.Module):
    def __init__(self, seq_len, pred_len, n_vars, kernel_size=25, **kwargs):
        super().__init__()
        self.H = pred_len
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.lin_trend = nn.Linear(seq_len, pred_len)
        self.lin_seasonal = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend, seasonal = series_decomp(x, self.kernel_size)
        out = self.lin_trend(trend.transpose(1, 2)) + self.lin_seasonal(seasonal.transpose(1, 2))
        return out.transpose(1, 2)


class NLinear(nn.Module):
    def __init__(self, seq_len, pred_len, n_vars, **kwargs):
        super().__init__()
        self.lin = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last = x[:, -1:, :]
        xn = x - last
        out = self.lin(xn.transpose(1, 2)).transpose(1, 2)
        return out + last
