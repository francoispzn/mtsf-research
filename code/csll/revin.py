"""Reversible Instance Normalisation (RevIN), Kim et al., ICLR 2022.

Per-window, per-channel normalisation applied *inside* a model: subtract the look-back
mean/std, forecast in normalised space, then invert with the same statistics. Handles
non-stationary mean/variance shift between look-back and horizon. Applied on top of the
harness's global (train-stat) standardisation; the two compose harmlessly.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        self._mean = None
        self._std = None

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        # x: (B, T, N)
        if mode == "norm":
            self._mean = x.mean(dim=1, keepdim=True).detach()
            self._std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            x = (x - self._mean) / self._std
            if self.affine:
                x = x * self.gamma + self.beta
            return x
        elif mode == "denorm":
            if self.affine:
                x = (x - self.beta) / (self.gamma + 1e-8)
            x = x * self._std + self._mean
            return x
        else:
            raise ValueError(f"RevIN mode must be 'norm' or 'denorm', got {mode}")
