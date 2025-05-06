"""CSLL: Complex Spectral Lead-Lag Network for multivariate time series forecasting.

Research code accompanying the paper. All experiments run through a single shared
data/training/evaluation harness (see `data.py`, `train.py`, `evaluate.py`) so that the
proposed method and every baseline are compared under identical conditions.
"""
from __future__ import annotations

__version__ = "0.1.0"

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and torch (incl. CUDA/MPS) for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # MPS shares the CPU generator seed path via torch.manual_seed.
    except Exception:
        pass
