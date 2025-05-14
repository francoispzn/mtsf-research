"""Forecast error metrics.

Conventions (documented in the paper):
  * MSE, MAE, RMSE are computed on the STANDARDISED series (train-stat z-scored), matching
    the dominant LTSF convention (Informer/Autoformer/DLinear/PatchTST report on scaled data).
  * MASE is computed on the ORIGINAL scale (scale-free), dividing the forecast MAE by the
    in-sample seasonal-naive MAE of the training series (per channel, then averaged).
  * sMAPE is reported for completeness but is ill-defined for series that cross zero; on
    standardised data it is unreliable, so we compute it on the original scale and flag it.

All array inputs have shape (num_samples, pred_len, N) in standardised units.
`mean`, `std`, `mase_scale` have shape (N,).
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def _to_original(z: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return z * std + mean


def mse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean((pred - true) ** 2))


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true)))


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def mase(pred: np.ndarray, true: np.ndarray, mean: np.ndarray, std: np.ndarray,
         mase_scale: np.ndarray) -> float:
    p = _to_original(pred, mean, std)
    t = _to_original(true, mean, std)
    # per-channel MAE over (samples, horizon)
    per_channel = np.mean(np.abs(p - t), axis=(0, 1))       # (N,)
    return float(np.mean(per_channel / mase_scale))


def smape(pred: np.ndarray, true: np.ndarray, mean: np.ndarray, std: np.ndarray,
          eps: float = 1e-8) -> float:
    p = _to_original(pred, mean, std)
    t = _to_original(true, mean, std)
    denom = np.abs(p) + np.abs(t) + eps
    return float(np.mean(2.0 * np.abs(p - t) / denom) * 100.0)


def compute_all(pred: np.ndarray, true: np.ndarray, mean: np.ndarray, std: np.ndarray,
                mase_scale: np.ndarray) -> Dict[str, float]:
    return {
        "mse": mse(pred, true),
        "mae": mae(pred, true),
        "rmse": rmse(pred, true),
        "mase": mase(pred, true, mean, std, mase_scale),
        "smape": smape(pred, true, mean, std),
    }
