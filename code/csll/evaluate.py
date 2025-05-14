"""Evaluation: streaming metrics (memory-safe for high-dim/long-horizon) + per-window
losses for Diebold-Mariano tests."""
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import torch

from . import metrics as M


def metrics_from_arrays(preds: np.ndarray, trues: np.ndarray, bundle) -> Tuple[Dict[str, float], np.ndarray]:
    """Compute metrics from full arrays (used by VAR on low-dim datasets)."""
    m = M.compute_all(preds, trues, bundle.mean, bundle.std, bundle.mase_scale)
    per_window = ((preds - trues) ** 2).mean(axis=(1, 2))
    return m, per_window


@torch.no_grad()
def score_torch(model, loader, device, bundle) -> Tuple[Dict[str, float], np.ndarray]:
    """Stream over a loader, accumulating metrics without materialising all predictions."""
    model.eval()
    mean = torch.as_tensor(bundle.mean, device=device)
    std = torch.as_tensor(bundle.std, device=device)
    mase_scale = bundle.mase_scale

    se = ae = 0.0
    count = 0
    ch_abs = np.zeros(bundle.n_vars, dtype=np.float64)
    ch_count = 0
    smape_sum = 0.0
    smape_count = 0
    per_window = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        p = model(x)                                  # (b,H,N) standardised
        err = p - y
        se += float((err ** 2).sum())
        ae += float(err.abs().sum())
        count += err.numel()
        per_window.append((err ** 2).mean(dim=(1, 2)).detach().cpu().numpy())
        po = p * std + mean                           # original scale
        yo = y * std + mean
        ch_abs += (po - yo).abs().sum(dim=(0, 1)).detach().cpu().numpy().astype(np.float64)
        ch_count += po.shape[0] * po.shape[1]
        denom = po.abs() + yo.abs() + 1e-8
        smape_sum += float((2.0 * (po - yo).abs() / denom).sum())
        smape_count += po.numel()

    mse = se / count
    mae = ae / count
    metrics = {
        "mse": mse,
        "mae": mae,
        "rmse": math.sqrt(mse),
        "mase": float(np.mean((ch_abs / ch_count) / mase_scale)),
        "smape": 100.0 * smape_sum / smape_count,
    }
    return metrics, np.concatenate(per_window) if per_window else np.zeros(0)
