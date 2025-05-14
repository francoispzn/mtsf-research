"""Unified training loop (shared by CSLL and all torch baselines)."""
from __future__ import annotations

import copy
import os
import time
from typing import Dict, Tuple

import torch
import torch.nn as nn


def get_device() -> torch.device:
    dev = os.environ.get("MTSF_DEVICE")
    if dev:
        return torch.device(dev)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # NOTE: Apple's MPS backend has op-coverage gaps that abort on our custom spectral
    # einsums (Metal MPSNDArray buffer assertion). We default to CPU for reliability and
    # reproducibility; set MTSF_DEVICE=mps to opt in on machines where it works.
    return torch.device("cpu")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _has_trainable(model: nn.Module) -> bool:
    return any(p.requires_grad for p in model.parameters())


@torch.no_grad()
def _val_loss(model, loader, device, crit) -> float:
    model.eval()
    tot, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss = crit(model(x), y)
        tot += float(loss) * x.shape[0]
        n += x.shape[0]
    return tot / max(1, n)


def train_torch(model, bundle, device, cfg: Dict) -> Tuple[nn.Module, Dict]:
    """Train with Adam + early stopping on val MSE. Returns (best model, history)."""
    model = model.to(device)
    loaders = bundle.loaders(batch_size=cfg.get("batch_size", 32),
                             num_workers=cfg.get("num_workers", 0))
    crit = nn.MSELoss()

    if not _has_trainable(model):  # e.g. SeasonalNaive
        vloss = _val_loss(model, loaders["val"], device, crit)
        return model, {"epochs_run": 0, "best_val": vloss, "train_time_s": 0.0,
                       "params": 0}

    if cfg.get("xcorr_init") and hasattr(model, "init_delays_from_xcorr"):
        # v2: seed the delay positions from lagged cross-correlation on the (standardised)
        # training series before gradient training refines them. "pairwise" solves the
        # sensor-network LS system (for local/corridor lead-lag structure).
        model.init_delays_from_xcorr(bundle.train.series.numpy(),
                                     pairwise=(cfg.get("xcorr_init") == "pairwise"))

    if hasattr(model, "optim_groups"):
        # model-defined param groups (CSLL: weight-decay only on the spectral branch)
        opt = torch.optim.Adam(model.optim_groups(cfg.get("branch_wd", 1e-3)),
                               lr=cfg.get("lr", 1e-3))
    else:
        opt = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3),
                               weight_decay=cfg.get("weight_decay", 0.0))
    epochs = cfg.get("epochs", 15)
    patience = cfg.get("patience", 4)
    clip = cfg.get("grad_clip", 5.0)

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    bad = 0
    history = {"val": []}
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
        vloss = _val_loss(model, loaders["val"], device, crit)
        history["val"].append(vloss)
        if vloss < best_val - 1e-7:
            best_val = vloss
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    # Post-training gate selection. v1 forced this safeguard (undisclosed val-based selection
    # of the branch); v2 sets gate_select=False so the honestly-trained warm gate stands and
    # the branch's effect is measured, not masked. When enabled we still only RECORD the
    # would-be decision rather than mutating alpha unless asked.
    if hasattr(model, "select_gate") and cfg.get("gate_select", True):
        model.select_gate(loaders["val"], device)
    history.update({"epochs_run": epoch + 1, "best_val": best_val,
                    "train_time_s": time.time() - t0, "params": count_params(model),
                    "gate_kept": getattr(model, "_gate_kept", None)})
    return model, history
