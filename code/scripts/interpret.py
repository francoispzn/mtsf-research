#!/usr/bin/env python3
"""Interpretability: does CSLL's learned delay matrix recover planted lead-lag?

(1) Train CSLL on the synthetic planted-delay data; compare learned D_ij to the ground-truth
    tau_i - tau_j (scatter + Pearson r + heatmaps).
(2) Learned lead-lag heatmap on a real low-dim dataset (exchange).
Writes figures to results/figures/ and a small JSON summary to results/tables/."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                   # noqa: E402
import torch                         # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from csll import set_seed                                    # noqa: E402
from csll.data import build_dataset                           # noqa: E402
from csll.models.csll import CSLL                             # noqa: E402
from csll.train import train_torch, get_device               # noqa: E402

FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"


def train_csll(dataset, seq_len, pred_len, n_bands, epochs, seed=0):
    set_seed(seed)
    bundle = build_dataset(dataset, seq_len, pred_len)
    net = CSLL(seq_len=seq_len, pred_len=pred_len, n_vars=bundle.n_vars,
               n_bands=n_bands, use_phase=True, use_revin=True, tau_max=seq_len / 2.0)
    net, hist = train_torch(net, bundle, get_device(),
                            dict(lr=1e-3, batch_size=32, epochs=epochs, patience=6, grad_clip=5.0))
    return net, bundle


def synthetic_delay_recovery():
    print("== synthetic delay recovery ==")
    net, bundle = train_csll("synthetic", 96, 96, n_bands=1, epochs=40)
    D_learned = net.learned_delays(band=0).cpu().numpy()
    taus = np.load(ROOT / "data" / "raw" / "synthetic.delays.npy")
    D_true = taus[:, None] - taus[None, :]
    off = ~np.eye(len(taus), dtype=bool)
    x, y = D_true[off], D_learned[off]
    r = float(np.corrcoef(x, y)[0, 1])
    slope = float(np.polyfit(x, y, 1)[0])
    print(f"  planted tau = {taus.tolist()}")
    print(f"  Pearson r(D_true, D_learned) = {r:.3f}, slope = {slope:.3f}")

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    ax[0].scatter(x, y, s=18, alpha=0.7)
    ax[0].set_xlabel(r"true $\tau_i-\tau_j$"); ax[0].set_ylabel(r"learned $D_{ij}$")
    ax[0].set_title(f"delay recovery (r={r:.2f})"); ax[0].grid(alpha=0.3)
    im1 = ax[1].imshow(D_true, cmap="RdBu", vmin=-20, vmax=20); ax[1].set_title(r"true $\tau_i-\tau_j$")
    im2 = ax[2].imshow(D_learned, cmap="RdBu", vmin=-20, vmax=20); ax[2].set_title(r"learned $D_{ij}$")
    fig.colorbar(im1, ax=ax[1], fraction=0.046); fig.colorbar(im2, ax=ax[2], fraction=0.046)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "delay_recovery.pdf")
    print("  wrote delay_recovery.pdf")
    return {"pearson_r": r, "slope": slope, "planted_tau": taus.tolist()}


def real_delay_heatmap(dataset="exchange"):
    print(f"== learned lead-lag on {dataset} ==")
    net, bundle = train_csll(dataset, 96, 96, n_bands=4, epochs=20)
    D = net.learned_delays(band=0).cpu().numpy()
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    lim = np.abs(D).max() + 1e-6
    im = ax.imshow(D, cmap="RdBu", vmin=-lim, vmax=lim)
    ax.set_title(f"CSLL learned delays $D_{{ij}}$ ({dataset}, band 0)")
    ax.set_xlabel("source j"); ax.set_ylabel("target i")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIG / f"delays_{dataset}.pdf")
    print(f"  wrote delays_{dataset}.pdf")


if __name__ == "__main__":
    summary = synthetic_delay_recovery()
    try:
        real_delay_heatmap("exchange")
    except Exception as e:
        print("real heatmap skipped:", e)
    TAB.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(TAB / "delay_recovery.json", "w"), indent=2)
    print("INTERPRET_DONE")
