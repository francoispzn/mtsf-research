#!/usr/bin/env python3
"""Delay-validation figures (honest: exact on synthetic, partial on real).

  fig_killtest.pdf   synthetic planted-delay recovery: learned vs true (v2 exact, v1 collapsed)
  fig_delay_real.pdf (left) METR-LA learned delay vs road distance; (right) US-state ILI
                     input-conditioned delay scale vs empirical inter-region lag.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import torch                             # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from csll.plotstyle import apply, PALETTE, COL_W, DBL_W       # noqa: E402
apply()
from csll.config import model_config, seq_len_for            # noqa: E402
from csll.data import load_matrix, split_indices             # noqa: E402
from csll.models import build_model                          # noqa: E402

TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"
RAW = ROOT / "results" / "raw"
FIG.mkdir(parents=True, exist_ok=True)


def fig_killtest():
    """Synthetic recovery: v2 (exact) vs v1 (collapsed), from controlled_v2.json."""
    f = TAB / "controlled_v2.json"
    if not f.exists():
        print("skip killtest fig (no controlled_v2.json)"); return
    d = json.load(open(f))
    A = d.get("A_in_range", {})
    v2 = A.get("v2-static", {}).get("recovery")
    v1 = A.get("v1-static", {}).get("recovery")
    if not v2:
        print("skip killtest fig (no recovery data)"); return
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.86))
    t = np.array(v2["true_tau"], float)
    lo, hi = t.min() - 3, t.max() + 3
    ax.plot([lo, hi], [lo, hi], ":", color="#999", lw=1.2, label=r"ideal $D=\tau$", zorder=2)
    if v1:
        ax.scatter(v1["true_tau"], v1["learned_leader_col"], s=52, color="#8a8a8a",
                   marker="x", linewidths=1.4, zorder=3,
                   label=f"naive est. (slope {v1['slope']:.2f})")
    ax.scatter(t, v2["learned_leader_col"], s=62, color=PALETTE["csll"], edgecolors="#1a1a1a",
               linewidths=0.7, zorder=5,
               label=f"full est. (slope {v2['slope']:.2f}, $r{{=}}${v2['r_leader_col']:.2f})")
    ax.set_xlabel(r"planted delay $\tau_i$ (samples)")
    ax.set_ylabel(r"learned delay $D_{i,\,\mathrm{leader}}$")
    ax.set_title("Delay recovery (controlled)")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_xlim(lo, hi)
    fig.tight_layout()
    fig.savefig(FIG / "fig_killtest.pdf"); fig.savefig(FIG / "fig_killtest.png", dpi=200)
    plt.close(fig); print("wrote fig_killtest")


def _metrla_delay_vs_distance():
    ck = RAW / "metr_la__CSLL2__L96__H12__s0.pt"
    if not ck.exists():
        return None
    ids_df = load_matrix("metr_la")
    import pandas as pd
    df = pd.read_csv(ROOT / "data" / "raw" / "metr_la.csv", nrows=1)
    ids = [c for c in df.columns if c != "date"]
    dist = pd.read_csv(ROOT / "data" / "raw" / "st" / "distances_la_2012.csv")
    dist.columns = ["frm", "to", "cost"]
    dist["frm"] = dist.frm.astype(float).astype(int).astype(str)
    dist["to"] = dist.to.astype(float).astype(int).astype(str)
    s = set(ids); dist = dist[dist.frm.isin(s) & dist.to.isin(s) & (dist.cost > 0) & (dist.cost < 5000)]
    col = {x: i for i, x in enumerate(ids)}
    cfg = model_config("CSLL2", "metr_la", 96, 12, len(ids))
    net = build_model("CSLL2", cfg); net.load_state_dict(torch.load(ck, map_location="cpu")); net.eval()
    D = net.learned_delays(band=None).detach().numpy()
    ai = np.array([col[a] for a in dist.frm]); bi = np.array([col[b] for b in dist.to])
    return dist.cost.to_numpy() / 1000.0, D[bi, ai]


def fig_delay_real():
    md = _metrla_delay_vs_distance()
    fig, ax = plt.subplots(1, 2, figsize=(DBL_W * 0.74, COL_W * 0.82))
    if md is not None:
        dk, dl = md
        ax[0].scatter(dk, dl, s=9, alpha=0.30, color=PALETTE["blue"], edgecolors="none", zorder=3)
        # binned median trend to show the (weak) relationship cleanly
        bins = np.linspace(dk.min(), dk.max(), 9)
        idx = np.digitize(dk, bins)
        bx = [dk[idx == i].mean() for i in range(1, len(bins)) if (idx == i).sum() > 3]
        by = [np.median(dl[idx == i]) for i in range(1, len(bins)) if (idx == i).sum() > 3]
        ax[0].plot(bx, by, "-o", color=PALETTE["ink"], lw=1.4, ms=3.5, zorder=5, label="binned median")
        r = np.corrcoef(dk, dl)[0, 1]
        ax[0].set_xlabel("road distance (km)")
        ax[0].set_ylabel("learned delay (samples)")
        ax[0].set_title(f"METR-LA: weak, noisy ($r{{=}}${r:.2f})")
        ax[0].legend(fontsize=6.8, loc="upper right")
    ck = RAW / "ili_us_state__CSLL2__L36__H8__s0.pt"
    if ck.exists():
        X = load_matrix("ili_us_state"); N = X.shape[1]
        cfg = model_config("CSLL2", "ili_us_state", 36, 8, N)
        net = build_model("CSLL2", cfg); net.load_state_dict(torch.load(ck, map_location="cpu")); net.eval()
        from csll.data import build_dataset
        b = build_dataset("ili_us_state", 36, 8)
        xs = torch.stack([b.test[i][0] for i in range(min(300, len(b.test)))])
        with torch.no_grad():
            D = np.mean([net.delays_for_input(xs, band=bd).numpy() for bd in range(net.n_bands)], 0)
        off = ~np.eye(N, dtype=bool)
        vals = np.abs(D[off]).ravel()
        ax[1].hist(vals, bins=28, color=PALETTE["csll"], alpha=0.85, edgecolor="white", linewidth=0.3)
        ax[1].axvline(3.7, color=PALETTE["ink"], ls="--", lw=1.3, label="empirical lag 3.7 wk")
        ax[1].axvline(np.median(vals), color="#159090", ls="-", lw=1.3,
                      label=f"learned median {np.median(vals):.1f} wk")
        ax[1].set_xlabel(r"$|$learned delay$|$ (weeks)")
        ax[1].set_ylabel("pair count")
        ax[1].set_title("US-state ILI: epidemic scale")
        ax[1].legend(fontsize=6.8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "fig_delay_real.pdf"); fig.savefig(FIG / "fig_delay_real.png", dpi=200)
    plt.close(fig); print("wrote fig_delay_real")


if __name__ == "__main__":
    fig_killtest()
    fig_delay_real()
    print("FIG_DELAYS_DONE")
