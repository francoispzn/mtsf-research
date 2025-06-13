#!/usr/bin/env python3
"""Generate publication-quality figures from the aggregated result tables.
Writes vector PDFs (and PNGs for the README) to results/figures/."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from csll.plotstyle import apply, PALETTE   # noqa: E402

apply()
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"

# consistent per-model colour + style across every figure
MODEL_STYLE = {
    "CSLL":         dict(color=PALETTE["csll"],   marker="o", ls="-",  lw=2.2, z=5),
    "iTransformer": dict(color=PALETTE["green"],  marker="D", ls="--", lw=1.4, z=3),
    "PatchTST":     dict(color=PALETTE["blue"],   marker="^", ls="--", lw=1.4, z=3),
    "DLinear":      dict(color=PALETTE["grey"],   marker="s", ls=":",  lw=1.3, z=2),
    "NLinear":      dict(color=PALETTE["teal"],   marker="v", ls=":",  lw=1.3, z=2),
}
PLOT_MODELS = ["DLinear", "NLinear", "PatchTST", "iTransformer", "CSLL"]


def fig_mse_vs_horizon():
    f = TAB / "main_mse.csv"
    if not f.exists():
        print("main_mse.csv missing; run aggregate.py first"); return
    df = pd.read_csv(f)
    # keep datasets with a full horizon sweep (>=4 points) for clean curves
    counts = df.groupby("dataset")["pred_len"].nunique()
    datasets = [d for d in df["dataset"].unique() if counts.get(d, 0) >= 4]
    order = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "exchange", "ili"]
    datasets = [d for d in order if d in datasets] + [d for d in datasets if d not in order]
    ncol = 4
    nrow = int(np.ceil(len(datasets) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.55 * ncol, 2.1 * nrow), squeeze=False)
    handles = {}
    for i, ds in enumerate(datasets):
        ax = axes[i // ncol][i % ncol]
        sub = df[df.dataset == ds].sort_values("pred_len")
        for m in PLOT_MODELS:
            if m in sub.columns and sub[m].notna().any():
                st = MODEL_STYLE[m]
                (h,) = ax.plot(sub["pred_len"], sub[m], st["ls"], color=st["color"],
                               marker=st["marker"], ms=4, lw=st["lw"], zorder=st["z"], label=m)
                handles[m] = h
        ax.set_title(ds, fontsize=9.5)
        ax.set_xlabel("horizon $H$"); ax.set_ylabel("MSE")
        ax.margins(x=0.05)
    for j in range(len(datasets), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    # single shared legend in the spare panel (or below)
    leg_ax = axes[(len(datasets)) // ncol][(len(datasets)) % ncol] if len(datasets) < nrow * ncol else None
    if leg_ax is not None:
        leg_ax.legend([handles[m] for m in PLOT_MODELS if m in handles],
                      [m for m in PLOT_MODELS if m in handles],
                      loc="center", fontsize=9, frameon=True, title="model")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "mse_vs_horizon.pdf")
    fig.savefig(FIG / "mse_vs_horizon.png")
    plt.close(fig)
    print("wrote mse_vs_horizon.pdf")


def fig_ablation():
    f = TAB / "ablation_mse.csv"
    if not f.exists():
        print("ablation_mse.csv missing"); return
    df = pd.read_csv(f)
    variants = [c for c in df.columns if c not in ("dataset", "pred_len")]
    d = df.dropna(subset=[v for v in variants if v != "full"])
    # mean relative to the full model (%), averaged over settings; + = worse than full
    rel = pd.DataFrame({v: (d[v] / d["full"] - 1.0) * 100 for v in variants})
    means = rel.mean(axis=0).sort_values()
    labels = {"full": "full (dynamic)", "As_static": "static delay",
              "A1_realphase": "phase-off", "A0_backbone": "no branch",
              "A2_1band": "1 band", "A4_2band": "2 bands", "A4_8band": "8 bands",
              "A3_freecplx": "free complex", "A5_lowrank8": "low-rank 8"}
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    colors = [PALETTE["csll"] if v == "full" else PALETTE["blue"] for v in means.index]
    ax.bar(range(len(means)), means.values, color=colors, width=0.7, edgecolor="#222", linewidth=0.5)
    ax.axhline(0, color="#222", lw=0.8)
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels([labels.get(v, v) for v in means.index], rotation=35, ha="right")
    ax.set_ylabel(r"mean $\Delta$MSE vs. full (%)")
    ax.set_title("Ablation: change relative to the full model (5 datasets)")
    ax.grid(alpha=0.5, axis="y")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "ablation.pdf")
    fig.savefig(FIG / "ablation.png")
    plt.close(fig)
    print("wrote ablation.pdf")


def fig_pareto():
    f = TAB / "summary_ranked.csv"
    if not f.exists():
        print("summary_ranked.csv missing; run aggregate.py first"); return
    df = pd.read_csv(f, index_col=0)
    df = df[df["params"] > 0]                       # drop param-free SeasonalNaive/VAR

    def frontier(sub, xcol):
        keep = []
        for m, r in sub.iterrows():
            dom = any((sub.loc[o, "avg_rank"] <= r["avg_rank"] and sub.loc[o, xcol] <= r[xcol]
                       and (sub.loc[o, "avg_rank"] < r["avg_rank"] or sub.loc[o, xcol] < r[xcol]))
                      for o in sub.index if o != m)
            if not dom:
                keep.append(m)
        return sub.loc[keep].sort_values(xcol)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))
    for ax, xcol, xlabel in [(axes[0], "params", "parameters (median)"),
                             (axes[1], "train_s", "train time / run (s)")]:
        fr = frontier(df, xcol)
        ax.plot(fr[xcol], fr["avg_rank"], "--", color=PALETTE["grey"], lw=1.1, zorder=1)
        ax.fill_between(fr[xcol], fr["avg_rank"], df["avg_rank"].max() + 0.4,
                        color=PALETTE["grey"], alpha=0.05, zorder=0)
        for m, r in df.iterrows():
            csll = m.startswith("CSLL")
            ax.scatter(r[xcol], r["avg_rank"], s=190 if csll else 62,
                       c=(PALETTE["csll"] if csll else PALETTE["blue"]),
                       marker=("*" if csll else "o"), edgecolors="#111", linewidths=0.7,
                       zorder=5 if csll else 4)
            ax.annotate(m, (r[xcol], r["avg_rank"]), fontsize=7.5,
                        xytext=(5, 4), textcoords="offset points",
                        fontweight=("bold" if csll else "normal"))
        ax.set_xscale("log")
        ax.set_xlabel(xlabel); ax.set_ylabel("average rank (lower = better)")
        ax.invert_yaxis()
        ax.margins(x=0.16)
    axes[0].set_title("Accuracy vs. model size")
    axes[1].set_title("Accuracy vs. training cost")
    fig.suptitle("CSLL on the accuracy–cost Pareto frontier (7 datasets, 28 settings)",
                 fontsize=10.5, y=1.02)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "pareto.pdf")
    fig.savefig(FIG / "pareto.png")
    plt.close(fig)
    print("wrote pareto.pdf + pareto.png")


if __name__ == "__main__":
    fig_mse_vs_horizon()
    fig_ablation()
    fig_pareto()
