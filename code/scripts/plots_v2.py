#!/usr/bin/env python3
"""v2 figures (publication pass). Robust to partial results.

  fig_envelope.pdf   the honest negative: phase-vs-mixing gain against the a-priori Delta(H);
                     influenza (phase helps) sits near Delta=0, traffic (mixing) at Delta<<0.
  fig_mechanism.pdf  mean |learned delay| vs horizon on METR-LA (the D(H) curve).
  fig_pareto_v2.pdf  accuracy (avg rank on traffic) vs parameters; CSLL on the cheap frontier.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from csll.plotstyle import apply, PALETTE, MODEL_STYLE, DATASET_COLOR, COL_W, DBL_W, style  # noqa: E402

apply()
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def _save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"wrote {name}")


def fig_envelope():
    f = TAB / "v2_envelope.csv"
    if not f.exists():
        print("skip envelope"); return
    df = pd.read_csv(f).dropna(subset=["phase_gain_pct", "delta"])
    if df.empty:
        print("skip envelope (empty)"); return
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.82))
    # quadrant shading: where sign(Delta) would (wrongly) predict the outcome
    ax.axhspan(0, 100, color="#f4dede", alpha=0.5, zorder=0)      # phase-helps band
    ax.axhspan(-100, 0, color="#dde8f2", alpha=0.5, zorder=0)     # mixing band
    for ds, g in df.groupby("dataset"):
        is_flu = ds.startswith("ili")
        ax.scatter(g["delta"], g["phase_gain_pct"], s=46,
                   color=DATASET_COLOR.get(ds, "#777"),
                   marker="o" if is_flu else "s",
                   edgecolors="#1a1a1a", linewidths=0.6, zorder=4,
                   label=ds.replace("ili_", "").replace("_", " "))
    ax.axhline(0, color="#444", lw=0.8, ls="-")
    ax.axvline(0, color="#444", lw=0.9, ls="--")
    # annotate the failure: influenza helps but sits at Delta<=0
    ax.annotate("influenza:\nphase helps,\nyet $\\Delta\\!\\leq\\!0$",
                xy=(-0.03, 14), xytext=(-0.28, 15.5), fontsize=7.2, color="#7a1f22",
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color="#7a1f22", lw=0.8))
    ax.annotate("traffic:\nmixing wins", xy=(-0.55, -3.3), xytext=(-0.57, 7),
                fontsize=7.2, color="#22496e", ha="left",
                arrowprops=dict(arrowstyle="->", color="#22496e", lw=0.8))
    ax.set_xlabel(r"a-priori delay advantage $\Delta(H)$")
    ax.set_ylabel(r"phase gain over mixing (%)")
    ax.set_title(r"No cheap predictor: $\mathrm{sign}\,\Delta(H)$ misses the split")
    ax.legend(fontsize=6.6, ncol=2, loc="lower right", handletextpad=0.3)
    ax.set_ylim(-7, 21)
    fig.tight_layout()
    _save(fig, "fig_envelope")


def fig_mechanism():
    rows = []
    for p in sorted(TAB.glob("delay_validation_metrla_CSLL2-static_H*.json")):
        d = json.load(open(p))
        b0 = next((b for b in d["bands"] if b["bins"][0] == 0), d["bands"][0])
        rows.append((d["H"], b0.get("mean_absD_samples", np.nan)))
    if len(rows) < 2:
        print("skip mechanism (need >=2 horizons)"); return
    rows.sort()
    H = np.array([r[0] for r in rows], float)
    D = np.array([r[1] for r in rows], float)
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.8))
    ax.plot(H, H, ls=":", color="#999", lw=1.2, label=r"$D=H$ (pure lead)", zorder=2)
    ax.plot(H, D, "-", color=PALETTE["csll"], lw=2.0, zorder=4)
    ax.scatter(H, D, s=42, color=PALETTE["csll"], edgecolors="#1a1a1a",
               linewidths=0.6, zorder=5, label="learned $|D|$ (band 0)")
    ax.set_xlabel(r"forecast horizon $H$ (samples)")
    ax.set_ylabel(r"mean $|$learned delay$|$ (samples)")
    ax.set_title("Learned delay tracks the horizon")
    ax.legend(fontsize=7.4, loc="upper left")
    fig.tight_layout()
    _save(fig, "fig_mechanism")


def fig_pareto():
    st = TAB / "v2_st_mse.csv"
    eff = TAB / "v2_efficiency.csv"
    if not (st.exists() and eff.exists()):
        print("skip pareto"); return
    piv = pd.read_csv(st, index_col=[0, 1]).dropna(axis=1, how="all")
    ranks = piv.rank(axis=1).mean(axis=0)
    E = pd.read_csv(eff).set_index("name")
    pts = []
    for name in ranks.index:
        base = ("CSLL2" if name.startswith("CSLL2-") else name)
        pr = E.params.get("CSLL2" if name == "CSLL2H" else base, np.nan)
        if pr != pr or pr <= 0:
            continue
        pts.append((name, pr, ranks[name]))
    if not pts:
        print("skip pareto (no points)"); return
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.86))
    # Pareto frontier (lower rank + fewer params is better): staircase over non-dominated pts
    P = sorted(pts, key=lambda t: t[1])
    front, best = [], np.inf
    for nm, pr, rk in P:
        if rk < best:
            front.append((pr, rk)); best = rk
    if len(front) > 1:
        fx, fy = zip(*front)
        ax.step(fx, fy, where="post", color="#bbb", lw=1.0, ls="--", zorder=1)
    ax.set_xscale("log")
    prs = [p for _, p, _ in pts]
    xlo, xhi = min(prs), max(prs)
    ax.set_xlim(xlo / 2.2, xhi * 3.2)                 # headroom so labels never hug the edge
    xmid = (np.log10(xlo) + np.log10(xhi)) / 2
    for name, pr, rk in pts:
        s = style(name)
        ours = name.startswith("CSLL2")
        ax.scatter(pr, rk, s=(120 if name == "CSLL2H" else 72 if ours else 46),
                   marker=s["m"], color=s["c"], edgecolors="#1a1a1a",
                   linewidths=0.7, zorder=s["z"] + 2)
        # points on the right half get left-anchored labels so text stays inside the axes
        right = np.log10(pr) > xmid
        ha = "right" if right else "left"
        dx = -7 if right else 6
        ax.annotate(s["label"].split(" (")[0], (pr, rk), fontsize=6.6, ha=ha, va="center",
                    xytext=(dx, 6), textcoords="offset points", color="#222", zorder=10)
    ax.set_xlabel("parameters (log scale)")
    ax.set_ylabel("mean rank on traffic (lower better)")
    ax.set_title("Accuracy vs. cost on traffic")
    ax.invert_yaxis()
    fig.tight_layout()
    _save(fig, "fig_pareto_v2")


def fig_mseh():
    """MSE vs horizon on the phase-essential (influenza) and mixing (traffic) regimes."""
    panels = [("ili_us_state", "US-state influenza", [2, 4, 8]),
              ("metr_la", "METR-LA traffic", [12, 24, 48, 96])]
    models = ["NLinear", "DLinear", "iTransformer", "CSLL2-phaseoff", "CSLL2", "CSLL2H"]
    frames = {}
    for tag in ["v2_epi_mse", "v2_st_mse"]:
        f = TAB / f"{tag}.csv"
        if f.exists():
            frames[tag] = pd.read_csv(f, index_col=[0, 1])
    if not frames:
        print("skip mseh"); return
    allpiv = pd.concat(frames.values())
    fig, axs = plt.subplots(1, 2, figsize=(DBL_W * 0.78, COL_W * 0.86))
    for ax, (ds, title, Hs) in zip(axs, panels):
        for m in models:
            if m not in allpiv.columns:
                continue
            ys = [allpiv.loc[(ds, h), m] if (ds, h) in allpiv.index else np.nan for h in Hs]
            if all(np.isnan(y) for y in ys):
                continue
            st = style(m)
            ax.plot(Hs, ys, marker=st["m"], color=st["c"], ls=st["ls"], lw=st["lw"],
                    ms=st["ms"] - 1, label=st["label"].split(" (")[0], zorder=st["z"])
        ax.set_title(title); ax.set_xlabel("horizon $H$"); ax.set_ylabel("test MSE")
        ax.set_xticks(Hs)
    axs[0].legend(fontsize=6.4, loc="upper left", ncol=1)
    fig.tight_layout()
    _save(fig, "fig_mseh")


if __name__ == "__main__":
    fig_envelope()
    fig_mechanism()
    fig_pareto()
    fig_mseh()
    print("PLOTS_V2_DONE")
