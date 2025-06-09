#!/usr/bin/env python3
"""Controlled "value-envelope" experiments: WHEN does CSLL's phase/lead-lag mechanism help?

Two synthetic regimes, each run with four CSLL variants under one identical trainer:
  * full          -- dynamic phase (input-adaptive delays)
  * static        -- static phase/delay (LTI)
  * phase-off     -- real-valued same-time mixing (D := 0)     [isolates the PHASE]
  * backbone-only -- channel-independent backbone, branch off  [isolates the BRANCH]

Regime A -- LEADER-FOLLOWER (phase is *essential*): a white-noise leader; each follower is the
  leader delayed by tau_i. Only a model that reads the correct per-channel delay from another
  series can forecast -> phase-off and CI must fail; phase must win. We also check delay recovery.
Regime B -- SELF-PREDICTABLE shared source (planted delays, but each series is autoregressive and
  predictable from its own past) -> cross-series delay is redundant; the mechanism should be ~flat.

The contrast is the paper's core message: phase-aware cross-series modelling helps exactly when
cross-series information is essential AND delayed; standard MTSF series are self-predictable, so it
does not -- which is why channel-independent models are so strong.

Writes results/tables/controlled.json and results/figures/controlled.pdf/.png.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                   # noqa: E402
import torch                         # noqa: E402
from torch.utils.data import DataLoader, TensorDataset   # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from csll import set_seed                                   # noqa: E402
from csll.models.csll import CSLL                           # noqa: E402
from csll.synthetic import make_leadfollower, make_synthetic  # noqa: E402
from csll.train import get_device                           # noqa: E402
from csll.plotstyle import apply as _apply_style, PALETTE   # noqa: E402

_apply_style()

FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"

VARIANTS = {
    "full (dynamic)":   dict(use_phase=True, dynamic=True),
    "static delay":     dict(use_phase=True, dynamic=False),
    "phase-off (real)": dict(use_phase=False, dynamic=False),
    "backbone-only":    dict(freeze_gate=True),
}


def _windows(a, L, H):
    xs = [a[s:s + L] for s in range(len(a) - L - H + 1)]
    ys = [a[s + L:s + L + H] for s in range(len(a) - L - H + 1)]
    return torch.tensor(np.array(xs)), torch.tensor(np.array(ys))


def _standardise_split(data, L, H, ptr=0.7, pva=0.1):
    n = len(data)
    i1, i2 = int(n * ptr), int(n * (ptr + pva))
    mu, sd = data[:i1].mean(0), data[:i1].std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    d = (data - mu) / sd
    return (_windows(d[:i1], L, H), _windows(d[i1 - L:i2], L, H), _windows(d[i2 - L:], L, H))


def _train_eval(net, tr, va, te, device, epochs=30, patience=5):
    net = net.to(device)
    opt = torch.optim.Adam(net.optim_groups(0.0), lr=1e-3)
    crit = torch.nn.MSELoss()

    def ev(split):
        net.eval(); tot = n = 0
        dl = DataLoader(TensorDataset(*split), batch_size=128)
        with torch.no_grad():
            for x, y in dl:
                x, y = x.to(device), y.to(device)
                tot += float(((net(x) - y) ** 2).mean()) * len(x); n += len(x)
        return tot / n

    trdl = DataLoader(TensorDataset(*tr), batch_size=64, shuffle=True)
    best, bad = 1e9, 0
    best_state = copy.deepcopy(net.state_dict())
    for _ in range(epochs):
        net.train()
        for x, y in trdl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); crit(net(x), y).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
        v = ev(va)
        if v < best - 1e-7:
            best, best_state, bad = v, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    return ev(te)


def run_regime(name, data, L, H, device):
    tr, va, te = _standardise_split(data, L, H)
    N = data.shape[1]
    out = {}
    for tag, cfg in VARIANTS.items():
        set_seed(0)
        net = CSLL(L, H, N, n_bands=4, use_revin=True, tau_max=L / 2.0, **cfg)
        out[tag] = _train_eval(net, tr, va, te, device)
        print(f"  [{name}] {tag:20s} test MSE = {out[tag]:.4f}")
    return out


def delay_recovery(tauA, data, L, H, device):
    """Train the static, phase-on model in the phase-essential regime and measure how well its
    learned delays recover the planted follower->leader lags."""
    tr, va, te = _standardise_split(data, L, H)
    N = data.shape[1]
    set_seed(0)
    net = CSLL(L, H, N, n_bands=4, use_revin=True, tau_max=L / 2.0, use_phase=True, dynamic=False)
    _train_eval(net, tr, va, te, device)
    D = net.learned_delays(band=0).cpu().numpy()
    A = np.abs(net.A[0].detach().cpu().numpy())
    t = np.concatenate([[0], tauA]).astype(float)
    D_true = t[:, None] - t[None, :]
    off = ~np.eye(N, dtype=bool)

    def wcorr(x, y, w):
        w = w / w.sum()
        mx, my = (w * x).sum(), (w * y).sum()
        cov = (w * (x - mx) * (y - my)).sum()
        return float(cov / np.sqrt((w * (x - mx) ** 2).sum() * (w * (y - my) ** 2).sum() + 1e-12))

    r_leadcol = float(np.corrcoef(t[1:], D[1:, 0])[0, 1])    # each follower's delay to the leader
    r_full = float(np.corrcoef(D_true[off], D[off])[0, 1])
    r_w = wcorr(D_true[off], D[off], A[off])
    print(f"  [delay recovery] follower->leader r={r_leadcol:.3f}  full-matrix r={r_full:.3f}  "
          f"|A|-weighted r={r_w:.3f}")
    return dict(r_leader_col=r_leadcol, r_full_matrix=r_full, r_absA_weighted=r_w,
                true_tau=[int(x) for x in tauA], learned_leader_col=[float(x) for x in D[1:, 0]])


def make_figure(res):
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.1))
    # panel 1: MSE normalised to each regime's channel-independent baseline (=1.0), so the two
    # different tasks are comparable in SHAPE. Lower = better; below 1.0 = beats CI backbone.
    tags = list(VARIANTS.keys())
    A = [res["phase_essential"][t] / res["phase_essential"]["backbone-only"] for t in tags]
    B = [res["self_predictable"][t] / res["self_predictable"]["backbone-only"] for t in tags]
    x = np.arange(len(tags)); w = 0.38
    ax[0].bar(x - w / 2, A, w, label="phase-essential\n(leader--follower)",
              color=PALETTE["csll"], edgecolor="#222", linewidth=0.5)
    ax[0].bar(x + w / 2, B, w, label="self-predictable\n(shared source)",
              color=PALETTE["blue"], edgecolor="#222", linewidth=0.5)
    ax[0].axhline(1.0, color="#222", lw=0.8, ls=":")
    ax[0].set_xticks(x); ax[0].set_xticklabels(tags, rotation=25, ha="right", fontsize=8)
    ax[0].set_ylabel("test MSE / CI-backbone MSE")
    ax[0].legend(fontsize=7.5, loc="lower left")
    ax[0].set_title("Phase helps only when cross-series\ninformation is essential and delayed")
    ax[0].grid(alpha=0.5, axis="y")
    # panel 2: delay recovery scatter (follower -> leader) with fit line
    dr = res["delay_recovery"]
    xt, yt = np.array(dr["true_tau"], float), np.array(dr["learned_leader_col"], float)
    sl, ic = np.polyfit(xt, yt, 1)
    xs = np.linspace(xt.min() - 2, xt.max() + 2, 50)
    ax[1].plot(xs, sl * xs + ic, "--", color=PALETTE["grey"], lw=1.1, zorder=1)
    ax[1].scatter(xt, yt, s=70, c=PALETTE["csll"], edgecolors="#111", linewidths=0.7, zorder=3)
    ax[1].set_xlabel(r"true follower delay $\tau_i$")
    ax[1].set_ylabel(r"learned delay $D_{i,\,\mathrm{leader}}$")
    ax[1].set_title(f"Delay recovery in-regime ($r={dr['r_leader_col']:.2f}$)")
    fig.tight_layout()
    fig.savefig(FIG / "controlled.pdf")
    fig.savefig(FIG / "controlled.png")
    plt.close(fig)
    print("wrote controlled.pdf + controlled.png")


if __name__ == "__main__":
    if "--fig-only" in sys.argv and (TAB / "controlled.json").exists():
        make_figure(json.load(open(TAB / "controlled.json")))
        print("CONTROLLED_FIG_DONE"); sys.exit(0)
    device = get_device()
    print("== Regime A: leader-follower (phase essential), L=96 H=24 ==")
    dataA, tauA, _ = make_leadfollower(T=12000, n_follow=6, seed=0)
    resA = run_regime("phase-essential", dataA, 96, 24, device)

    print("== Regime B: self-predictable shared source, L=96 H=96 ==")
    dataB, _ = make_synthetic(T=8000, n_vars=8, seed=0)
    resB = run_regime("self-predictable", dataB, 96, 96, device)

    print("== Delay recovery (phase-essential regime) ==")
    dr = delay_recovery(tauA, dataA, 96, 24, device)

    res = {"phase_essential": resA, "self_predictable": resB, "delay_recovery": dr}
    TAB.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(TAB / "controlled.json", "w"), indent=2)
    make_figure(res)
    print("CONTROLLED_DONE")
