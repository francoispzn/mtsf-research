#!/usr/bin/env python3
"""Kill-test for the v2 estimator: does fixing the estimator fix delay recovery?

The v1 estimator failed delay recovery for identified reasons: (i) gradient descent on the
delay parameters has an oscillatory loss surface (classic time-delay-estimation pathology)
and converges to near-zero local minima; (ii) the additive parameterisation allowed |D| up
to 2*tau_max, past the wrap-free identifiable range, and the planted delays (up to 72)
exceeded tau_max=48; (iii) the circular phase-ramp mismatches the linear shifts in the data;
(iv) the zero-initialised gate passes no gradient to the branch until alpha moves.

v2 fixes: strict bound (|D| <= tau_max), zero-padded linear-shift basis (pad2x), warm gate
(alpha0 = 0.05), and cross-correlation initialisation of the delays.

Grid (regime A, leader-follower, planted tau in [24,44] -- inside the design range):
  v1-static            legacy flags, no init          (reproduces the failure)
  v2-static-noinit     bound+pad+gate fixes only      (isolates the optimisation problem)
  v2-static            all fixes incl. xcorr init
  v2-dynamic           full v2
  v2-phase-off         real-valued mixing under v2    (value-envelope control)
  backbone-only        CI backbone                    (value-envelope control)

Regime A-hard: planted tau in [24,72] (v1 paper's setting; 60,72 exceed tau_max=48) run with
v1-static and v2-static -- documents behaviour beyond the representable range.

Regime B (self-predictable, signed gains, circular shifts -- unchanged from v1): v2 variants,
value-envelope check that the fixes do not manufacture spurious gains where none exist.

Success criterion (pre-registered): v2-static recovery slope in [0.8, 1.2] and r > 0.9 on the
follower->leader column, with v1-static reproducing the published failure (slope ~ 0.12).

Writes results/tables/controlled_v2.json.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from csll import set_seed                                    # noqa: E402
from csll.models.csll import CSLL                            # noqa: E402
from csll.synthetic import make_leadfollower, make_synthetic  # noqa: E402
from csll.train import get_device                            # noqa: E402

TAB = ROOT / "results" / "tables"

V1 = dict(strict_bound=False, pad2x=False, gate_init=0.0)
# direct=True: the branch forecasts by synthesising the phase-shifted spectrum at future
# positions (no L->H head). For use_phase=False the model falls back to the head path
# automatically (a real mixer cannot reach the future by construction).
V2 = dict(strict_bound=True, pad2x=True, gate_init=0.05, direct=True)

VARIANTS = {
    "v1-static":        dict(flags=V1, use_phase=True,  dynamic=False, init=False),
    "v2-static-noinit": dict(flags=V2, use_phase=True,  dynamic=False, init=False),
    "v2-static":        dict(flags=V2, use_phase=True,  dynamic=False, init=True),
    "v2-dynamic":       dict(flags=V2, use_phase=True,  dynamic=True,  init=True),
    "v2-phase-off":     dict(flags=V2, use_phase=False, dynamic=False, init=False),
    "backbone-only":    dict(flags=dict(gate_init=0.0), use_phase=True, dynamic=False,
                             init=False, freeze_gate=True),
}


def _windows(a, L, H):
    xs = [a[s:s + L] for s in range(len(a) - L - H + 1)]
    ys = [a[s + L:s + L + H] for s in range(len(a) - L - H + 1)]
    return torch.tensor(np.array(xs)), torch.tensor(np.array(ys))


def _split(data, L, H, ptr=0.7, pva=0.1):
    n = len(data)
    i1, i2 = int(n * ptr), int(n * (ptr + pva))
    mu, sd = data[:i1].mean(0), data[:i1].std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    d = (data - mu) / sd
    return (d[:i1],
            _windows(d[:i1], L, H), _windows(d[i1 - L:i2], L, H), _windows(d[i2 - L:], L, H))


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


def recovery_metrics(net, tau_true):
    """Follower->leader column of the aggregate learned delay matrix vs planted lags."""
    D = net.learned_delays(band=None).cpu().numpy()
    t = np.concatenate([[0], tau_true]).astype(float)
    col = D[1:, 0]
    n = len(tau_true)
    r = float(np.corrcoef(tau_true, col)[0, 1])
    slope = float(np.polyfit(tau_true, col, 1)[0])
    N = len(t)
    D_true = t[:, None] - t[None, :]
    off = ~np.eye(N, dtype=bool)
    r_full = float(np.corrcoef(D_true[off], D[off])[0, 1])
    return dict(r_leader_col=r, slope=slope, n=n, r_full_matrix=r_full,
                true_tau=[int(x) for x in tau_true],
                learned_leader_col=[float(x) for x in col])


def run_regime(name, data, L, H, device, variants, tau_true=None):
    train_series, tr, va, te = _split(data, L, H)
    N = data.shape[1]
    out = {}
    for tag, spec in variants.items():
        set_seed(0)
        kw = dict(spec["flags"])
        if spec.get("freeze_gate"):
            kw["freeze_gate"] = True
        net = CSLL(L, H, N, n_bands=4, use_revin=True, tau_max=L / 2.0,
                   use_phase=spec["use_phase"], dynamic=spec["dynamic"], **kw)
        rec_init = None
        if spec["init"]:
            net.init_delays_from_xcorr(train_series)
            if tau_true is not None:
                rec_init = recovery_metrics(net, tau_true)
        mse = _train_eval(net, tr, va, te, device)
        entry = dict(test_mse=mse,
                     final_alpha=float(net.alpha.detach().cpu()) if hasattr(net, "alpha") else None)
        if tau_true is not None and spec["use_phase"] and not spec.get("freeze_gate"):
            entry["recovery"] = recovery_metrics(net, tau_true)
            if rec_init is not None:
                entry["recovery_at_init"] = rec_init
        out[tag] = entry
        rec = entry.get("recovery")
        extra = (f"  slope={rec['slope']:.3f} r={rec['r_leader_col']:.3f}" if rec else "")
        print(f"  [{name}] {tag:18s} test MSE={mse:.4f} alpha={entry['final_alpha']}{extra}",
              flush=True)
    return out


if __name__ == "__main__":
    device = get_device()
    res = {}

    print("== Regime A: leader-follower, tau in [24,44] (inside design range), L=96 H=24 ==", flush=True)
    dataA, tauA, _ = make_leadfollower(T=12000, n_follow=6, seed=0, tau_lo=24, tau_hi=44)
    res["A_in_range"] = run_regime("A", dataA, 96, 24, device, VARIANTS, tau_true=tauA)

    print("== Regime A-hard: tau in [24,72] (60,72 exceed tau_max=48; v1 paper setting) ==", flush=True)
    dataH, tauH, _ = make_leadfollower(T=12000, n_follow=6, seed=0, tau_lo=24, tau_hi=72)
    hard = {k: VARIANTS[k] for k in ("v1-static", "v2-static")}
    res["A_beyond_range"] = run_regime("A-hard", dataH, 96, 24, device, hard, tau_true=tauH)

    print("== Regime B: self-predictable (signed gains, circular shifts), L=96 H=96 ==", flush=True)
    dataB, tauB = make_synthetic(T=8000, n_vars=8, seed=0)
    vb = {k: VARIANTS[k] for k in ("v2-dynamic", "v2-static", "v2-phase-off", "backbone-only")}
    # make_synthetic returns per-channel taus with taus[0]=0 anchored; recovery_metrics
    # prepends the reference-channel 0 itself, so pass the non-anchor channels only.
    res["B_self_predictable"] = run_regime("B", dataB, 96, 96, device, vb, tau_true=tauB[1:])

    TAB.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(TAB / "controlled_v2.json", "w"), indent=2)
    print("CONTROLLED_V2_DONE", flush=True)
