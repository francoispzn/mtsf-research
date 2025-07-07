#!/usr/bin/env python3
"""Geographic validation of CSLL2's learned delays on regional ILI.

Ground truth: the Cola-GNN spatial adjacency matrices (data/raw/epi/*-adj.txt) — regions
that share a border. Epidemiological prior: influenza spreads between adjacent regions with
a lag, so (i) neighbouring regions should couple more strongly (|A| higher on graph edges),
and (ii) the learned lead-lag delays should be organised by the spread geography (a region's
net delay ~ how early/late its epidemic peaks relative to neighbours).

For each trained CSLL2 checkpoint (seed 0) this extracts per-band coupling |A| and the
aggregate delay matrix D, and tests:
  1. edge enrichment: mean |A| on adjacency edges vs non-edges (Mann-Whitney direction +
     ratio) -- does the model concentrate coupling on true neighbours?
  2. delay antisymmetry vs empirical lead-lag: r(model D_ij, empirical xcorr lag_ij) on the
     coupled (top-|A|) neighbour pairs.
  3. net-delay vs peak-timing: each region's row-mean delay vs its empirical mean epidemic
     peak week (spatial diffusion ordering).

Writes results/tables/delay_validation_<dataset>.json.

Run (after campaign_v2.py produces epi CSLL2 checkpoints):
  MTSF_DEVICE=cpu .venv/bin/python3 code/scripts/validate_delays_epi.py --dataset ili_us_state --H 2
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from csll.config import model_config, seq_len_for            # noqa: E402
from csll.data import load_matrix, split_indices             # noqa: E402
from csll.models import build_model                          # noqa: E402

RAW = ROOT / "results" / "raw"
TAB = ROOT / "results" / "tables"
ADJ = {"ili_japan": "japan-adj.txt", "ili_us_hhs": "region-adj.txt",
       "ili_us_state": "state-adj.txt"}


def opt(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def empirical_lags(R, maxlag=12):
    N = R.shape[1]
    nfft = 1
    while nfft < 2 * len(R):
        nfft *= 2
    F = np.fft.rfft(R, nfft, axis=0)
    lag = np.zeros((N, N)); pk = np.zeros((N, N))
    for a in range(N):
        for b in range(N):
            if a == b:
                continue
            c = np.fft.irfft(F[:, b] * np.conj(F[:, a]), nfft)
            c = np.concatenate([c[-maxlag:], c[:maxlag + 1]])
            lag[b, a] = np.abs(c).argmax() - maxlag
            pk[b, a] = np.abs(c).max() / len(R)
    return lag, pk


def deseasonalise(Xtr, period=52):
    Z = np.log1p(np.clip(Xtr, 0, None))
    wk = np.arange(len(Z)) % period
    prof = np.stack([Z[wk == w].mean(0) for w in range(period)])
    R = Z - prof[wk]
    return (R - R.mean(0)) / (R.std(0) + 1e-8)


if __name__ == "__main__":
    ds = opt("--dataset", "ili_us_state")
    H = int(opt("--H", "2"))
    L = seq_len_for(ds)
    variant = opt("--variant", "CSLL2")
    ckpt = RAW / f"{ds}__{variant}__L{L}__H{H}__s0.pt"
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")

    A = np.loadtxt(ROOT / "data" / "raw" / "epi" / ADJ[ds], delimiter=",")
    N = A.shape[0]
    edge = (A > 0) & ~np.eye(N, dtype=bool)
    nonedge = (A == 0) & ~np.eye(N, dtype=bool)

    X = load_matrix(ds)
    tr0, tr1 = split_indices(ds, len(X), L)["train"]
    R = deseasonalise(X[tr0:tr1])
    emp_lag, emp_pk = empirical_lags(R)
    peak_week = np.array([np.bincount(np.argmax(
        np.log1p(np.clip(X[tr0:tr1, i], 0, None)).reshape(-1, 52)[
            :len(X[tr0:tr1]) // 52 * 52 // 52] if False else
        np.log1p(np.clip(X[tr0:tr1, i], 0, None))[:len(R) // 52 * 52].reshape(-1, 52), axis=1),
        minlength=52).argmax() for i in range(N)])

    cfg = model_config("CSLL2", ds, L, H, N)
    if variant.endswith("-static"):
        cfg["dynamic"] = False
    net = build_model("CSLL2", cfg)
    net.load_state_dict(torch.load(ckpt, map_location="cpu"))
    net.eval()

    # DELAY read-out: use the input-conditioned (dynamic) delays actually applied at test time,
    # not the static base (which the controller offsets to the epidemic scale). Average the
    # per-band delay matrix over test windows.
    from csll.data import build_dataset
    bundle = build_dataset(ds, L, H)
    xs = torch.stack([bundle.test[i][0] for i in range(min(400, len(bundle.test)))])
    absA = np.zeros((N, N))
    Dbands = []
    with torch.no_grad():
        for b in range(net.n_bands):
            Dbands.append(net.delays_for_input(xs, band=b).numpy())
            Ab = (net.U[b] @ net.V[b].T).detach().numpy() if net.low_rank else net.A[b].detach().numpy()
            absA += np.abs(Ab)
    absA /= net.n_bands
    # energy-weight bands by |A| so the reported D reflects where coupling actually lives
    D = np.mean(Dbands, axis=0)

    report = dict(dataset=ds, H=H, variant=variant, N=int(N),
                  alpha=float(net.alpha.detach()))
    # 1. edge enrichment of coupling
    report["absA_edge_mean"] = float(absA[edge].mean())
    report["absA_nonedge_mean"] = float(absA[nonedge].mean())
    report["absA_edge_ratio"] = float(absA[edge].mean() / (absA[nonedge].mean() + 1e-12))
    # 2. model delay vs empirical lag on coupled neighbour pairs
    strong = emp_pk > 0.3
    coupled = absA >= np.quantile(absA[~np.eye(N, dtype=bool)], 0.8)
    sel = edge & strong & coupled
    if sel.sum() > 20:
        report["r_D_vs_emplag_edges"] = float(np.corrcoef(D[sel], emp_lag[sel])[0, 1])
        report["n_edge_coupled_pairs"] = int(sel.sum())
    # 3. net delay (row-mean over coupled) vs empirical peak week
    net_delay = np.array([D[i, coupled[i]].mean() if coupled[i].sum() else 0.0 for i in range(N)])
    if np.std(net_delay) > 1e-6:
        report["r_netdelay_vs_peakweek"] = float(np.corrcoef(net_delay, peak_week)[0, 1])
    report["mean_absD_edges"] = float(np.abs(D[edge]).mean())
    report["mean_emplag_edges"] = float(np.abs(emp_lag[edge & strong]).mean())

    TAB.mkdir(parents=True, exist_ok=True)
    out = TAB / f"delay_validation_{ds}_{variant}_H{H}.json"
    json.dump(report, open(out, "w"), indent=2)
    print(json.dumps(report, indent=2))
    print(f"EPI_DELAY_VALIDATION_DONE -> {out.name}")
