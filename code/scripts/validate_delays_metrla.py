#!/usr/bin/env python3
"""Physical validation of CSLL2's learned delays on METR-LA.

Ground truth: the DCRNN sensor graph (road-network distances between sensor pairs) and
kinematic traffic-flow theory (congestion back-propagates upstream at ~15-20 km/h, i.e.
~3-4 min/km; free-flow disturbances travel with traffic much faster, mostly sub-sample
at 5-min resolution).

For the trained CSLL2 checkpoint this script extracts, per frequency band:
  * effective coupling magnitudes |A_eff| (low-rank U V^T),
  * static/base delay matrix D_ij = p_i - q_j (in samples; x5 = minutes),
  * input-conditioned dynamic delays for rush-hour vs off-peak test windows,
and evaluates, over the directed sensor pairs (<5 km) of distances_la_2012.csv:
  1. r(model delay, empirical xcorr lag) on the pairs the model actually couples
     (top-|A| quantile) -- does the model match the estimation ceiling?
  2. regression of model delay vs road distance on downstream-leads pairs ->
     implied back-propagation speed, to compare with the empirical ~16 km/h and
     the 15-20 km/h kinematic-wave range.
  3. rush vs off-peak delay contrast (does the dynamic variant sharpen delays
     exactly when congestion exists?).

Writes results/tables/delay_validation_metrla.json.

Run after pilot_metrla.py:
  MTSF_DEVICE=cpu MTSF_TORCH_THREADS=4 .venv/bin/python3 code/scripts/validate_delays_metrla.py
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
from csll.config import model_config                      # noqa: E402
from csll.data import load_matrix, split_indices          # noqa: E402
from csll.models import build_model                       # noqa: E402

RAW = ROOT / "results" / "raw"
TAB = ROOT / "results" / "tables"
L, SEED = 96, 0
H = int(sys.argv[sys.argv.index("--H") + 1]) if "--H" in sys.argv else 12
VARIANT = sys.argv[sys.argv.index("--variant") + 1] if "--variant" in sys.argv else "CSLL2"
CKPT = RAW / f"metr_la__{VARIANT}__L{L}__H{H}__s{SEED}.pt"


def load_sensor_frame():
    df = pd.read_csv(ROOT / "data" / "raw" / "metr_la.csv", parse_dates=["date"])
    ids = [c for c in df.columns if c != "date"]
    return df, ids


def load_pairs(ids, max_m=5000.0):
    d = pd.read_csv(ROOT / "data" / "raw" / "st" / "distances_la_2012.csv")
    d.columns = ["frm", "to", "cost"]
    d["frm"] = d.frm.astype(float).astype(int).astype(str)
    d["to"] = d.to.astype(float).astype(int).astype(str)
    s = set(ids)
    d = d[d.frm.isin(s) & d.to.isin(s) & (d.cost > 0) & (d.cost < max_m)]
    col = {sid: i for i, sid in enumerate(ids)}
    return np.array([[col[a], col[b], c] for a, b, c in zip(d.frm, d.to, d.cost)])


def empirical_lags(X, dates, pairs, maxlag=12):
    """|xcorr| argmax lag per pair on deseasonalised residuals (rush hours only)."""
    Xm = X.copy(); Xm[Xm == 0] = np.nan
    day, N = 288, X.shape[1]
    tod = np.arange(len(Xm)) % day
    prof = np.zeros((day, N))
    for t in range(day):
        prof[t] = np.nanmean(Xm[tod == t], axis=0)
    R = np.where(np.isnan(Xm - prof[tod]), 0.0, Xm - prof[tod])
    R = (R - R.mean(0)) / (R.std(0) + 1e-8)
    hh, wd = dates.dt.hour.to_numpy(), dates.dt.dayofweek.to_numpy()
    rush = (wd < 5) & (((hh >= 6) & (hh < 10)) | ((hh >= 15) & (hh < 20)))
    R[~rush] = 0.0
    nfft = 1
    while nfft < 2 * len(R):
        nfft *= 2
    F = np.fft.rfft(R, nfft, axis=0)
    lags, peaks = [], []
    for a, b, _ in pairs:
        c = np.fft.irfft(F[:, int(b)] * np.conj(F[:, int(a)]), nfft)
        c = np.concatenate([c[-maxlag:], c[:maxlag + 1]])
        lags.append(int(np.abs(c).argmax()) - maxlag)
        peaks.append(float(np.abs(c).max() / max(1, rush.sum())))
    return np.array(lags, float), np.array(peaks)


def band_delays_and_coupling(net):
    out = []
    for b in range(net.n_bands):
        p = (net.pq_bound * torch.tanh(net.p_base[b])).detach().numpy()
        q = (net.pq_bound * torch.tanh(net.q_base[b])).detach().numpy()
        D = p[:, None] - q[None, :]
        if net.low_rank:
            A = (net.U[b] @ net.V[b].T).detach().numpy()
        else:
            A = net.A[b].detach().numpy()
        lo, hi = net.bands[b]
        out.append(dict(D=D, absA=np.abs(A), bins=(lo, hi)))
    return out


def dynamic_delays(net, X_std, test_rows, dates, band, cond_mask, n_max=512):
    """Mean input-conditioned delay matrix over test windows whose END satisfies cond."""
    a, bnd = test_rows
    ends = np.arange(a + L - 1, bnd - H)
    ok = np.where(cond_mask[ends])[0]
    if len(ok) == 0:
        return None
    sel = ok[np.linspace(0, len(ok) - 1, min(n_max, len(ok))).astype(int)]
    ws = np.stack([X_std[a + k:a + k + L] for k in sel])
    x = torch.tensor(ws, dtype=torch.float32)
    with torch.no_grad():
        return net.delays_for_input(x, band=band).numpy()


def pair_stats(D, absA, pairs, emp_lag, emp_peak, a_quant=0.8):
    """Correlations/regressions of model delays vs empirical lags and distance."""
    ai = pairs[:, 0].astype(int); bi = pairs[:, 1].astype(int)
    d_km = pairs[:, 2] / 1000.0
    # model delay for coupling a->b is D[target=b, source=a]
    md = D[bi, ai]
    w = absA[bi, ai]
    keep = w >= np.quantile(w, a_quant)                    # pairs the model actually couples
    strong = emp_peak > 0.25
    res = dict(n_pairs=int(len(pairs)), n_used=int(keep.sum()))
    if keep.sum() > 30:
        res["r_model_vs_emp_lag_usedA"] = float(np.corrcoef(md[keep & strong], emp_lag[keep & strong])[0, 1]) \
            if (keep & strong).sum() > 30 else None
        neg = keep & (md < -0.15)                          # downstream-leads pairs per the model
        res["n_shockwave_pairs"] = int(neg.sum())
        if neg.sum() > 30:
            sl = np.polyfit(d_km[neg], 5 * md[neg], 1)[0]  # min per km
            res["shockwave_slope_min_per_km"] = float(sl)
            res["implied_backprop_speed_kmh"] = float(60 / abs(sl)) if abs(sl) > 1e-6 else None
    return res


if __name__ == "__main__":
    if not CKPT.exists():
        raise SystemExit(f"checkpoint not found: {CKPT} (run pilot_metrla.py first)")
    df, ids = load_sensor_frame()
    X = df[ids].to_numpy(np.float64)
    dates = df["date"]
    pairs = load_pairs(ids)
    print(f"pairs <5km among sensors: {len(pairs)}")

    idx = split_indices("metr_la", len(X), L)
    tr0, tr1 = idx["train"]
    mu, sd = X[tr0:tr1].mean(0), X[tr0:tr1].std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    X_std = ((X - mu) / sd).astype(np.float32)

    emp_lag, emp_peak = empirical_lags(X[tr0:tr1], dates[tr0:tr1], pairs)
    print(f"empirical rush-hour lags computed; strong pairs: {(emp_peak>0.25).sum()}")

    cfg = model_config("CSLL2", "metr_la", L, H, len(ids))
    if VARIANT.endswith("-static"):
        cfg["dynamic"] = False
    net = build_model("CSLL2", cfg)
    net.load_state_dict(torch.load(CKPT, map_location="cpu"))
    net.eval()
    alpha = float(net.alpha.detach())
    bands = band_delays_and_coupling(net)

    report = dict(alpha=alpha, H=H, bands=[])
    hh, wd = dates.dt.hour.to_numpy(), dates.dt.dayofweek.to_numpy()
    rush_mask = (wd < 5) & (((hh >= 6) & (hh < 10)) | ((hh >= 15) & (hh < 20)))
    off_mask = (wd < 5) & (hh >= 10) & (hh < 15)

    for b, info in enumerate(bands):
        stats = pair_stats(info["D"], info["absA"], pairs, emp_lag, emp_peak)
        stats["bins"] = list(info["bins"])
        stats["mean_absD_samples"] = float(np.abs(info["D"]).mean())
        # dynamic contrast on the aggregate delay (rush vs off-peak)
        Dr = dynamic_delays(net, X_std, idx["test"], dates, b, rush_mask)
        Do = dynamic_delays(net, X_std, idx["test"], dates, b, off_mask)
        if Dr is not None and Do is not None:
            ai, bi = pairs[:, 0].astype(int), pairs[:, 1].astype(int)
            w = info["absA"][bi, ai]
            keep = w >= np.quantile(w, 0.8)
            stats["dyn_mean_absD_rush"] = float(np.abs(Dr[bi, ai][keep]).mean())
            stats["dyn_mean_absD_offpeak"] = float(np.abs(Do[bi, ai][keep]).mean())
        report["bands"].append(stats)
        print(f"band {b} bins {info['bins']}: {stats}")

    TAB.mkdir(parents=True, exist_ok=True)
    out = TAB / f"delay_validation_metrla_{VARIANT}_H{H}.json"
    json.dump(report, open(out, "w"), indent=2)
    print(f"DELAY_VALIDATION_DONE -> {out.name}")
