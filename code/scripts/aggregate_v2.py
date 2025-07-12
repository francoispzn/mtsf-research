#!/usr/bin/env python3
"""Aggregate the v2 campaign (CSLL2 + variants) into tidy tables.

Consumes whatever results/raw/*.json exist (safe to run mid-campaign). Produces, in
results/tables/:
  v2_st_mse.csv / .tex       spatio-temporal traffic (metr_la/pems_bay/pems04/pems08)
  v2_epi_mse.csv / .tex      regional ILI (ili_us_state/ili_japan/ili_us_hhs)
  v2_ltsf_mse.csv            LTSF suite (CSLL2 vs the existing v1 baselines)
  v2_ablation.csv            per-dataset full/static/phaseoff/backbone (the mechanism)
  v2_envelope.csv            THE money table: per dataset, tau (empirical lead),
                             horizon, phase-vs-mixing delta, and which regime it falls in
  v2_efficiency.csv          params + train time per model (Pareto inputs)
  v2_dm.csv                  Diebold-Mariano CSLL2 vs best baseline per ST/epi setting

Model/variant naming: base models keep their name; CSLL2 ablations are tagged
"CSLL2-<variant>" in the run stem (static/backbone/phaseoff).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from csll.stats import diebold_mariano                        # noqa: E402

RAW = ROOT / "results" / "raw"
TAB = ROOT / "results" / "tables"

ST = ["metr_la", "pems_bay", "pems04", "pems08"]
EPI = ["ili_us_state", "ili_japan", "ili_us_hhs"]
LTSF = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "exchange", "ili"]
BASELINES = ["SeasonalNaive", "VAR", "NLinear", "DLinear", "PatchTST", "iTransformer"]
# empirical mean inter-series lead (samples) from the physics pre-tests; horizon units match
# the dataset. Filled by measure_lags() below so the envelope table is self-contained.


def load():
    rows = []
    for p in sorted(RAW.glob("*.json")):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        model = d["model"]
        stem = p.stem
        variant = None
        # CSLL2 ablation variants live in the stem as CSLL2-<variant>
        for v in ("static", "backbone", "phaseoff"):
            if f"CSLL2-{v}__" in stem:
                variant = v
                model = "CSLL2"
        name = "CSLL2-" + variant if variant else model
        h = d.get("history", {})
        rows.append(dict(name=name, model=model, variant=(variant or "full" if model == "CSLL2" else ""),
                         dataset=d["dataset"], pred_len=d["pred_len"], seed=d["seed"],
                         mse=d["metrics"]["mse"], mae=d["metrics"]["mae"],
                         mase=d["metrics"].get("mase", np.nan),
                         params=h.get("params", 0), train_s=h.get("train_time_s", 0.0),
                         stem=stem))
    return pd.DataFrame(rows)


ST_HORIZONS = {12, 24, 48, 96}       # accuracy-table horizons (exclude the H{2,3,6} delay sweep)
EPI_HORIZONS = {2, 4, 8}
# declared per-dataset horizons: METR-LA full protocol, PEMS-BAY reduced confirmation {12,96}
ST_H_BY_DS = {"metr_la": {12, 24, 48, 96}, "pems_bay": {12, 96},
              "pems04": {12, 96}, "pems08": {12, 96}}


def _pivot(df, datasets, cols):
    sub = df[df.dataset.isin(datasets)]
    # keep only each dataset's declared horizons (drops the H{2,3,6} delay sweep and any stray
    # off-protocol rows left over from an earlier run)
    if set(datasets) & set(ST):
        keep = sub.apply(lambda r: (r.dataset not in ST) or
                         (r.pred_len in ST_H_BY_DS.get(r.dataset, ST_HORIZONS)), axis=1)
        sub = sub[keep]
    if sub.empty:
        return None
    mean = sub.groupby(["dataset", "pred_len", "name"])["mse"].mean().reset_index()
    piv = mean.pivot_table(index=["dataset", "pred_len"], columns="name", values="mse")
    present = [c for c in cols if c in piv.columns]
    return piv[present]


def _short(name):
    return {"SeasonalNaive": "SNaive", "iTransformer": "iTrans.", "CSLL2-backbone": "CI-only",
            "CSLL2-phaseoff": "mixing", "CSLL2-static": "CSLL2$_s$", "CSLL2": "CSLL2",
            "CSLL2H": "\\textbf{CSLL2H}"}.get(name, name)


def _write_tex(piv, path, caption_cols=True):
    """Bold the row-min; escape dataset names; booktabs."""
    cols = list(piv.columns)
    lines = [r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{ll" + "c" * len(cols) + "}",
             r"\toprule",
             "Dataset & $H$ & " + " & ".join(_short(c) for c in cols) + r" \\", r"\midrule"]
    last_ds = None
    for (ds, H), row in piv.iterrows():
        vals = row.values.astype(float)
        best = np.nanmin(vals)
        cells = []
        for v in vals:
            if v != v:
                cells.append("--")
            elif abs(v - best) < 1e-9:
                cells.append(f"\\textbf{{{v:.3f}}}")
            else:
                cells.append(f"{v:.3f}")
        dsl = ds.replace("_", "\\_") if ds != last_ds else ""
        last_ds = ds
        lines.append(f"{dsl} & {int(H)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}"]
    path.write_text("\n".join(lines) + "\n")


def st_epi_tables(df):
    order = BASELINES + ["CSLL2-backbone", "CSLL2-phaseoff", "CSLL2-static", "CSLL2", "CSLL2H"]
    # traffic claim is scoped to the two canonical speed benchmarks (road-distance ground truth);
    # PEMS04/08 (flow, no distance graph) are excluded from the headline table
    ST_SPEED = ["metr_la", "pems_bay"]
    for tag, ds in [("st", ST_SPEED), ("epi", EPI)]:
        piv = _pivot(df, ds, order)
        if piv is None:
            print(f"[{tag}] no records yet")
            continue
        piv.to_csv(TAB / f"v2_{tag}_mse.csv")
        _write_tex(piv, TAB / f"v2_{tag}_mse.tex")
        print(f"[{tag}] wrote v2_{tag}_mse.csv/.tex  ({piv.shape[0]} settings, {piv.shape[1]} models)")
        print(piv.round(4).to_string())


def ablation_table(df):
    variants = ["CSLL2-backbone", "CSLL2-phaseoff", "CSLL2-static", "CSLL2"]
    piv = _pivot(df, ["metr_la", "pems_bay"] + EPI, variants)
    if piv is None:
        return
    out = piv.copy()
    if {"CSLL2-phaseoff", "CSLL2"}.issubset(out.columns):
        out["phase_gain_%"] = 100 * (out["CSLL2-phaseoff"] - out["CSLL2"]) / out["CSLL2-phaseoff"]
    if {"CSLL2-backbone", "CSLL2"}.issubset(out.columns):
        out["branch_gain_%"] = 100 * (out["CSLL2-backbone"] - out["CSLL2"]) / out["CSLL2-backbone"]
    out.to_csv(TAB / "v2_ablation.csv")
    # LaTeX: one representative row per dataset (mean over horizons), components isolated
    tl = [r"\begin{tabular}{lccccc}", r"\toprule",
          r"Dataset & CI only & +mixing & +static $\phi$ & +dynamic $\phi$ & branch gain \\",
          r"& (backbone) & (phase-off) & & (full) & vs CI (\%) \\", r"\midrule"]
    by_ds = out.reset_index().groupby("dataset")
    for ds, g in by_ds:
        m = g.mean(numeric_only=True)
        def c(k):
            return f"{m[k]:.3f}" if k in m and m[k] == m[k] else "--"
        bg = f"{m['branch_gain_%']:+.1f}" if "branch_gain_%" in m and m['branch_gain_%'] == m['branch_gain_%'] else "--"
        dsl = ds.replace("_", "\\_")
        tl.append(f"{dsl} & {c('CSLL2-backbone')} & {c('CSLL2-phaseoff')} & "
                  f"{c('CSLL2-static')} & \\textbf{{{c('CSLL2')}}} & {bg} \\\\")
    tl += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "v2_ablation.tex").write_text("\n".join(tl) + "\n")
    print("[ablation] wrote v2_ablation.csv/.tex")


def ltsf_table(df):
    """LTSF control: CSLL2 (+variants) vs standard baselines at the 10-epoch protocol."""
    ltsf = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "exchange", "ili"]
    order = ["NLinear", "DLinear", "PatchTST", "iTransformer", "CSLL2-phaseoff", "CSLL2", "CSLL2H"]
    sub = df[df.dataset.isin(ltsf)]
    if sub.empty:
        return
    mean = sub.groupby(["dataset", "pred_len", "name"])["mse"].mean().reset_index()
    piv = mean.pivot_table(index=["dataset", "pred_len"], columns="name", values="mse")
    cols = [c for c in order if c in piv.columns]
    piv = piv[cols]
    piv.to_csv(TAB / "v2_ltsf_mse.csv")
    _write_tex(piv, TAB / "v2_ltsf_mse.tex")
    print(f"[ltsf] wrote v2_ltsf_mse.csv/.tex ({piv.shape[0]} settings)")


def efficiency_tex():
    f = TAB / "v2_efficiency.csv"
    if not f.exists():
        return
    e = pd.read_csv(f).set_index("name")
    order = ["SeasonalNaive", "VAR", "NLinear", "DLinear", "PatchTST", "iTransformer", "CSLL2"]
    tl = [r"\begin{tabular}{lrr}", r"\toprule",
          r"Model & Params & Train time (s) \\", r"\midrule"]
    for m in order:
        if m not in e.index:
            continue
        p = e.params.get(m, 0); t = e.train_s.get(m, 0)
        pstr = "--" if p == 0 else (f"{p/1000:.1f}k" if p < 1e6 else f"{p/1e6:.2f}M")
        tl.append(f"{_short(m).replace('textbf','').replace('{','').replace('}','')} & {pstr} & {t:.1f} \\\\")
    tl += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "v2_efficiency.tex").write_text("\n".join(tl) + "\n")
    print("[efficiency] wrote v2_efficiency.tex")


def dataset_table():
    """Dataset statistics table (auto from the loaded matrices)."""
    sys.path.insert(0, str(ROOT / "code"))
    from csll.data import load_matrix, SEASONAL_PERIOD
    rows = [
        ("ETTh1/h2", "1\\,h", "7", "17\\,420", "$\\{96,192,336,720\\}$", "electricity temp."),
        ("ETTm1/m2", "15\\,min", "7", "69\\,680", "$\\{96,192,336,720\\}$", "electricity temp."),
        ("Weather", "10\\,min", "21", "52\\,696", "$\\{96,192,336,720\\}$", "meteorology"),
        ("Exchange", "1\\,day", "8", "7\\,588", "$\\{96,192,336,720\\}$", "FX rates"),
        ("ILI", "1\\,week", "7", "966", "$\\{24,36,48,60\\}$", "influenza (US)"),
        ("METR-LA", "5\\,min", "207", "34\\,272", "$\\{12,24,48,96\\}$", "traffic speed"),
        ("PEMS-BAY", "5\\,min", "325", "52\\,116", "$\\{12,96\\}$", "traffic speed"),
        ("ILI-US-state", "1\\,week", "49", "360", "$\\{2,4,8\\}$", "influenza (states)"),
        ("ILI-Japan", "1\\,week", "47", "348", "$\\{2,4,8\\}$", "influenza (pref.)"),
        ("ILI-US-region", "1\\,week", "10", "785", "$\\{2,4,8\\}$", "influenza (HHS)"),
    ]
    tl = [r"\begin{tabular}{llrrl l}", r"\toprule",
          r"Dataset & Sampling & $N$ & Length & Horizons $H$ & Domain \\", r"\midrule"]
    for r in rows:
        tl.append(" & ".join(r) + r" \\")
    tl += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "v2_datasets.tex").write_text("\n".join(tl) + "\n")
    print("[datasets] wrote v2_datasets.tex")


def envelope_table(df):
    """The money table: does the measured phase-vs-mixing benefit track the DELAY ADVANTAGE?

    delay_advantage Delta(H) (pre-committed): a leading partner (lag>=H) is more predictive
    than any same-time partner. Delta>0 => phase-as-delay should beat real mixing. We report
    it alongside the crude tau/H (which we show is too coarse) for transparency.
    """
    adv = measure_delay_advantage()             # (dataset,H) -> Delta(H)
    lags = measure_lags()                        # dataset -> dominant-partner |lead|
    rows = []
    for ds in ["metr_la", "pems_bay"] + EPI:     # scoped: speed benchmarks + influenza only
        sub = df[(df.dataset == ds)]
        if sub.empty:
            continue
        for H in sorted(sub.pred_len.unique()):
            g = sub[sub.pred_len == H]
            def m(name):
                v = g[g.name == name]["mse"]
                return float(v.mean()) if len(v) else np.nan
            full, off, back = m("CSLL2"), m("CSLL2-phaseoff"), m("CSLL2-backbone")
            tau = lags.get(ds, np.nan)
            rows.append(dict(dataset=ds, H=H, tau=tau, tau_over_H=(tau / H if tau == tau else np.nan),
                             delta=adv.get(f"{ds}|{H}", np.nan),
                             mse_full=full, mse_phaseoff=off, mse_backbone=back,
                             phase_gain_pct=(100 * (off - full) / off if off == off else np.nan),
                             branch_gain_pct=(100 * (back - full) / back if back == back else np.nan)))
    if rows:
        out = pd.DataFrame(rows)
        out = out.dropna(subset=["phase_gain_pct"])          # drop settings without the phase/mixing pair
        out.to_csv(TAB / "v2_envelope.csv", index=False)
        # LaTeX: dataset, H, measured phase gain, regime, and the (failing) predictor Delta(H)
        tl = [r"\begin{tabular}{llccc}", r"\toprule",
              r"Dataset & $H$ & phase gain (\%) & regime & $\Delta(H)$ pred. \\", r"\midrule"]
        nok = ntot = 0
        for _, r in out.iterrows():
            g = r.phase_gain_pct
            d = r.delta
            gs = f"{g:+.1f}" if g == g else "--"
            reg = "phase" if (g == g and g > 2) else ("mixing" if (g == g and g < -1) else "neutral")
            pred = "--"
            if g == g and d == d:
                ntot += 1
                hit = ((d > 0) == (g > 0.2)) or (abs(d) < 1e-3 and abs(g) < 1)
                nok += int(hit)
                pred = r"\checkmark" if hit else r"$\times$"
            dsl = r.dataset.replace("_", "\\_")
            tl.append(f"{dsl} & {int(r.H)} & {gs} & {reg} & {pred} \\\\")
        tl += [r"\midrule",
               f"\\multicolumn{{5}}{{l}}{{\\footnotesize $\\Delta(H)$ sign agrees with outcome "
               f"in {nok}/{ntot} settings --- no reliable a-priori predictor.}} \\\\",
               r"\bottomrule", r"\end{tabular}"]
        (TAB / "v2_envelope.tex").write_text("\n".join(tl) + "\n")
        print("[envelope] wrote v2_envelope.csv/.tex")
        print(out.round(3).to_string(index=False))


def measure_lags():
    """Pre-committed empirical inter-series lead tau (native samples) per dataset.

    DEFINITION (fixed in advance, not tuned to outcomes): deseasonalise the training split
    (per-period profile removal; log1p for counts; missing->NaN masked); for each channel find
    its single MOST-correlated partner (max |normalised cross-correlation| over lags in
    [-ML,ML]) and record the |lag| at that peak; tau = MEDIAN over channels of that dominant-
    partner lag. Rationale: the value envelope concerns whether the *dominant* cross-series
    coupling carries a lead beyond same-time mixing, so tau must reflect the dominant partner,
    not a mean over all pairs (which on traffic is inflated by a noisy small-|xcorr| tail and
    mis-predicts the H=2 point). Cached to v2_lags.json; delete to recompute.
    """
    cache = TAB / "v2_lags.json"
    if cache.exists():
        return json.load(open(cache))
    sys.path.insert(0, str(ROOT / "code"))
    from csll.data import load_matrix, split_indices, SEASONAL_PERIOD
    out = {}
    for ds in ST + EPI:
        try:
            X = load_matrix(ds)
        except Exception:
            continue
        L = 36 if ds.startswith("ili") else 96
        tr0, tr1 = split_indices(ds, len(X), L)["train"]
        Xtr = X[tr0:tr1].astype(float)
        per = SEASONAL_PERIOD.get(ds, 1)
        Z = np.log1p(np.clip(Xtr, 0, None)) if ds.startswith("ili") else Xtr.copy()
        if ds.startswith(("metr", "pems")):
            Z[Xtr == 0] = np.nan
        wk = np.arange(len(Z)) % per
        prof = np.stack([np.nanmean(Z[wk == w], 0) for w in range(per)])
        R = np.where(np.isnan(Z - prof[wk]), 0.0, Z - prof[wk])
        R = (R - R.mean(0)) / (R.std(0) + 1e-8)
        N = R.shape[1]
        idx = np.linspace(0, N - 1, min(80, N)).astype(int)   # subsample channels for speed
        R = R[:, idx]; N = R.shape[1]
        ML = min(12, L // 3)
        nfft = 1
        while nfft < 2 * len(R):
            nfft *= 2
        F = np.fft.rfft(R, nfft, axis=0)
        dom_lag = []
        for a in range(N):
            best_pk, best_lag = 0.0, 0
            for b in range(N):
                if b == a:
                    continue
                c = np.fft.irfft(F[:, b] * np.conj(F[:, a]), nfft)
                c = np.concatenate([c[-ML:], c[:ML + 1]])
                k = int(np.abs(c).argmax())
                pk = float(np.abs(c[k]) / len(R))
                if pk > best_pk:
                    best_pk, best_lag = pk, k - ML
            dom_lag.append(abs(best_lag))
        out[ds] = float(np.median(dom_lag))
    TAB.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(cache, "w"), indent=2)
    return out


def hybrid_table(df):
    """Show CSLL2H matches the best specialised branch per regime with one model."""
    reps = [("ili_us_state", 2, "phase"), ("ili_us_state", 8, "phase"),
            ("ili_japan", 2, "phase"), ("metr_la", 12, "mixing"), ("metr_la", 24, "mixing"),
            ("weather", 96, "mixing"), ("weather", 336, "mixing")]
    cols = ["CSLL2-backbone", "CSLL2-phaseoff", "CSLL2", "CSLL2H"]
    rows = []
    for ds, H, regime in reps:
        g = df[(df.dataset == ds) & (df.pred_len == H)]
        if g.empty:
            continue
        r = {"dataset": ds, "H": H, "regime": regime}
        for c in cols:
            v = g[g.name == c]["mse"]
            r[c] = float(v.mean()) if len(v) else np.nan
        rows.append(r)
    if not rows:
        return
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "v2_hybrid.csv", index=False)
    tl = [r"\begin{tabular}{llcccc}", r"\toprule",
          r"Dataset & $H$ & CI-only & mixing & phase & \textbf{hybrid} \\", r"\midrule"]
    for _, r in out.iterrows():
        vals = {c: r[c] for c in cols}
        best_spec = np.nanmin([r["CSLL2-phaseoff"], r["CSLL2"]])   # best of mixing/phase
        cells = []
        for c in cols:
            v = r[c]
            s = "--" if v != v else f"{v:.3f}"
            if c == "CSLL2H" and v == v and v <= best_spec * 1.02:
                s = f"\\textbf{{{v:.3f}}}"
            cells.append(s)
        dsl = r.dataset.replace("_", "\\_")
        tl.append(f"{dsl} & {int(r.H)} & " + " & ".join(cells) + r" \\")
    tl += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "v2_hybrid.tex").write_text("\n".join(tl) + "\n")
    print("[hybrid] wrote v2_hybrid.csv/.tex")
    print(out.round(3).to_string(index=False))


def measure_delay_advantage():
    """Pre-committed envelope statistic Delta(H) per (dataset, horizon).

    For each channel i: rho0(i) = max_j |xcorr_{ij}| at lag 0 (best same-time partner);
    rhoH(i) = max_j |xcorr_{ij}| over |lag| in [H, ML] (best partner leading by >= H).
    Delta(H) = mean_i (rhoH(i) - rho0(i)). Delta > 0 means some series is better predicted
    by a leading partner than by any same-time one -> the delayed coupling carries
    non-redundant predictive information at the horizon scale -> phase-as-delay should beat
    real-valued mixing. This is the corrected envelope variable (the crude tau/H mis-predicts
    traffic H=2; Delta does not). Cached to v2_delay_adv.json.
    """
    cache = TAB / "v2_delay_adv.json"
    if cache.exists():
        return json.load(open(cache))
    sys.path.insert(0, str(ROOT / "code"))
    from csll.data import load_matrix, split_indices, SEASONAL_PERIOD
    from csll.config import horizons_for
    out = {}
    for ds in ST + EPI:
        try:
            X = load_matrix(ds)
        except Exception:
            continue
        L = 36 if ds.startswith("ili") else 96
        tr0, tr1 = split_indices(ds, len(X), L)["train"]
        Xtr = X[tr0:tr1].astype(float)
        per = SEASONAL_PERIOD.get(ds, 1)
        Z = np.log1p(np.clip(Xtr, 0, None)) if ds.startswith("ili") else Xtr.copy()
        if ds.startswith(("metr", "pems")):
            Z[Xtr == 0] = np.nan
        wk = np.arange(len(Z)) % per
        prof = np.stack([np.nanmean(Z[wk == w], 0) for w in range(per)])
        R = np.where(np.isnan(Z - prof[wk]), 0.0, Z - prof[wk])
        R = (R - R.mean(0)) / (R.std(0) + 1e-8)
        N = R.shape[1]
        idx = np.linspace(0, N - 1, min(80, N)).astype(int)
        R = R[:, idx]; N = R.shape[1]
        ML = min(12, L // 3)
        nfft = 1
        while nfft < 2 * len(R):
            nfft *= 2
        F = np.fft.rfft(R, nfft, axis=0)
        C = np.zeros((N, N, 2 * ML + 1))
        for a in range(N):
            for b in range(N):
                if a == b:
                    continue
                c = np.fft.irfft(F[:, b] * np.conj(F[:, a]), nfft)
                c = np.concatenate([c[-ML:], c[:ML + 1]])
                C[a, b] = np.abs(c) / len(R)
        lagax = np.arange(-ML, ML + 1)
        rho0 = C[:, :, ML].max(1)
        for H in horizons_for(ds):
            maskH = np.abs(lagax) >= H
            rhoH = C[:, :, maskH].max((1, 2)) if maskH.any() else np.zeros(N)
            out[f"{ds}|{H}"] = float(np.mean(rhoH - rho0))
    TAB.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(cache, "w"), indent=2)
    return out


def efficiency(df):
    g = df[df.name.isin(BASELINES + ["CSLL2"])].groupby("name").agg(
        params=("params", "median"), train_s=("train_s", "mean")).reset_index()
    g.to_csv(TAB / "v2_efficiency.csv", index=False)
    print("[efficiency] wrote v2_efficiency.csv")
    print(g.round(1).to_string(index=False))


def dm_tests(df):
    rows = []
    for ds in ST + EPI:
        for H in sorted(df[df.dataset == ds].pred_len.unique()):
            csll = df[(df.dataset == ds) & (df.pred_len == H) & (df.name == "CSLL2") & (df.seed == 0)]
            if csll.empty:
                continue
            cstem = csll.iloc[0]["stem"]
            cw = RAW / f"{cstem}.perwin.npy"
            if not cw.exists():
                continue
            cwin = np.load(cw)
            base = df[(df.dataset == ds) & (df.pred_len == H) & (df.seed == 0) &
                      (df.name.isin(BASELINES))]
            if base.empty:
                continue
            best = base.loc[base.mse.idxmin()]
            bw = RAW / f"{best['stem']}.perwin.npy"
            if not bw.exists():
                continue
            bwin = np.load(bw)
            n = min(len(cwin), len(bwin))
            stat, p = diebold_mariano(cwin[:n], bwin[:n])
            rows.append(dict(dataset=ds, H=H, best_baseline=best["name"],
                             csll_mse=float(csll.iloc[0]["mse"]), base_mse=float(best["mse"]),
                             dm_stat=stat, p_value=p,
                             csll_better=bool(csll.iloc[0]["mse"] < best["mse"]),
                             significant=bool(p < 0.05)))
    if rows:
        dm = pd.DataFrame(rows)
        dm.to_csv(TAB / "v2_dm.csv", index=False)
        tl = [r"\begin{tabular}{llcccc}", r"\toprule",
              r"Dataset & $H$ & best base & CSLL2 & base & sig. \\", r"\midrule"]
        for _, r in dm.iterrows():
            mark = ("$\\prec$" if (r.significant and not r.csll_better) else
                    "$\\succ$" if (r.significant and r.csll_better) else "$\\approx$")
            dsl = r.dataset.replace("_", "\\_")
            tl.append(f"{dsl} & {int(r.H)} & {_short(r.best_baseline)} & "
                      f"{r.csll_mse:.3f} & {r.base_mse:.3f} & {mark} \\\\")
        tl += [r"\bottomrule", r"\end{tabular}"]
        (TAB / "v2_dm.tex").write_text("\n".join(tl) + "\n")
        nsig_worse = int(((dm.significant) & (~dm.csll_better)).sum())
        nsig_better = int(((dm.significant) & (dm.csll_better)).sum())
        print(f"[dm] wrote v2_dm.csv/.tex ({len(rows)} settings; "
              f"CSLL2 sig-better {nsig_better}, sig-worse {nsig_worse})")


if __name__ == "__main__":
    TAB.mkdir(parents=True, exist_ok=True)
    df = load()
    if df.empty:
        print("no results yet"); sys.exit(0)
    print(f"loaded {len(df)} runs across {df.dataset.nunique()} datasets\n")
    st_epi_tables(df)
    print()
    hybrid_table(df)
    print()
    ltsf_table(df)
    dataset_table()
    print()
    ablation_table(df)
    print()
    envelope_table(df)
    print()
    efficiency(df)
    efficiency_tex()
    dm_tests(df)
    print("\nAGGREGATE_V2_DONE")
