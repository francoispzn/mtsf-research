#!/usr/bin/env python3
"""Aggregate results/raw/*.json into tidy tables (CSV + LaTeX), win counts, and
Diebold-Mariano significance tests. Writes to results/tables/."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from csll.stats import diebold_mariano, paired_ttest         # noqa: E402

RAW = ROOT / "results" / "raw"
TAB = ROOT / "results" / "tables"
MAIN_ORDER = ["SeasonalNaive", "VAR", "NLinear", "DLinear", "LSTM", "Transformer",
              "PatchTST", "iTransformer", "CSLL"]
TRACTABLE = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "exchange", "ili"]  # real, non-heavy


def load_records():
    main, ablin = [], []
    for p in sorted(RAW.glob("*.json")):
        d = json.load(open(p))
        rec = dict(model=d["model"], dataset=d["dataset"], pred_len=d["pred_len"],
                   seed=d["seed"], seq_len=d.get("seq_len"), tag=d.get("tag"),
                   stem=p.stem, params=d.get("history", {}).get("params", 0),
                   train_time=d.get("history", {}).get("train_time_s", 0.0),
                   **{k: d["metrics"][k] for k in ["mse", "mae", "rmse", "mase", "smape"]})
        if d.get("tag") and "CSLL-" in d["tag"]:
            rec["variant"] = d["tag"].split("CSLL-")[1].split("__")[0]
            ablin.append(rec)
        else:
            main.append(rec)
    return pd.DataFrame(main), pd.DataFrame(ablin)


def main_tables(df: pd.DataFrame):
    if df.empty:
        print("no main records"); return
    TAB.mkdir(parents=True, exist_ok=True)
    for metric in ["mse", "mae", "mase"]:
        mean = df.groupby(["dataset", "pred_len", "model"])[metric].mean().reset_index()
        piv = mean.pivot_table(index=["dataset", "pred_len"], columns="model", values=metric)
        cols = [m for m in MAIN_ORDER if m in piv.columns]
        piv = piv[cols]
        piv.to_csv(TAB / f"main_{metric}.csv")
        std = df.groupby(["dataset", "pred_len", "model"])[metric].std().reset_index()
        std.pivot_table(index=["dataset", "pred_len"], columns="model", values=metric).to_csv(
            TAB / f"main_{metric}_std.csv")
        _latex_table(piv, TAB / f"main_{metric}.tex", metric.upper())
    # summary: average metric per model (over dataset,horizon means) + win counts
    mse_mean = df.groupby(["dataset", "pred_len", "model"])["mse"].mean().reset_index()
    piv = mse_mean.pivot_table(index=["dataset", "pred_len"], columns="model", values="mse")
    wins = (piv.idxmin(axis=1).value_counts())
    summary = pd.DataFrame({
        "avg_mse": piv.mean(axis=0),
        "avg_mae": df.groupby("model")["mae"].mean(),
        "wins_mse": wins,
    }).reindex([m for m in MAIN_ORDER if m in piv.columns])
    summary.to_csv(TAB / "summary.csv")
    print("Summary (avg MSE across dataset x horizon settings; wins = #settings best):")
    print(summary.round(4).to_string())


def _esc(x) -> str:
    """Escape LaTeX-special characters in text cells (headers, labels)."""
    return str(x).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace(
        "%", r"\%").replace("&", r"\&").replace("#", r"\#")


def _latex_table(piv: pd.DataFrame, path: Path, caption: str):
    models = [_esc(m) for m in piv.columns]
    lines = [r"\begin{tabular}{ll" + "c" * len(models) + "}", r"\toprule",
             "Dataset & $H$ & " + " & ".join(models) + r" \\", r"\midrule"]
    for (ds, h), row in piv.iterrows():
        vals = row.values.astype(float)
        best = np.nanmin(vals)
        cells = []
        for v in vals:
            s = "--" if np.isnan(v) else f"{v:.3f}"
            if not np.isnan(v) and abs(v - best) < 1e-9:
                s = r"\textbf{" + s + "}"
            cells.append(s)
        lines.append(f"{_esc(ds)} & {h} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines))


def ablation_table(df: pd.DataFrame, main: pd.DataFrame):
    if df.empty:
        print("no ablation records"); return
    csll_full = main[main.model == "CSLL"][["dataset", "pred_len", "seed", "mse", "mae"]].copy()
    csll_full["variant"] = "full"
    keep = df[["dataset", "pred_len", "seed", "variant", "mse", "mae"]]
    allv = pd.concat([keep, csll_full], ignore_index=True)
    piv = allv.groupby(["dataset", "pred_len", "variant"])["mse"].mean().reset_index()
    piv = piv.pivot_table(index=["dataset", "pred_len"], columns="variant", values="mse")
    order = ["full", "As_static", "A1_realphase", "A0_backbone", "A2_1band",
             "A4_2band", "A4_8band", "A3_freecplx", "A5_lowrank8"]
    piv = piv[[c for c in order if c in piv.columns]]
    # drop rows with no ablation-variant data (only the full model) -> no all-dash clutter
    variant_cols = [c for c in piv.columns if c != "full"]
    piv = piv[piv[variant_cols].notna().any(axis=1)]
    piv.to_csv(TAB / "ablation_mse.csv")
    # human-readable column headers for the LaTeX table (no underscores)
    nice = {"full": "full", "As_static": "static", "A1_realphase": "phase-off",
            "A0_backbone": "no branch", "A2_1band": "1 band", "A4_2band": "2 bands",
            "A4_8band": "8 bands", "A3_freecplx": "free cplx", "A5_lowrank8": "low-rank"}
    _latex_table(piv.rename(columns=nice), TAB / "ablation_mse.tex", "Ablation MSE")
    print("\nAblation (MSE) written to ablation_mse.csv/.tex")
    print(piv.round(4).to_string())


def efficiency_table(main: pd.DataFrame):
    if main.empty:
        return
    g = main.groupby("model").agg(params=("params", "median"),
                                  avg_train_s=("train_time", "mean"))
    g = g.reindex([m for m in MAIN_ORDER if m in g.index])
    g.to_csv(TAB / "efficiency.csv")
    lines = [r"\begin{tabular}{lrr}", r"\toprule",
             r"Model & Params (median) & Avg train (s) \\", r"\midrule"]
    for m, row in g.iterrows():
        lines.append(f"{m} & {int(row['params']):,} & {row['avg_train_s']:.1f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "efficiency.tex").write_text("\n".join(lines))
    print("\nEfficiency (params / avg train time) -> efficiency.csv/.tex")
    print(g.round(1).to_string())


def rank_and_pareto(df: pd.DataFrame):
    """Scale-free average rank + accuracy/cost Pareto frontier on the settings where every
    main model has a result (the fair comparison set). Raw mean-MSE over-weights the few
    large-magnitude settings (ILI); average rank is magnitude-invariant and reported alongside.
    Writes summary_ranked.csv/.tex and pareto.csv."""
    if df.empty:
        print("no records for rank/pareto"); return
    g = df.groupby(["dataset", "pred_len", "model"]).agg(
        mse=("mse", "mean"), mae=("mae", "mean")).reset_index()
    mse_piv = g.pivot_table(index=["dataset", "pred_len"], columns="model", values="mse")
    mae_piv = g.pivot_table(index=["dataset", "pred_len"], columns="model", values="mae")
    models = [m for m in MAIN_ORDER if m in mse_piv.columns]

    complete = mse_piv[models].dropna(axis=0, how="any")          # common-complete settings
    ranks = complete.rank(axis=1, method="min")                   # 1 = best per setting
    avg_rank = ranks.mean(axis=0)
    wins = complete.idxmin(axis=1).value_counts()
    avg_mse = complete.mean(axis=0)
    avg_mae = mae_piv[models].dropna(axis=0, how="any").mean(axis=0)

    med_params = df.groupby("model")["params"].median()
    tract = df[df.dataset.isin(TRACTABLE)]
    train_s = tract.groupby("model")["train_time"].mean()          # tractable-only (fair)

    summ = pd.DataFrame({
        "avg_rank": avg_rank, "avg_mse": avg_mse, "avg_mae": avg_mae,
        "wins": wins, "params": med_params, "train_s": train_s,
    }).reindex(models)
    summ["wins"] = summ["wins"].fillna(0).astype(int)

    # Pareto frontier on (avg_rank lower-better, params lower-better): a model is on the
    # frontier if no other model is at least as good on BOTH axes and strictly better on one.
    def dominated(m):
        rm, pm = summ.loc[m, "avg_rank"], summ.loc[m, "params"]
        return any((summ.loc[o, "avg_rank"] <= rm and summ.loc[o, "params"] <= pm and
                    (summ.loc[o, "avg_rank"] < rm or summ.loc[o, "params"] < pm))
                   for o in summ.index if o != m)
    summ["pareto"] = [not dominated(m) for m in summ.index]
    summ.to_csv(TAB / "summary_ranked.csv")

    # LaTeX summary table (portfolio headline table)
    lines = [r"\begin{tabular}{lrrrrrrc}", r"\toprule",
             r"Model & Avg rank & Avg MSE & Avg MAE & Wins & Params & Train (s) & Pareto \\",
             r"\midrule"]
    best_rank = summ["avg_rank"].min()
    for m, r in summ.iterrows():
        rk = f"{r['avg_rank']:.2f}"
        if abs(r["avg_rank"] - best_rank) < 1e-9:
            rk = r"\textbf{" + rk + "}"
        star = r"$\checkmark$" if r["pareto"] else ""
        lines.append(f"{m} & {rk} & {r['avg_mse']:.3f} & {r['avg_mae']:.3f} & "
                     f"{int(r['wins'])} & {int(r['params']):,} & {r['train_s']:.1f} & {star}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "summary_ranked.tex").write_text("\n".join(lines))

    print(f"\nRanked summary + Pareto (common-complete settings: {len(complete)} of "
          f"{len(mse_piv)}); train_s on tractable only:")
    print(summ.round(3).to_string())
    frontier = [m for m in summ.index if summ.loc[m, "pareto"]]
    print("Pareto-optimal (avg_rank vs params):", ", ".join(frontier))
    return summ


def dm_tests(main: pd.DataFrame):
    """For each (dataset,horizon,seed=0): DM test CSLL vs the best-mean baseline."""
    rows = []
    for (ds, h), g in main.groupby(["dataset", "pred_len"]):
        base = g[(g.model != "CSLL")]
        if base.empty or (g.model == "CSLL").sum() == 0:
            continue
        best_model = base.groupby("model")["mse"].mean().idxmin()
        cs = RAW / f"{ds}__CSLL__L{int(g.seq_len.iloc[0])}__H{h}__s0.perwin.npy"
        bs = RAW / f"{ds}__{best_model}__L{int(g.seq_len.iloc[0])}__H{h}__s0.perwin.npy"
        if not (cs.exists() and bs.exists()):
            continue
        a, b = np.load(cs), np.load(bs)
        dm, p = diebold_mariano(a, b, h=h, power=1)
        rows.append(dict(dataset=ds, pred_len=h, best_baseline=best_model,
                         csll_mse=float(a.mean()), base_mse=float(b.mean()),
                         dm_stat=dm, p_value=p,
                         csll_better=bool(a.mean() < b.mean()), significant=bool(p < 0.05)))
    if rows:
        d = pd.DataFrame(rows)
        d.to_csv(TAB / "dm_tests.csv", index=False)
        print("\nDiebold-Mariano (CSLL vs best baseline) written to dm_tests.csv")
        print(d.round(4).to_string())


if __name__ == "__main__":
    main_df, abl_df = load_records()
    print(f"loaded {len(main_df)} main + {len(abl_df)} ablation records")
    # Synthetic is an INTERPRETABILITY control (planted delays), not a headline benchmark:
    # keep it out of the real-data benchmark tables (nine real datasets), but retain it for
    # the ablation table where its full-CSLL row is the baseline for the synthetic ablations.
    bench_df = main_df[main_df.dataset != "synthetic"].copy()
    main_tables(bench_df)
    efficiency_table(bench_df)
    rank_and_pareto(bench_df)
    ablation_table(abl_df, main_df)
    dm_tests(bench_df)
