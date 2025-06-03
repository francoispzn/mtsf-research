#!/usr/bin/env python3
"""Orchestrate the full experiment sweep (resumable; each run writes its own JSON).

Encodes the compute-aware protocol:
  * main benchmark: 9 real datasets x models x horizons x seeds
  * VAR only on low-dim datasets; CSLL uses low-rank on electricity/traffic
  * seeds: 3 on the 7 tractable datasets, 1 on electricity/traffic (documented)
  * ablations: CSLL variants on a representative subset

Usage examples:
  python code/scripts/run_all.py --which main --datasets ETTh1 ETTm1 --dry-run
  python code/scripts/run_all.py --which main --skip-existing
  python code/scripts/run_all.py --which ablation --datasets ETTh1 exchange traffic
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from csll.config import horizons_for, seq_len_for, LOWDIM_FOR_VAR   # noqa: E402
import run_experiment as R                                          # noqa: E402

ALL_DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "exchange", "ili",
                "electricity", "traffic"]
MAIN_MODELS = ["SeasonalNaive", "VAR", "NLinear", "DLinear", "LSTM", "Transformer",
               "PatchTST", "iTransformer", "CSLL"]
HEAVY = {"electricity", "traffic"}
BIG = {"ETTm1", "ETTm2", "weather"}          # 2 seeds here (compute), 3 elsewhere

# CSLL ablation variants: (tag_suffix, cfg_overrides)
ABLATIONS = {
    "A0_backbone":  {"freeze_gate": True},                 # spectral branch off (pure CI)
    "As_static":    {"dynamic": False},                    # static (LTI) delays vs dynamic
    "A1_realphase": {"use_phase": False, "dynamic": False}, # phase off (real mixing)
    "A2_1band":     {"n_bands": 1},
    "A4_2band":     {"n_bands": 2},
    "A4_8band":     {"n_bands": 8},
    "A3_freecplx":  {"free_complex": True},                # unstructured complex operator
    "A5_lowrank8":  {"low_rank": 8},
}


def seeds_for(dataset: str, model: str, base_seeds):
    if model in ("VAR", "SeasonalNaive"):
        return [0]                          # deterministic
    if dataset in HEAVY:
        return [0]                          # compute-aware: 1 seed on heavy datasets
    if dataset in BIG:
        return list(base_seeds)[:2]         # 2 seeds on the three largest datasets
    return base_seeds


def main_jobs(datasets, models, base_seeds, horizons=None):
    jobs = []
    for ds in datasets:
        L = seq_len_for(ds)
        for H in (horizons or horizons_for(ds)):
            for m in models:
                if m == "VAR" and ds not in LOWDIM_FOR_VAR:
                    continue
                for s in seeds_for(ds, m, base_seeds):
                    jobs.append(dict(model=m, dataset=ds, pred_len=H, seq_len=L, seed=s))
    return jobs


def ablation_jobs(datasets, base_seeds, horizons=None):
    jobs = []
    for ds in datasets:
        if ds in HEAVY:            # full-mode ablations are intractable at N>=321; skip
            continue
        L = seq_len_for(ds)
        for H in (horizons or horizons_for(ds)):
            for tag, over in ABLATIONS.items():
                over = {k: v for k, v in over.items() if not k.startswith("_")}
                for s in seeds_for(ds, "CSLL", base_seeds):
                    jobs.append(dict(model="CSLL", dataset=ds, pred_len=H, seq_len=L,
                                     seed=s, cfg_over=over,
                                     tag=f"{ds}__CSLL-{tag}__L{L}__H{H}__s{s}"))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["main", "ablation", "all"], default="main")
    ap.add_argument("--datasets", nargs="*", default=ALL_DATASETS)
    ap.add_argument("--models", nargs="*", default=MAIN_MODELS)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--horizons", nargs="*", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0)      # this worker's index
    ap.add_argument("--nshards", type=int, default=1)    # total workers (job i -> i % nshards)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    jobs = []
    if args.which in ("main", "all"):
        jobs += main_jobs(args.datasets, args.models, args.seeds, args.horizons)
    if args.which in ("ablation", "all"):
        jobs += ablation_jobs([d for d in args.datasets], args.seeds, args.horizons)

    if args.nshards > 1:
        jobs = [j for idx, j in enumerate(jobs) if idx % args.nshards == args.shard]
    print(f"[run_all] {len(jobs)} jobs (which={args.which}, shard {args.shard}/{args.nshards})")
    if args.dry_run:
        for j in jobs[:40]:
            print("  ", j.get("tag") or f"{j['dataset']}/{j['model']}/H{j['pred_len']}/s{j['seed']}")
        if len(jobs) > 40:
            print(f"   ... (+{len(jobs)-40} more)")
        return

    done = 0
    t_start = time.time()
    for i, j in enumerate(jobs):
        stem = j.get("tag") or f"{j['dataset']}__{j['model']}__L{j['seq_len']}__H{j['pred_len']}__s{j['seed']}"
        out = R.RESULTS / f"{stem}.json"
        if args.skip_existing and out.exists():
            done += 1
            continue
        train_over = {"epochs": args.epochs} if args.epochs else None
        t0 = time.time()
        try:
            res = R.run(train_over=train_over, **j)
            done += 1
            mse = res["metrics"]["mse"]
            print(f"[{i+1}/{len(jobs)}] {stem}  MSE={mse:.4f}  "
                  f"{time.time()-t0:.1f}s  (elapsed {int(time.time()-t_start)}s)", flush=True)
        except Exception as e:
            print(f"[{i+1}/{len(jobs)}] {stem}  ERROR: {e}", flush=True)
    print(f"[run_all] completed {done}/{len(jobs)} in {int(time.time()-t_start)}s")


if __name__ == "__main__":
    main()
