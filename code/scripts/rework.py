#!/usr/bin/env python3
"""Rework experiments: CSLLX (iTransformer backbone + lead-lag module) vs iTransformer on
the lead-lag datasets. Prints results/errors clearly; skips already-done; sequential (OMP full)."""
import sys, traceback, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
import run_experiment as R
JOBS = [("CSLLX","traffic",96),("CSLLX","electricity",96),
        ("CSLLX","traffic",336),("CSLLX","electricity",336),
        ("iTransformer","traffic",336)]
_ap = argparse.ArgumentParser()
_ap.add_argument("--shard", type=int, default=0)
_ap.add_argument("--nshards", type=int, default=1)
_a = _ap.parse_args()
if _a.nshards > 1:
    JOBS = [j for i, j in enumerate(JOBS) if i % _a.nshards == _a.shard]
for model, ds, H in JOBS:
    stem = f"{ds}__{model}__L96__H{H}__s0"
    if (R.RESULTS / f"{stem}.json").exists():
        print(f"skip {stem}", flush=True); continue
    try:
        res = R.run(model, ds, H, 96, 0, train_over={"epochs": 6})
        print(f"OK {stem}: MSE={res['metrics']['mse']:.4f}  ({res['wall_time_s']:.0f}s)", flush=True)
    except Exception as e:
        print(f"FAIL {stem}: {e}\n{traceback.format_exc()}", flush=True)
print("REWORK_DONE", flush=True)
