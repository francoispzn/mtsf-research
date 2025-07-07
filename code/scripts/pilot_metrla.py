#!/usr/bin/env python3
"""METR-LA pilot (go/no-go for the v2 campaign).

Trains, under ONE uniform protocol (L=96, H in {12,48}, 30 epochs / patience 5, seed 0,
sequential for memory safety):
  baselines : SeasonalNaive, NLinear, DLinear, iTransformer
  ours      : CSLL2 (dynamic, direct), CSLL2-static, CSLL2-backbone (branch off),
              CSLL2-phaseoff (real mixing; falls back to the head path -- a real mixer
              cannot reach the future by construction)

CSLL2 checkpoints are saved so validate_delays_metrla.py can extract learned delays and
compare them against road-network distances / kinematic shockwave physics.

Run:  MTSF_DEVICE=cpu MTSF_TORCH_THREADS=4 .venv/bin/python3 code/scripts/pilot_metrla.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "scripts"))

from run_experiment import run   # noqa: E402

DATASET = "metr_la"
L = 96
HORIZONS = [12, 48]
BUDGET = dict(epochs=30, patience=5)      # uniform for every model (addresses review M-9)


if __name__ == "__main__":
    done, failed = [], []
    # sequential execution: one job at a time (memory-bound machine)
    for H in HORIZONS:
        for model, over, suffix in [
            ("SeasonalNaive", {}, None),
            ("NLinear", {}, None),
            ("DLinear", {}, None),
            ("iTransformer", {}, None),
            ("CSLL2", {}, None),
            ("CSLL2", dict(dynamic=False), "static"),
            ("CSLL2", dict(freeze_gate=True), "backbone"),
            ("CSLL2", dict(use_phase=False, direct=False), "phaseoff"),
        ]:
            name = model + (f"-{suffix}" if suffix else "")
            tag = f"{DATASET}__{name}__L{L}__H{H}__s0"
            if (ROOT / "results" / "raw" / f"{tag}.json").exists():
                print(f"[skip] {tag}", flush=True)
                continue
            try:
                res = run(model, DATASET, H, L, 0, cfg_over=(over or None),
                          train_over=dict(BUDGET), tag=tag,
                          save_model=(model.startswith("CSLL2") and not suffix))
                m = res["metrics"]
                print(f"[done] {tag}: mse={m['mse']:.4f} mae={m['mae']:.4f} "
                      f"params={res['history'].get('params')} "
                      f"wall={res['wall_time_s']:.0f}s", flush=True)
                done.append(tag)
            except Exception:
                print(f"[FAIL] {tag}\n{traceback.format_exc()}", flush=True)
                failed.append(tag)
    print(f"PILOT_DONE ok={len(done)} failed={len(failed)}", flush=True)
