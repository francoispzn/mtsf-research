#!/usr/bin/env python3
"""Backbone vs static vs dynamic CSLL on the high-dimensional lead-lag datasets
(Electricity, Traffic), where the dynamic-delay contribution is expected to matter most.
Uses the low-rank variant. Resumable (skips existing tags)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
import run_experiment as R  # noqa: E402

VARIANTS = {
    "A0_backbone":  {"freeze_gate": True, "low_rank": 16},
    "As_static":    {"dynamic": False, "low_rank": 16},
    # dynamic CSLL main is produced by the heavy main stage (model "CSLL", low_rank auto)
}

if __name__ == "__main__":
    for ds in ["electricity", "traffic"]:
        for H in [96, 336]:
            for tag, over in VARIANTS.items():
                stem = f"{ds}__CSLL-{tag}__L96__H{H}__s0"
                if (R.RESULTS / f"{stem}.json").exists():
                    print("skip", stem); continue
                try:
                    res = R.run("CSLL", ds, H, 96, 0, cfg_over=over,
                                train_over={"epochs": 6, "batch_size": 8 if ds == "traffic" else 16},
                                tag=stem)
                    print(f"{stem}  MSE={res['metrics']['mse']:.4f}", flush=True)
                except Exception as e:
                    print(f"{stem} ERROR: {e}", flush=True)
    print("HEAVY_ABLATIONS_DONE")
