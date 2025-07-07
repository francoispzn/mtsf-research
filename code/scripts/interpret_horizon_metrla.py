#!/usr/bin/env python3
"""Horizon-dependence of learned delays on METR-LA (the physics-recovery experiment).

Theory (from the kill-test analysis): in direct mode the forecast-optimal delay for a pair
with physical lag l is l itself only while h < l; for horizons beyond the lead, the loss
prefers larger delays that exploit the source's own autocorrelation ("prediction-lead
smear"). Prediction: learned delays approach the PHYSICAL lags as H shrinks toward the lag
scale (1-3 samples on METR-LA), and drift upward as H grows.

This script trains CSLL2-static at H in {2, 3, 6, 24} (H=12/48 come from the pilot),
saves checkpoints, then runs the delay validation for each horizon.

Run (after pilot_metrla.py finishes; sequential, memory-safe):
  MTSF_DEVICE=cpu MTSF_TORCH_THREADS=4 .venv/bin/python3 code/scripts/interpret_horizon_metrla.py
"""
from __future__ import annotations

import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "scripts"))

from run_experiment import run   # noqa: E402

L = 96
HORIZONS = [2, 3, 6]
BUDGET = dict(epochs=30, patience=5)

VARIANTS = [
    ("CSLL2", dict(dynamic=False), "static"),          # physics/recovery target
    ("CSLL2", {}, None),                               # dynamic
    ("CSLL2", dict(use_phase=False, direct=False), "phaseoff"),
    ("CSLL2", dict(freeze_gate=True, gate_init=0.0), "backbone"),
]

if __name__ == "__main__":
    for H in HORIZONS:
        for model, over, suffix in VARIANTS:
            name = model + (f"-{suffix}" if suffix else "")
            tag = f"metr_la__{name}__L{L}__H{H}__s0"
            if not (ROOT / "results" / "raw" / f"{tag}.json").exists():
                try:
                    res = run(model, "metr_la", H, L, 0, cfg_over=(over or None),
                              train_over=dict(BUDGET), tag=tag,
                              save_model=(suffix in (None, "static")))
                    print(f"[done] {tag}: mse={res['metrics']['mse']:.4f} "
                          f"wall={res['wall_time_s']:.0f}s", flush=True)
                except Exception:
                    print(f"[FAIL] {tag}\n{traceback.format_exc()}", flush=True)
                    continue
            else:
                print(f"[skip] {tag}", flush=True)
        subprocess.run([sys.executable, str(ROOT / "code/scripts/validate_delays_metrla.py"),
                        "--H", str(H), "--variant", "CSLL2-static"], check=False)
    # also validate the pilot's H=12 static model if its checkpoint exists
    if (ROOT / "results" / "raw" / f"metr_la__CSLL2-static__L{L}__H12__s0.pt").exists():
        subprocess.run([sys.executable, str(ROOT / "code/scripts/validate_delays_metrla.py"),
                        "--H", "12", "--variant", "CSLL2-static"], check=False)
    print("INTERPRET_HORIZON_DONE", flush=True)
