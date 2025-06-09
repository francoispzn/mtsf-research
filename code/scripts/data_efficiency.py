#!/usr/bin/env python3
"""Data-efficiency study: is the explicit delay prior more sample-efficient than attention?

Hypothesis: a structural phase-as-delay prior degrades more gracefully than a high-capacity
attention model as training data shrinks, because it needs far fewer examples to fit an
inter-series delay than to fit N-to-N attention. Epidemic forecasting is inherently low-data
(a few hundred weekly points), so this is the realistic deployment regime.

For each dataset x model x train_frac we retrain from scratch on the most-recent train_frac of
the training split (val/test fixed) and evaluate test MSE, at the phase-essential horizon.

Datasets: ili_us_state (phase regime, H=8), metr_la (mixing regime, H=12) as a control.
Models: DLinear, iTransformer, CSLL2 (DLinear+phase), CSLL2X (iTransformer+phase).
Writes results/tables/data_efficiency.json.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "scripts"))
from run_experiment import run   # noqa: E402

TAB = ROOT / "results" / "tables"
FRACS = [0.15, 0.25, 0.4, 0.6, 1.0]
SEEDS = [0, 1, 2]
BUDGET = dict(epochs=30, patience=5)
CFG = {"DLinear": {}, "iTransformer": {}, "CSLL2": {}, "CSLL2X": {}}
SETTINGS = [("ili_us_state", 36, 8), ("ili_us_state", 36, 4), ("metr_la", 96, 12)]


if __name__ == "__main__":
    out = {}
    cache = TAB / "data_efficiency.json"
    if cache.exists():
        out = json.load(open(cache))
    for ds, L, H in SETTINGS:
        for model in CFG:
            for fr in FRACS:
                for seed in SEEDS:
                    key = f"{ds}|{model}|H{H}|f{fr}|s{seed}"
                    if key in out:
                        continue
                    try:
                        r = run(model, ds, H, L, seed, cfg_over=(CFG[model] or None),
                                train_over=dict(BUDGET), train_frac=fr,
                                tag=f"deff_{ds}__{model}__H{H}__f{int(fr*100)}__s{seed}")
                        out[key] = r["metrics"]["mse"]
                        print(f"[done] {key}: {out[key]:.4f}", flush=True)
                        json.dump(out, open(cache, "w"), indent=2)
                    except Exception:
                        print(f"[FAIL] {key}\n{traceback.format_exc()}", flush=True)
    print("DATA_EFFICIENCY_DONE", flush=True)
