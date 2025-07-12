#!/usr/bin/env python3
"""Clean, DECLARED, bounded completion of the traffic evidence.

Rationale (stated for the record, decided a priori -- not outcome-driven): the two canonical
traffic-*speed* benchmarks METR-LA and PEMS-BAY carry the sensor road-distance graph that the
paper uses as delay ground truth; PEMS04/08 are *flow* datasets without it. We therefore treat
METR-LA + PEMS-BAY as the traffic evidence and scope the traffic claim to them.

Protocol:
  * METR-LA -- FULL (already complete: 3 seeds, H in {12,24,48,96}, full model suite). Primary.
  * PEMS-BAY -- DECLARED CONFIRMATION protocol: 1 seed, H in {12,96} (short + long), core model
    set [SeasonalNaive, NLinear, DLinear, iTransformer, CSLL2, CSLL2-phaseoff, CSLL2-backbone,
    CSLL2H]. Confirms (or refutes) the METR-LA phase-vs-mixing dichotomy on a second speed
    benchmark. Reduced seeds/horizons are disclosed; we do not claim per-setting significance
    from PEMS-BAY, only the qualitative regime.
  * PEMS04/PEMS08 -- optional additional (flow datasets), same reduced protocol, run only if
    reached; reported as breadth, never in place of the speed benchmarks.

Everything is sequential (memory-bound), resumable (skip-existing), uniform 30-epoch budget.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "scripts"))
from run_experiment import run                       # noqa: E402

B30 = dict(epochs=30, patience=5)
CORE = [
    ("SeasonalNaive", {}, None), ("NLinear", {}, None), ("DLinear", {}, None),
    ("iTransformer", {}, None), ("CSLL2", {}, None),
    ("CSLL2", dict(use_phase=False, direct=False), "phaseoff"),
    ("CSLL2", dict(freeze_gate=True, gate_init=0.0), "backbone"),
    ("CSLL2H", {}, None),
]
# (dataset, horizons, seeds)  -- priority order
PLAN = [
    ("pems_bay", [12, 96], [0]),
    ("pems04", [12, 96], [0]),
    ("pems08", [12, 96], [0]),
]

if __name__ == "__main__":
    ok = fail = skip = 0
    for ds, Hs, seeds in PLAN:
        for H in Hs:
            for s in seeds:
                for model, over, suffix in CORE:
                    name = model + (f"-{suffix}" if suffix else "")
                    tag = f"{ds}__{name}__L96__H{H}__s{s}"
                    if (ROOT / "results" / "raw" / f"{tag}.json").exists():
                        skip += 1
                        continue
                    try:
                        r = run(model, ds, H, 96, s, cfg_over=(over or None),
                                train_over=dict(B30), tag=tag,
                                save_model=(model in ("CSLL2", "CSLL2H") and suffix is None))
                        print(f"[done] {tag}: mse={r['metrics']['mse']:.4f} "
                              f"wall={r['wall_time_s']:.0f}s", flush=True)
                        ok += 1
                    except Exception:
                        print(f"[FAIL] {tag}\n{traceback.format_exc()}", flush=True)
                        fail += 1
    print(f"TRAFFIC_FINISH_DONE ok={ok} skip={skip} fail={fail}", flush=True)
