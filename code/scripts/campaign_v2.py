#!/usr/bin/env python3
"""v2 campaign: spatio-temporal traffic benchmark + LTSF-suite secondary results.

Phase 1 (primary, ST traffic; uniform 30-epoch/patience-5 budget for EVERY model):
  datasets : metr_la, pems_bay, pems04, pems08   (priority order)
  horizons : 12, 24, 48, 96                       (1h/2h/4h/8h at 5-min)
  models   : SeasonalNaive(1 seed), NLinear(3), DLinear(3),
             CSLL2(3), CSLL2-static(3), CSLL2-backbone(3), CSLL2-phaseoff(1),
             iTransformer(2), PatchTST(1, H in {12,96} only -- expensive)
Phase 2 (secondary, LTSF tractable suite under the ORIGINAL 10-epoch protocol so numbers
  are comparable with the existing v1 tables): CSLL2 + CSLL2-static, 3 seeds
  (2 on ETTm1/m2/weather, matching the v1 seed scheme).

Everything is sequential (memory-bound machine), resumable (skip-existing), and logs one
line per job. Checkpoints are saved for CSLL2/CSLL2-static seed 0 (delay validation).

Run:  MTSF_DEVICE=cpu MTSF_TORCH_THREADS=4 nice -n 5 .venv/bin/python3 code/scripts/campaign_v2.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "scripts"))

from run_experiment import run   # noqa: E402
from csll.config import seq_len_for  # noqa: E402

RAW = ROOT / "results" / "raw"
B30 = dict(epochs=30, patience=5)

ST_DATASETS = ["metr_la", "pems_bay", "pems04", "pems08"]
ST_HORIZONS = [12, 24, 48, 96]

# EPI (weekly regional ILI): the real-world tau >= h test. US-state has 3.7-wk mean
# inter-region lag (phase-essential at H=2); Japan marginal; US-HHS synchronised (inert
# control). Tiny (N<=49) so full model suite incl. iTransformer/PatchTST is cheap.
EPI_DATASETS = ["ili_us_state", "ili_japan", "ili_us_hhs"]
EPI_HORIZONS = [2, 4, 8]

LTSF = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "exchange", "ili"]
LTSF_SEEDS = {"ETTm1": 2, "ETTm2": 2, "weather": 2}          # match v1 seed scheme (else 3)


def _st_epi_block(ds, H, full_seeds=(0, 1, 2)):
    """One dataset x horizon: full model suite + all four CSLL2 variants."""
    fs = list(full_seeds)
    yield ds, H, "SeasonalNaive", {}, None, [0], B30
    yield ds, H, "NLinear", {}, None, fs, B30
    yield ds, H, "DLinear", {}, None, fs, B30
    yield ds, H, "CSLL2", {}, None, fs, B30
    yield ds, H, "CSLL2", dict(dynamic=False), "static", fs, B30
    yield ds, H, "CSLL2", dict(freeze_gate=True, gate_init=0.0), "backbone", fs, B30
    yield ds, H, "CSLL2", dict(use_phase=False, direct=False), "phaseoff", fs, B30
    yield ds, H, "iTransformer", {}, None, [0, 1], B30
    if ds in EPI_DATASETS:
        yield ds, H, "VAR", {}, None, [0], None        # tractable N: fair CD baseline
        yield ds, H, "PatchTST", {}, None, [0], B30
    elif H in (12, 96):
        yield ds, H, "PatchTST", {}, None, [0], B30


def jobs():
    # ---- Phase 1: EPI FIRST (the centerpiece phase-essential result; tiny + fast) ----
    for ds in EPI_DATASETS:
        for H in EPI_HORIZONS:
            yield from _st_epi_block(ds, H)
    # ---- Phase 2: LTSF suite (secondary "correctly inert" control; also fast) ----
    for ds in LTSF:
        for H in ([24, 36, 48, 60] if ds == "ili" else [96, 192, 336, 720]):
            seeds = list(range(LTSF_SEEDS.get(ds, 3)))
            yield ds, H, "CSLL2", {}, None, seeds, None
            yield ds, H, "CSLL2", dict(dynamic=False), "static", seeds, None
            yield ds, H, "CSLL2", dict(use_phase=False, direct=False), "phaseoff", seeds, None
    # ---- Phase 3: ST traffic (heavy; metr_la mostly cached, then pems) ----
    for ds in ST_DATASETS:
        for H in ST_HORIZONS:
            yield from _st_epi_block(ds, H)


if __name__ == "__main__":
    n_ok = n_fail = n_skip = 0
    for ds, H, model, over, suffix, seeds, budget in jobs():
        L = seq_len_for(ds)
        for seed in seeds:
            name = model + (f"-{suffix}" if suffix else "")
            tag = f"{ds}__{name}__L{L}__H{H}__s{seed}"
            if (RAW / f"{tag}.json").exists():
                n_skip += 1
                continue
            try:
                res = run(model, ds, H, L, seed, cfg_over=(over or None),
                          train_over=(dict(budget) if budget else None), tag=tag,
                          save_model=(model == "CSLL2" and seed == 0 and (suffix in (None, "static"))))
                m = res["metrics"]
                print(f"[done] {tag}: mse={m['mse']:.4f} mae={m['mae']:.4f} "
                      f"wall={res['wall_time_s']:.0f}s", flush=True)
                n_ok += 1
            except Exception:
                print(f"[FAIL] {tag}\n{traceback.format_exc()}", flush=True)
                n_fail += 1
    print(f"CAMPAIGN_V2_DONE ok={n_ok} skip={n_skip} fail={n_fail}", flush=True)
