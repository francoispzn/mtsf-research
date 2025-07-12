#!/usr/bin/env python3
"""CSLL2H (hybrid: gated direct-phase + gated real-mixing) multi-seed runs on the settings
that decide the adaptive-model story: epidemic (phase regime), weather/ETTh1 (mixing regime
where pure phase regresses), and traffic (mixing regime). Fast settings first.

The claim under test: ONE model (CSLL2H), without being told the regime, stays within noise of
the best specialised CSLL2 variant in each regime -- i.e. it never suffers the -17% weather
regression of pure direct-phase, and keeps the +15% epidemic gain.
Writes standard results/raw entries (skip-existing).
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "scripts"))
from run_experiment import run                       # noqa: E402
from csll.config import seq_len_for                  # noqa: E402

B30 = dict(epochs=30, patience=5)
# (dataset, horizons, seeds)  -- fast (epi + LTSF) first, heavy traffic last
PLAN = [
    ("ili_us_state", [2, 4, 8], [0, 1, 2]),
    ("ili_japan", [2, 4, 8], [0, 1, 2]),
    ("ili_us_hhs", [2, 4, 8], [0, 1, 2]),
    ("weather", [96, 192, 336, 720], [0, 1]),
    ("ETTh1", [96, 336], [0, 1, 2]),
    ("exchange", [96, 336], [0, 1, 2]),
    ("metr_la", [12, 24, 48, 96], [0, 1, 2]),
    ("pems_bay", [12, 24, 48, 96], [0]),
    ("pems04", [12, 24, 48, 96], [0]),
    ("pems08", [12, 24, 48, 96], [0]),
]

if __name__ == "__main__":
    ok = fail = skip = 0
    for ds, Hs, seeds in PLAN:
        L = seq_len_for(ds)
        for H in Hs:
            for s in seeds:
                tag = f"{ds}__CSLL2H__L{L}__H{H}__s{s}"
                if (ROOT / "results" / "raw" / f"{tag}.json").exists():
                    skip += 1
                    continue
                try:
                    r = run("CSLL2H", ds, H, L, s, train_over=dict(B30), tag=tag,
                            save_model=(s == 0))
                    print(f"[done] {tag}: mse={r['metrics']['mse']:.4f} "
                          f"params={r['history'].get('params')} wall={r['wall_time_s']:.0f}s", flush=True)
                    ok += 1
                except Exception:
                    print(f"[FAIL] {tag}\n{traceback.format_exc()}", flush=True)
                    fail += 1
    print(f"HYBRID_CAMPAIGN_DONE ok={ok} skip={skip} fail={fail}", flush=True)
