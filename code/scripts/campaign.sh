#!/usr/bin/env bash
# Full experiment campaign (resumable via --skip-existing). Each job writes its own JSON,
# so an interrupted run can be relaunched and will continue where it left off.
set -uo pipefail
cd /Users/fpetizon/mtsf-research
export MTSF_DEVICE=cpu OMP_NUM_THREADS=8
LOG=results/logs/campaign.log
say() { echo "$(date '+%H:%M:%S') $*" | tee -a "$LOG"; }

say "=== CAMPAIGN START ==="

say "--- STAGE 1: tractable main (8 datasets, 3 seeds; 2 on ETTm1/m2/weather) ---"
python3 -u code/scripts/run_all.py --which main \
  --datasets ETTh1 ETTh2 ETTm1 ETTm2 weather exchange ili synthetic \
  --seeds 0 1 2 --skip-existing 2>&1 | tee -a "$LOG"

say "--- STAGE 2: heavy main (electricity, traffic; 1 seed, H in {96,336}, epochs 6) ---"
python3 -u code/scripts/run_all.py --which main \
  --datasets electricity traffic \
  --models SeasonalNaive DLinear NLinear CSLL iTransformer PatchTST \
  --horizons 96 336 --epochs 6 --skip-existing 2>&1 | tee -a "$LOG"

say "--- STAGE 3: ablations (seed 0, H in {96,336}) ---"
python3 -u code/scripts/run_all.py --which ablation \
  --datasets ETTh1 ETTm1 weather exchange synthetic \
  --seeds 0 --horizons 96 336 --skip-existing 2>&1 | tee -a "$LOG"

say "--- STAGE 4: heavy ablations (backbone/static vs dynamic on electricity/traffic) ---"
python3 -u code/scripts/heavy_ablations.py 2>&1 | tee -a "$LOG"

say "=== CAMPAIGN DONE ==="
