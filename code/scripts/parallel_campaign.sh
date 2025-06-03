#!/usr/bin/env bash
# Parallel campaign: N sharded workers per stage (job i -> worker i % N), ~8 cores total,
# leaving ~2 free. Resumable (--skip-existing); each job writes its own result file, and
# shards are disjoint so there is no double-work or race.
set -uo pipefail
cd /Users/fpetizon/mtsf-research
export MTSF_DEVICE=cpu OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=3
NW=3
LOG=results/logs/campaign.log
say() { echo "$(date '+%H:%M:%S') $*" | tee -a "$LOG"; }

run_sharded() {              # $1 = description; rest = run_all.py args (no --shard/--nshards)
  local desc="$1"; shift
  say "--- $desc  (${NW} workers) ---"
  local pids=()
  for i in $(seq 0 $((NW-1))); do
    python3 -u code/scripts/run_all.py "$@" --shard "$i" --nshards "$NW" --skip-existing \
      >> "results/logs/worker_${i}.log" 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}"
  say "    done: $desc"
}

say "=== PARALLEL CAMPAIGN START (NW=${NW}, OMP=3) ==="

run_sharded "STAGE 1 tractable main" --which main \
  --datasets ETTh1 ETTh2 ETTm1 ETTm2 weather exchange ili synthetic --seeds 0 1 2

run_sharded "STAGE 2 heavy main" --which main \
  --datasets electricity traffic \
  --models SeasonalNaive DLinear NLinear CSLL iTransformer PatchTST \
  --horizons 96 336 --epochs 6

run_sharded "STAGE 3 ablations" --which ablation \
  --datasets ETTh1 ETTm1 weather exchange synthetic --seeds 0 --horizons 96 336

say "--- STAGE 4 heavy ablations ---"
python3 -u code/scripts/heavy_ablations.py >> "$LOG" 2>&1

say "=== PARALLEL CAMPAIGN DONE ==="
