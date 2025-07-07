#!/usr/bin/env bash
set -uo pipefail
cd /Users/fpetizon/mtsf-research
export MTSF_DEVICE=cpu OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=3 MTSF_TORCH_THREADS=3
NW=3
pids=()
for i in $(seq 0 $((NW-1))); do
  python3 -u code/scripts/rework.py --shard "$i" --nshards "$NW" >> "results/logs/rework_${i}.log" 2>&1 &
  pids+=($!)
done
wait "${pids[@]}"
echo "PARALLEL_REWORK_DONE"
