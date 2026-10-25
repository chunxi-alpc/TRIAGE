#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../.."
source /home/gcx/gcx261014/bin/activate
out=/home/gcx/aster_sparse_dynamic_qwen_seed0_20261025
pids=()
for gpu in 0 1 6; do
  CUDA_VISIBLE_DEVICES="$gpu" python experiments/aster_sparse_dynamic_qwen_seed0/run_gpu.py --shard "$gpu" > "$out/shard${gpu}.log" 2>&1 &
  pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
printf '%s\n' "$rc" > "$out/workers_exit_code.txt"
exit "$rc"
