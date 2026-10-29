#!/usr/bin/env bash
set -euo pipefail
cd /home/gcx/TRIAGE-worktrees/aster-fullscale
source /home/gcx/gcx261014/bin/activate
run="${1:?pass prepared run directory}"
script=experiments/aster_fullscale_20261029/scripts/parallel_directions.py
while [ ! -f "$run/direction_catalog.json" ]; do sleep 2; done
cards=(0 1 4 5 6)
pids=()
for rank in "${!cards[@]}"; do
 CUDA_VISIBLE_DEVICES="${cards[$rank]}" python "$script" worker --run "$run" --rank "$rank" --world 5 >"$run/fit_gpu${cards[$rank]}.log" 2>&1 &
 pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
python "$script" merge --run "$run" --world 5
pipeline=experiments/aster_fullscale_20261029/scripts/fullscale_run.py
CUDA_VISIBLE_DEVICES=0 python "$pipeline" full --run "$run" --output "$run/full_gpu0" --worker 0 --artifact "$run/parallel_fit/directions.pt" >"$run/gpu0.log" 2>&1 &
worker0=$!
CUDA_VISIBLE_DEVICES=1 python "$pipeline" full --run "$run" --output "$run/full_gpu1" --worker 1 >"$run/gpu1.log" 2>&1 &
worker1=$!
wait "$worker0"
wait "$worker1"
