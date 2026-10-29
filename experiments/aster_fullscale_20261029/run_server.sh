#!/usr/bin/env bash
set -euo pipefail
cd /home/gcx/TRIAGE-worktrees/aster-fullscale
source /home/gcx/gcx261014/bin/activate
run="${1:?pass a fresh run directory}"
script=experiments/aster_fullscale_20261029/scripts/fullscale_run.py
python "$script" prepare --run "$run"
CUDA_VISIBLE_DEVICES=0 python "$script" smoke --run "$run" --output "$run/smoke"
CUDA_VISIBLE_DEVICES=0 python "$script" full --run "$run" --output "$run/full_gpu0" --worker 0 >"$run/gpu0.log" 2>&1 &
worker0=$!
CUDA_VISIBLE_DEVICES=1 python "$script" full --run "$run" --output "$run/full_gpu1" --worker 1 >"$run/gpu1.log" 2>&1 &
worker1=$!
wait "$worker0"
wait "$worker1"
