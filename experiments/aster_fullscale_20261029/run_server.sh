#!/usr/bin/env bash
set -euo pipefail
cd /home/gcx/TRIAGE-worktrees/aster-fullscale
source /home/gcx/gcx261014/bin/activate
run="${1:?pass a fresh run directory}"
script=experiments/aster_fullscale_20261029/scripts/fullscale_run.py
python "$script" prepare --run "$run"
CUDA_VISIBLE_DEVICES=0 python "$script" smoke --run "$run" --output "$run/smoke"
bash experiments/aster_fullscale_20261029/run_parallel_server.sh "$run"
