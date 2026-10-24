#!/usr/bin/env bash
set -euo pipefail
cd /home/gcx/TRIAGE-worktrees/cs44-threshold-order
source /home/gcx/gcx261014/bin/activate
P=/home/gcx/cs44_threshold_order_20261024/private
E=experiments/cs44_threshold_order_20261024
exec 9>"$P/run.lock"
flock -n 9 || exit 1
CUDA_VISIBLE_DEVICES=0 python "$E/experiment.py" worker --shard 0 > "$P/gpu0.log" 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 python "$E/experiment.py" worker --shard 1 > "$P/gpu1.log" 2>&1 &
P1=$!
echo "$P0 $P1" > "$P/worker_pids.txt"
S0=0; S1=0
wait "$P0" || S0=$?
wait "$P1" || S1=$?
if ((S0!=0 || S1!=0)); then exit 1; fi
python "$E/experiment.py" analyze > "$P/analysis.log" 2>&1
python "$E/report.py" > "$P/report.log" 2>&1
