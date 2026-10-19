#!/usr/bin/env bash
set -u
cd /home/gcx/TRIAGE
source /home/gcx/gcx261014/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
out=results/acre_pilot/oracle_switch_diagnostic
date -u +%FT%TZ > "$out/logs/full_start.txt"
gpus=(0 1 2 3 6 7)
pids=()
for worker in 0 1 2 3 4 5; do
    CUDA_VISIBLE_DEVICES="${gpus[$worker]}" python scripts/run_acre_oracle_factorial.py --worker "$worker" --workers 6 > "$out/logs/worker_$worker.log" 2>&1 &
    pids+=("$!")
done
printf '%s\n' "${pids[@]}" > "$out/logs/worker_pids.txt"
failure=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failure=1; fi
done
date -u +%FT%TZ > "$out/logs/full_end.txt"
printf '%s\n' "$failure" > "$out/logs/exit_status.txt"
exit "$failure"
