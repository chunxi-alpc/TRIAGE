#!/usr/bin/env bash
set -euo pipefail
source /home/gcx/gcx261014/bin/activate
export CUDA_VISIBLE_DEVICES=0,1
triage_root=/home/gcx/TRIAGE-worktrees
triage_code=$triage_root/applicability-decision-20261029-code
triage_repo=$triage_root/applicability-decision-20261029
triage_run=$triage_root/applicability-decision-20261029-run02
triage_assets=$triage_root/applicability-decision-20261029-assets
mkdir -p "$triage_run/logs"
triage_phase=waiting_for_weights
trap 'triage_exit=$?; printf "{\"status\":\"exited\",\"phase\":\"%s\",\"exit_code\":%s}\n" "$triage_phase" "$triage_exit" > "$triage_run/pipeline_exit.json"' EXIT
printf '%s\n' "$$" > "$triage_run/supervisor.pid"
for triage_attempt in $(seq 1 180); do
  if [ -f "$triage_assets/model/model.safetensors" ]; then break; fi
  if ! kill -0 683659 2>/dev/null; then
    printf '%s\n' 'Weight downloader ended before producing the complete model.'
    exit 1
  fi
  sleep 10
done
test -f "$triage_assets/model/model.safetensors"
triage_phase=prepare
python "$triage_code/run_stages.py" prepare --repo "$triage_repo" --run "$triage_run" --model "$triage_assets/model" --trees "$triage_assets/data/train_val/decision_trees.jsonl" > "$triage_run/logs/prepare.log" 2>&1
triage_phase=smoke
python "$triage_code/smoke_on_server.py" > "$triage_run/logs/smoke.log" 2>&1
for triage_stage in audit localize train select confirm report; do
  triage_phase=$triage_stage
  triage_gpu=0
  if [ "$triage_stage" = localize ] || [ "$triage_stage" = confirm ]; then triage_gpu=1; fi
  printf '%s\n' "$triage_stage" > "$triage_run/current_stage.txt"
  python "$triage_code/run_stages.py" "$triage_stage" --repo "$triage_repo" --run "$triage_run" --gpu "$triage_gpu" > "$triage_run/logs/$triage_stage.log" 2>&1
done
triage_phase=completed
