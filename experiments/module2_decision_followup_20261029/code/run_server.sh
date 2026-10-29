#!/usr/bin/env bash
# Prepared, NOT yet executed. Run only on gpu-a100-01 as gcx.
set -euo pipefail
source /home/gcx/gcx261014/bin/activate
export CUDA_VISIBLE_DEVICES=0,1
triage_code_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
triage_repo=${1:?Pass isolated repository at the requested commit}
triage_run=${2:?Pass a new server output directory}
mkdir -p -- "$triage_run/logs"
printf '%s\n' 'Planned: GPUs 0/1, FP32 Qwen3-0.6B-Base; 6-16 GiB per active GPU; 48 new guidelines; about 1-3 hours, unmeasured estimate.'
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv > "$triage_run/logs/gpu_before.csv"
python "$triage_code_dir/run_stages.py" prepare --repo "$triage_repo" --run "$triage_run" > "$triage_run/logs/prepare.log" 2>&1
python "$triage_code_dir/run_stages.py" audit --repo "$triage_repo" --run "$triage_run" --gpu 0 > "$triage_run/logs/audit.log" 2>&1
python "$triage_code_dir/run_stages.py" localize --repo "$triage_repo" --run "$triage_run" --gpu 1 > "$triage_run/logs/localize.log" 2>&1
python "$triage_code_dir/run_stages.py" train --repo "$triage_repo" --run "$triage_run" --gpu 0 > "$triage_run/logs/train.log" 2>&1
python "$triage_code_dir/run_stages.py" select --repo "$triage_repo" --run "$triage_run" --gpu 0 > "$triage_run/logs/select.log" 2>&1
python "$triage_code_dir/run_stages.py" confirm --repo "$triage_repo" --run "$triage_run" --gpu 1 > "$triage_run/logs/confirm.log" 2>&1
python "$triage_code_dir/run_stages.py" report --repo "$triage_repo" --run "$triage_run" > "$triage_run/logs/report.log" 2>&1
