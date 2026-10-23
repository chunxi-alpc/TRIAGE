#!/usr/bin/env bash
set -euo pipefail
source /home/gcx/gcx261014/bin/activate
cd /home/gcx/TRIAGE-worktrees/aster-validation
export CUDA_VISIBLE_DEVICES=0 TRIAGE_JOB_OWNER=qwen_selected200_gpu0 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
r=/home/gcx/aster_validation_20261022/private/run500/cad200
e=experiments/aster_validation_20261022
m=/home/gcx/TRIAGE/models/modelscope/Qwen3-8B
trap 'printf "failed at line %s\n" "$LINENO" > "$r/gpu0_right_pending.status"' ERR
printf 'right_unsigned_selected200\n' > "$r/gpu0_right_pending.status"
python "$e/supplement_unsigned.py" --input "$r/right.jsonl" --model "$m" --base "$r/right_input_metadata.jsonl" --output "$r/right_unsigned.jsonl" --gain-file "$e/results/qwen/diagnostic/gain.json"
printf 'right_filtered_selected200\n' > "$r/gpu0_right_pending.status"
python "$e/cad_filtered.py" --input "$r/right.jsonl" --model "$m" --base "$r/right_input_metadata.jsonl" --output "$r/right_filtered.jsonl"
printf 'complete\n' > "$r/gpu0_right_pending.status"
