#!/usr/bin/env bash
set -euo pipefail
cd /home/gcx/TRIAGE-worktrees/aster-validation
source /home/gcx/gcx261014/bin/activate
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRIAGE_JOB_OWNER=qwen_parallel_gpu0
r=/home/gcx/aster_validation_20261022/private/run500
e=experiments/aster_validation_20261022
m=/home/gcx/TRIAGE/models/modelscope/Qwen3-8B
s="$r/parallel_qwen/gpu0.status"
trap 'printf "failed at line %s\n" "$LINENO" > "$s"' ERR
printf 'original_filtered_CAD\n' > "$s"
python "$e/cad_filtered.py" --model "$m" --input "$r/all.jsonl" --base "$r/qwen_full.jsonl" --output "$r/cad_fix/qwen_original_filtered.jsonl"
printf 'right_reorder_500\n' > "$s"
python "$e/run_cad128.py" --phase main --model "$m" --input "$r/option_order/right.jsonl" --output "$r/option_order/qwen_right.jsonl" --gain-file "$e/results/qwen/diagnostic/gain.json"
printf 'right_unsigned_500\n' > "$s"
python "$e/supplement_unsigned.py" --model "$m" --input "$r/option_order/right.jsonl" --base "$r/option_order/qwen_right.jsonl" --output "$r/option_order/qwen_right_unsigned.jsonl" --gain-file "$e/results/qwen/diagnostic/gain.json"
printf 'right_filtered_CAD\n' > "$s"
python "$e/cad_filtered.py" --model "$m" --input "$r/option_order/right.jsonl" --base "$r/option_order/qwen_right.jsonl" --output "$r/cad_fix/qwen_right_filtered.jsonl"
printf 'complete\n' > "$s"
