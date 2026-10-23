#!/usr/bin/env bash
set -euo pipefail
source /home/gcx/gcx261014/bin/activate
cd /home/gcx/TRIAGE-worktrees/aster-validation
export OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
r=/home/gcx/aster_validation_20261022/private/run500/cad200
e=experiments/aster_validation_20261022
m=/home/gcx/TRIAGE/models/modelscope/Qwen3-8B
gpu=$1
export CUDA_VISIBLE_DEVICES="$gpu"
s="$r/gpu$gpu.status"
trap 'printf "failed at line %s\n" "$LINENO" > "$s"' ERR
if [[ "$gpu" == 0 ]]; then
 printf 'original_filtered_selected200_remaining_only\n' > "$s"
 python "$e/cad_filtered.py" --model "$m" --input "$r/all.jsonl" --base "$r/base.jsonl" --output "$r/original_filtered.jsonl"
 printf 'left_unsigned_selected200\n' > "$s"
 python "$e/supplement_unsigned.py" --model "$m" --input "$r/left.jsonl" --base "$r/left_main.jsonl" --output "$r/left_unsigned.jsonl" --gain-file "$e/results/qwen/diagnostic/gain.json"
 python "$e/merge_unsigned.py" "$r/left_main.jsonl" "$r/left_unsigned.jsonl" "$r/left_complete.jsonl"
 printf 'left_filtered_selected200\n' > "$s"
 python "$e/cad_filtered.py" --model "$m" --input "$r/left.jsonl" --base "$r/left_main.jsonl" --output "$r/left_filtered.jsonl"
else
 printf 'right_main_selected200\n' > "$s"
 python "$e/run_cad128.py" --phase main --model "$m" --input "$r/right.jsonl" --output "$r/right_main.jsonl" --gain-file "$e/results/qwen/diagnostic/gain.json"
 printf 'right_unsigned_selected200\n' > "$s"
 python "$e/supplement_unsigned.py" --model "$m" --input "$r/right.jsonl" --base "$r/right_main.jsonl" --output "$r/right_unsigned.jsonl" --gain-file "$e/results/qwen/diagnostic/gain.json"
 python "$e/merge_unsigned.py" "$r/right_main.jsonl" "$r/right_unsigned.jsonl" "$r/right_complete.jsonl"
 printf 'right_filtered_selected200\n' > "$s"
 python "$e/cad_filtered.py" --model "$m" --input "$r/right.jsonl" --base "$r/right_main.jsonl" --output "$r/right_filtered.jsonl"
fi
printf 'complete\n' > "$s"
