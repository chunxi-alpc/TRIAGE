#!/usr/bin/env bash
set -euo pipefail
model_name=$1
model_path=$2
run_root=/home/gcx/aster_validation_20261022/private
exp=experiments/aster_validation_20261022
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
status="$run_root/${model_name}.status"
trap 'printf "failed at line %s\n" "$LINENO" > "$status"' ERR
printf 'diagnostic_analysis\n' > "$status"
python "$exp/analyze.py" --input "$run_root/${model_name}_diag.jsonl" --dataset-manifest "$run_root/dataset_manifest.json" --output "$exp/results/${model_name}/diagnostic"
python "$exp/plot.py" "$exp/results/${model_name}/diagnostic/summary.json"
printf 'main_smoke\n' > "$status"
python "$exp/run.py" --input "$run_root/all.jsonl" --model "$model_path" --phase main --limit 1 --gain-file "$exp/results/${model_name}/diagnostic/gain.json" --output "$run_root/${model_name}_smoke.jsonl"
python "$exp/analyze.py" --input "$run_root/${model_name}_smoke.jsonl" --dataset-manifest "$run_root/dataset_manifest.json" --output "$run_root/${model_name}_smoke_analysis"
printf 'full_3645_running\n' > "$status"
python "$exp/run.py" --input "$run_root/all.jsonl" --model "$model_path" --phase main --gain-file "$exp/results/${model_name}/diagnostic/gain.json" --output "$run_root/${model_name}_full.jsonl"
printf 'full_analysis\n' > "$status"
python "$exp/analyze.py" --input "$run_root/${model_name}_full.jsonl" --dataset-manifest "$run_root/dataset_manifest.json" --output "$exp/results/${model_name}/full"
python "$exp/plot.py" "$exp/results/${model_name}/full/summary.json"
printf 'complete\n' > "$status"
