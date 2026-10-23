#!/usr/bin/env bash
set -euo pipefail
model_name=$1
model_path=$2
worker_pid=$3
download_pid=$4
run_root=/home/gcx/aster_validation_20261022/private
repo=/home/gcx/TRIAGE-worktrees/aster-validation
exp=experiments/aster_validation_20261022
seka=/home/gcx/TRIAGE/third_party/SEKA
source /home/gcx/gcx261014/bin/activate
cd "$repo"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
status="$run_root/${model_name}.finish_status"
trap 'printf "failed at line %s\n" "$LINENO" > "$status"' ERR
printf 'waiting_for_main\n' > "$status"
tail --pid="$worker_pid" -f /dev/null
if [[ ! -f "$run_root/${model_name}_full.manifest.json" ]]; then
 printf 'blocked: main inference did not complete\n' > "$status"
 exit 1
fi
printf 'unsigned_ablation\n' > "$status"
python "$exp/supplement_unsigned.py" --input "$run_root/all.jsonl" --model "$model_path" --gain-file "$exp/results/${model_name}/diagnostic/gain.json" --base "$run_root/${model_name}_full.jsonl" --output "$run_root/${model_name}_unsigned.jsonl"
python "$exp/merge_unsigned.py" "$run_root/${model_name}_full.jsonl" "$run_root/${model_name}_unsigned.jsonl" "$run_root/${model_name}_complete.jsonl"
printf 'final_paired_analysis\n' > "$status"
python "$exp/analyze.py" --input "$run_root/${model_name}_complete.jsonl" --dataset-manifest "$run_root/dataset_manifest.json" --output "$exp/results/${model_name}/full"
python "$exp/plot.py" "$exp/results/${model_name}/full/summary.json"
printf 'waiting_for_positive_source\n' > "$status"
tail --pid="$download_pid" -f /dev/null
# Formatting is shared and must not be written by two processes simultaneously.
(
 flock -x 9
 if [[ ! -f "$run_root/positive_preparation.json" ]]; then
  python "$exp/prepare_positive.py" "$run_root"
 fi
) 9> "$run_root/positive_preparation.lock" || {
 printf 'main_complete; positive_blocked: see preparation log\n' > "$status"
 exit 0
}
printf 'positive_projection_build\n' > "$status"
positive_dir="$run_root/positive/${model_name}"
mkdir -p "$positive_dir"
cd "$seka"
if ! PYTHONPATH="$seka" python src/custom_builders/synthetic_qa_builder.py --model "$model_path" --data data/synthetic/pair_qa_new.jsonl --output_dir "$positive_dir/projections" --max_samples 200 --min_diff 0.20 --top_pct 0.90 --layers last10 > "$positive_dir/build.log" 2>&1; then
 printf 'main_complete; positive_blocked: projection build failed\n' > "$status"
 exit 0
fi
mapfile -t projections < <(find "$positive_dir/projections" -name '*pos_proj.pt' -type f)
if [[ ${#projections[@]} != 1 ]]; then
 printf 'main_complete; positive_blocked: projection artifact not unique\n' > "$status"
 exit 0
fi
printf 'positive_counterfact_efficacy_5000\n' > "$status"
for condition in baseline seka; do
 args=()
 if [[ "$condition" == seka ]]; then args=(--seka --pos "${projections[0]}" --amplify_pos 1.56 --amplify_neg 0.0 --layers last10); fi
 if ! PYTHONPATH="$seka" python benchmarks/eval_fact_gen.py --model "$model_path" --data_path "$run_root/counterfact_official" --output_dir "$positive_dir/$condition" --benchmarks efficacy --example_subset 5000:10000 --batch_size 1 --max_length 300 --max_new_tokens 1 "${args[@]}" > "$positive_dir/${condition}.log" 2>&1; then
  printf 'main_complete; positive_blocked: official evaluation failed\n' > "$status"
  exit 0
 fi
done
printf 'complete\n' > "$status"
