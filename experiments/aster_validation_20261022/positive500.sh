#!/usr/bin/env bash
set -euo pipefail
model_name=$1
model_path=$2
run_root=/home/gcx/aster_validation_20261022/private/run500
repo=/home/gcx/TRIAGE-worktrees/aster-validation
exp=experiments/aster_validation_20261022
seka=/home/gcx/TRIAGE/third_party/SEKA
status="$run_root/${model_name}.positive_status"
trap 'printf "failed at line %s\n" "$LINENO" > "$status"' ERR
cd "$repo"
printf 'waiting_for_positive_source\n' > "$status"
true
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
printf 'positive_counterfact_efficacy_500\n' > "$status"
for condition in baseline seka; do
 args=()
 if [[ "$condition" == seka ]]; then args=(--seka --pos "${projections[0]}" --amplify_pos 1.56 --amplify_neg 0.0 --layers last10); fi
 if ! PYTHONPATH="$seka" python benchmarks/eval_fact_gen.py --model "$model_path" --data_path "$run_root/counterfact_official" --output_dir "$positive_dir/$condition" --benchmarks efficacy --example_subset 5000:5500 --batch_size 1 --max_length 300 --max_new_tokens 1 "${args[@]}" > "$positive_dir/${condition}.log" 2>&1; then
  printf 'main_complete; positive_blocked: official evaluation failed\n' > "$status"
  exit 0
 fi
done
printf 'complete\n' > "$status"
