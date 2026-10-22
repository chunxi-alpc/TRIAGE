#!/usr/bin/env bash
set -euo pipefail
name=$1
model=$2
root=/home/gcx/aster_validation_20261022/private/run500
exp=experiments/aster_validation_20261022
cd /home/gcx/TRIAGE-worktrees/aster-validation
source /home/gcx/gcx261014/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
trap 'printf "failed at line %s\n" "$LINENO" > "$root/$name.status"' ERR
printf 'cad128_500_cases\n' > "$root/$name.status"
python "$exp/repair_cad.py" --model "$model" --base "$root/${name}_full.jsonl" --input "$root/all.jsonl"
printf 'unsigned_500_cases\n' > "$root/$name.status"
python "$exp/supplement_unsigned.py" --model "$model" --input "$root/all.jsonl" --gain-file "$exp/results/$name/diagnostic/gain.json" --base "$root/${name}_full.jsonl" --output "$root/${name}_unsigned.jsonl"
python "$exp/merge_unsigned.py" "$root/${name}_full.jsonl" "$root/${name}_unsigned.jsonl" "$root/${name}_complete.jsonl"
python "$exp/analyze.py" --input "$root/${name}_complete.jsonl" --dataset-manifest "$root/dataset_manifest.json" --output "$exp/results/run500/$name/full"
python "$exp/plot.py" "$exp/results/run500/$name/full/summary.json"
for rotation in left right; do
 printf 'option_order_%s_500_cases\n' "$rotation" > "$root/$name.status"
 input="$root/option_order/$rotation.jsonl"
 output="$root/option_order/${name}_$rotation.jsonl"
 python "$exp/run_cad128.py" --phase main --model "$model" --input "$input" --output "$output" --gain-file "$exp/results/$name/diagnostic/gain.json"
 python "$exp/supplement_unsigned.py" --input "$input" --model "$model" --gain-file "$exp/results/$name/diagnostic/gain.json" --base "$output" --output "$root/option_order/${name}_${rotation}_unsigned.jsonl"
 python "$exp/merge_unsigned.py" "$output" "$root/option_order/${name}_${rotation}_unsigned.jsonl" "$root/option_order/${name}_${rotation}_complete.jsonl"
done
python "$exp/option_order.py" analyze --root "$root" --model "$name" --out "$exp/results/run500/$name/option_order"
printf 'positive_control_500\n' > "$root/$name.status"
bash "$exp/positive500.sh" "$name" "$model"
printf 'complete; check positive_status for separate positive-control outcome\n' > "$root/$name.status"
