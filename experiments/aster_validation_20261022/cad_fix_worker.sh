#!/usr/bin/env bash
set -euo pipefail
name=$1
model=$2
previous=$3
root=/home/gcx/aster_validation_20261022/private/run500
exp=experiments/aster_validation_20261022
cd /home/gcx/TRIAGE-worktrees/aster-validation
source /home/gcx/gcx261014/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
trap 'printf "failed at line %s\n" "$LINENO" > "$root/$name.cad_fix_status"' ERR
printf 'waiting_for_existing_experiments\n' > "$root/$name.cad_fix_status"
tail --pid="$previous" -f /dev/null
mkdir -p "$root/cad_fix"
for order in original left right; do
 printf 'filtered_cad_%s\n' "$order" > "$root/$name.cad_fix_status"
 if [[ "$order" == original ]]; then
  input="$root/all.jsonl"
  base="$root/${name}_full.jsonl"
 else
  input="$root/option_order/$order.jsonl"
  base="$root/option_order/${name}_$order.jsonl"
 fi
 python "$exp/cad_filtered.py" --model "$model" --input "$input" --base "$base" --output "$root/cad_fix/${name}_${order}_filtered.jsonl"
 python "$exp/cad_choice_control.py" --input "$base" --output "$exp/results/run500/$name/cad_choice_$order"
done
python "$exp/analyze_cad_fix.py" --root "$root" --model "$name" --out "$exp/results/run500/$name/cad_fix"
printf 'complete\n' > "$root/$name.cad_fix_status"
