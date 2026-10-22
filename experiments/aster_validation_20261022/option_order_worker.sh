#!/usr/bin/env bash
set -euo pipefail
name=$1
model=$2
previous=$3
root=/home/gcx/aster_validation_20261022/private
exp=experiments/aster_validation_20261022
cd /home/gcx/TRIAGE-worktrees/aster-validation
source /home/gcx/gcx261014/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
trap 'printf "failed at line %s\n" "$LINENO" > "$root/$name.option_order_status"' ERR
printf 'waiting_for_existing_queue\n' > "$root/$name.option_order_status"
tail --pid="$previous" -f /dev/null
test -f "$root/${name}_complete.manifest.json"
for rotation in left right; do
 printf 'running_%s\n' "$rotation" > "$root/$name.option_order_status"
 input="$root/option_order/$rotation.jsonl"
 output="$root/option_order/${name}_$rotation.jsonl"
 python "$exp/run.py" --phase main --model "$model" --input "$input" --output "$output" --gain-file "$exp/results/$name/diagnostic/gain.json"
 python "$exp/repair_cad.py" --model "$model" --base "$output" --input "$input"
 python "$exp/supplement_unsigned.py" --input "$input" --model "$model" --gain-file "$exp/results/$name/diagnostic/gain.json" --base "$output" --output "$root/option_order/${name}_${rotation}_unsigned.jsonl"
 python "$exp/merge_unsigned.py" "$output" "$root/option_order/${name}_${rotation}_unsigned.jsonl" "$root/option_order/${name}_${rotation}_complete.jsonl"
done
python "$exp/option_order.py" analyze --root "$root" --model "$name" --out "$exp/results/$name/option_order"
printf 'complete\n' > "$root/$name.option_order_status"
