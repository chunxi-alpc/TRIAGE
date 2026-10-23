#!/usr/bin/env bash
set -euo pipefail
name=$1
model=$2
worker=$3
root=/home/gcx/aster_validation_20261022/private
cd /home/gcx/TRIAGE-worktrees/aster-validation
source /home/gcx/gcx261014/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
trap 'printf "CAD repair failed at line %s\n" "$LINENO" > "$root/$name.finish_status"' ERR
printf 'waiting_for_main_then_cad128\n' > "$root/$name.finish_status"
tail --pid="$worker" -f /dev/null
printf 'cad128_rerun\n' > "$root/$name.finish_status"
python experiments/aster_validation_20261022/repair_cad.py --model "$model" --base "$root/${name}_full.jsonl" --input "$root/all.jsonl"
bash experiments/aster_validation_20261022/finish_model.sh "$name" "$model" "$worker" 73696
