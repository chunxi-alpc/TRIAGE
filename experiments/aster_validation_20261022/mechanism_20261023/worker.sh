#!/usr/bin/env bash
set -euo pipefail
name=$1
model=$2
cad_pid=$3
cd /home/gcx/TRIAGE-worktrees/aster-validation
source /home/gcx/gcx261014/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
root=/home/gcx/aster_validation_20261022/private/run500
private="$root/mechanism"
exp=experiments/aster_validation_20261022
mech="$exp/mechanism_20261023"
public="$mech/results/$name"
trap 'printf "failed at line %s\n" "$LINENO" > "$private/$name.status"' ERR
printf 'waiting_for_modified_CAD_natural_completion\n' > "$private/$name.status"
tail --pid="$cad_pid" -f /dev/null
test "$(cat "$root/$name.cad_fix_status")" = complete
python "$mech/cad_audit.py" --root "$root" --model "$name" --out "$public/cad_finished_audit.json"
printf 'branch_diagnosis_500\n' > "$private/$name.status"
python "$mech/run.py" --phase branch --input "$private/test.jsonl" --model "$model" --base "$root/${name}_full.jsonl" --gain-file "$exp/results/$name/diagnostic/gain.json" --output "$private/${name}_branches.jsonl"
python "$mech/analyze.py" --input "$private/${name}_branches.jsonl" --dataset "$private/test.jsonl" --out "$public/branches"
printf 'paired_actions_500\n' > "$private/$name.status"
python "$mech/run.py" --phase action --branches "$private/${name}_branches.jsonl" --input "$private/test.jsonl" --model "$model" --base "$root/${name}_full.jsonl" --gain-file "$exp/results/$name/diagnostic/gain.json" --output "$private/${name}_complete.jsonl"
python "$mech/analyze.py" --input "$private/${name}_complete.jsonl" --dataset "$private/test.jsonl" --out "$public/full"
printf 'conditional_attention_gate\n' > "$private/$name.status"
python "$mech/attention.py" --summary "$public/full/summary.json" --dev "$private/dev.jsonl" --test "$private/test.jsonl" --model "$model" --out "$private/${name}_attention"
printf 'complete\n' > "$private/$name.status"
