#!/usr/bin/env bash
set -euo pipefail
source /home/gcx/gcx261014/bin/activate
cd /home/gcx/TRIAGE-worktrees/aster-validation
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
slot=$1
previous=$2
prior_model=$3
r=/home/gcx/aster_validation_20261022/private/run500
p="$r/mechanism/priority200"
s="$p/gpu$slot"
e=experiments/aster_validation_20261022
trap 'printf "failed at line %s\n" "$LINENO" > "$s/status"' ERR
printf 'waiting_for_existing_CAD_natural_completion\n' > "$s/status"
tail --pid="$previous" -f /dev/null
test "$(cat "$r/$prior_model.cad_fix_status")" = complete
printf 'branch_100\n' > "$s/status"
python "$e/mechanism_20261023/run.py" --phase branch --input "$s/test.jsonl" --model /home/gcx/TRIAGE/models/modelscope/Qwen3-8B --base "$r/qwen_full.jsonl" --gain-file "$e/results/qwen/diagnostic/gain.json" --output "$s/qwen_branches.jsonl"
printf 'action_100\n' > "$s/status"
python "$e/mechanism_20261023/run.py" --phase action --branches "$s/qwen_branches.jsonl" --input "$s/test.jsonl" --model /home/gcx/TRIAGE/models/modelscope/Qwen3-8B --base "$r/qwen_full.jsonl" --gain-file "$e/results/qwen/diagnostic/gain.json" --output "$s/qwen_complete.jsonl"
printf 'complete\n' > "$s/status"
