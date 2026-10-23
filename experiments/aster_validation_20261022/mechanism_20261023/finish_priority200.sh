#!/usr/bin/env bash
set -euo pipefail
source /home/gcx/gcx261014/bin/activate
cd /home/gcx/TRIAGE-worktrees/aster-validation
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1
r=/home/gcx/aster_validation_20261022/private/run500
p="$r/mechanism/priority200"
e=experiments/aster_validation_20261022/mechanism_20261023
o="$e/results/priority200/qwen"
trap 'printf "failed at line %s\n" "$LINENO" > "$p/qwen.status"' ERR
printf 'waiting_for_CAD_then_GPU0_GPU1_shards\n' > "$p/qwen.status"
for pid in "$@"; do tail --pid="$pid" -f /dev/null; done
test "$(cat "$p/gpu0/status")" = complete
test "$(cat "$p/gpu1/status")" = complete
printf 'merge_and_paired_statistics_200\n' > "$p/qwen.status"
python "$e/merge_priority200.py"
python "$e/analyze.py" --input "$p/qwen_branches.jsonl" --dataset "$p/test.jsonl" --out "$o/branches"
python "$e/analyze.py" --input "$p/qwen_complete.jsonl" --dataset "$p/test.jsonl" --out "$o/full"
python "$e/cad_audit.py" --root "$r" --model qwen --out "$o/cad_finished_audit_500.json"
printf 'conditional_attention_gate\n' > "$p/qwen.status"
python "$e/attention.py" --summary "$o/full/summary.json" --dev "$p/dev.jsonl" --test "$p/test.jsonl" --model /home/gcx/TRIAGE/models/modelscope/Qwen3-8B --out "$p/qwen_attention"
printf 'complete\n' > "$p/qwen.status"
