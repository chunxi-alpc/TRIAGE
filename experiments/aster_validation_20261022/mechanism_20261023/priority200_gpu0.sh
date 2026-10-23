#!/usr/bin/env bash
set -euo pipefail
source /home/gcx/gcx261014/bin/activate
cd /home/gcx/TRIAGE-worktrees/aster-validation
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0
r=/home/gcx/aster_validation_20261022/private/run500
p="$r/mechanism/priority200"
e=experiments/aster_validation_20261022/mechanism_20261023
o="$e/results/priority200/qwen"
trap 'printf "failed at line %s\n" "$LINENO" > "$p/qwen.status"' ERR
printf 'GPU0_direct_200; GPU1_existing_Qwen_CAD_unchanged\n' > "$p/execution_mode"
printf 'branch_diagnosis_200_GPU0\n' > "$p/qwen.status"
python "$e/run.py" --phase branch --input "$p/test.jsonl" --model /home/gcx/TRIAGE/models/modelscope/Qwen3-8B --base "$r/qwen_full.jsonl" --gain-file experiments/aster_validation_20261022/results/qwen/diagnostic/gain.json --output "$p/qwen_branches.jsonl"
python "$e/analyze.py" --input "$p/qwen_branches.jsonl" --dataset "$p/test.jsonl" --out "$o/branches"
python "$e/collect_priority200.py"
printf 'paired_actions_200_GPU0\n' > "$p/qwen.status"
python "$e/run.py" --phase action --branches "$p/qwen_branches.jsonl" --input "$p/test.jsonl" --model /home/gcx/TRIAGE/models/modelscope/Qwen3-8B --base "$r/qwen_full.jsonl" --gain-file experiments/aster_validation_20261022/results/qwen/diagnostic/gain.json --output "$p/qwen_complete.jsonl"
python "$e/analyze.py" --input "$p/qwen_complete.jsonl" --dataset "$p/test.jsonl" --out "$o/full"
printf 'conditional_attention_gate_GPU0\n' > "$p/qwen.status"
python "$e/attention.py" --summary "$o/full/summary.json" --dev "$p/dev.jsonl" --test "$p/test.jsonl" --model /home/gcx/TRIAGE/models/modelscope/Qwen3-8B --out "$p/qwen_attention"
# CAD audit remains read-only and only runs if the protected modified-CAD finished naturally.
if [[ "$(cat "$r/qwen.cad_fix_status")" == complete ]]; then
 python "$e/cad_audit.py" --root "$r" --model qwen --out "$o/cad_finished_audit_500.json"
fi
printf 'complete\n' > "$p/qwen.status"
