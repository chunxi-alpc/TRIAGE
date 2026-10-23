#!/usr/bin/env bash
set -euo pipefail
source /home/gcx/gcx261014/bin/activate
cd /home/gcx/TRIAGE-worktrees/aster-validation
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0
p=/home/gcx/aster_validation_20261022/private/run500/mechanism/priority200
e=experiments/aster_validation_20261022/mechanism_20261023
o="$e/results/priority200/qwen"
trap 'printf "failed parser repair at line %s\n" "$LINENO" > "$p/qwen.status"' ERR
printf 'repair_parser_v2_dependent_actions_GPU0\n' > "$p/qwen.status"
python "$e/run.py" --phase action --branches "$p/qwen_branches_parser_v2.jsonl" --reuse-actions "$p/qwen_complete_reparsed.jsonl" --input "$p/test.jsonl" --model /home/gcx/TRIAGE/models/modelscope/Qwen3-8B --base /home/gcx/aster_validation_20261022/private/run500/qwen_full.jsonl --gain-file experiments/aster_validation_20261022/results/qwen/diagnostic/gain.json --output "$p/qwen_complete_parser_v2.jsonl"
python "$e/analyze.py" --input "$p/qwen_complete_parser_v2.jsonl" --dataset "$p/test.jsonl" --out "$o/full_parser_v2"
python "$e/attention.py" --summary "$o/full_parser_v2/summary.json" --dev "$p/dev.jsonl" --test "$p/test.jsonl" --model /home/gcx/TRIAGE/models/modelscope/Qwen3-8B --out "$p/qwen_attention_parser_v2"
printf 'complete_parser_v2\n' > "$p/qwen.status"
python "$e/collect_priority200.py"
