#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1
SEKA_ROOT=/home/gcx/TRIAGE/third_party/SEKA
PRIVATE=/home/gcx/aster_e0_e8_20261024/private
cd "$SEKA_ROOT"
PYTHONPATH="$SEKA_ROOT" python src/custom_builders/synthetic_qa_builder.py --model /home/gcx/TRIAGE/models/modelscope/Meta-Llama-3.1-8B-Instruct --data data/synthetic/pair_qa_new.jsonl --output_dir "$PRIVATE/green_llama" --max_samples 100 --min_diff 0.20 --top_pct 0.90 --layers last10
