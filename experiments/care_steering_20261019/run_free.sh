#!/usr/bin/env bash
set -euo pipefail
cd /home/gcx/care_steering_20261019
source /home/gcx/gcx261014/bin/activate
export CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
python -u src/run_free.py > logs/free_generation.log 2>&1
python src/analyze.py --predictions results/free_predictions.jsonl --data data/cases.jsonl --output results/free_analysis.json --markdown results/free_analysis.md > logs/free_analysis.log 2>&1
