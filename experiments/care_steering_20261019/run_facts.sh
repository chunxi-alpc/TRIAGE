#!/usr/bin/env bash
set -euo pipefail
cd /home/gcx/care_steering_20261019
source /home/gcx/gcx261014/bin/activate
export CUDA_VISIBLE_DEVICES=6 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
python -u src/run_facts.py > logs/facts.log 2>&1
