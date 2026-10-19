#!/usr/bin/env bash
set -euo pipefail
source /home/gcx/gcx261014/bin/activate
cd /home/gcx/care_steering_20261019
export PYTHONPATH="/home/gcx/care_steering_20261019/.deps:/home/gcx/care_steering_20261019${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES=6
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python src/run_c_full.py "$@"
