#!/usr/bin/env bash
set -euo pipefail
source /home/gcx/gcx261014/bin/activate
cd /home/gcx/TRIAGE
export CUDA_VISIBLE_DEVICES=0
exec python src/module1/trialgpt_matching.py "$@"
