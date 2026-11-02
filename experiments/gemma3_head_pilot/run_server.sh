#!/usr/bin/env bash
set -euo pipefail
cd /home/gcx/TRIAGE-worktrees/gemma3-27b-head-pilot-20261103
source /home/gcx/gcx261014/bin/activate
export CUDA_VISIBLE_DEVICES=0,1
python -u -m experiments.gemma3_head_pilot all --devices 0,1
