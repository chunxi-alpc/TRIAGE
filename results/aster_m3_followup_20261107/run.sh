#!/usr/bin/env bash
set -euo pipefail
source /home/gcx/gcx261014/bin/activate
cd /home/gcx/TRIAGE-worktrees/aster-m3-p0-p3-20261107
export CUDA_VISIBLE_DEVICES=""
export TOKENIZERS_PARALLELISM=false
exec python -m experiments.aster_m3_followup.controller
