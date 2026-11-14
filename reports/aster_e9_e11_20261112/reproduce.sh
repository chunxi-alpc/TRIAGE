#!/usr/bin/env bash
set -euo pipefail
cd /home/gcx/aster-e9-e11-20261112
source /home/gcx/gcx261014/bin/activate
export CUDA_VISIBLE_DEVICES=""
# Reproduce statistics from saved predictions and measurements; no model inference.
python -c "from experiments.aster_e9_e11_20261112.summarize import e9; e9()"
python -m experiments.aster_e9_e11_20261112.e11_shared_missing merge
# GPU commands and executed settings are archived in:
# e11_shared_missing/merged/executed_codex06_jobs.json
# Historical launch commands remain in run_e9.sh and run_e11.sh.
