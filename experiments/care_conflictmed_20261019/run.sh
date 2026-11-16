#!/usr/bin/env bash
set -euo pipefail
source /home/gcx/gcx261014/bin/activate
TASK_ROOT=$(cd "$(dirname "$0")" && pwd)
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
python "$TASK_ROOT/src/run.py" > "$TASK_ROOT/logs/run.log" 2>&1
python "$TASK_ROOT/src/analyze.py" > "$TASK_ROOT/logs/analysis.log" 2>&1
