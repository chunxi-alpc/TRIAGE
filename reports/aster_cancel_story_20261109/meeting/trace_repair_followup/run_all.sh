#!/usr/bin/env bash
set -euo pipefail
source /home/gcx/gcx261014/bin/activate
cd "$(dirname "$0")/../../../.."
TV=reports/aster_cancel_story_20261109/meeting/trace_repair_followup
mkdir -p /home/gcx/aster_trace_repair_20261110/logs
python "$TV/data.py" > /home/gcx/aster_trace_repair_20261110/logs/prepare_data.log 2>&1
python "$TV/score_worker.py" --prepare > /home/gcx/aster_trace_repair_20261110/logs/prepare_reuse.log 2>&1
CUDA_VISIBLE_DEVICES=6 python -u "$TV/data/enhance_diagnostics.py" --run --batch 6 > /home/gcx/aster_trace_repair_20261110/logs/annotations.log 2>&1 &
annotation_pid=$!
python -u "$TV/pipeline.py" > /home/gcx/aster_trace_repair_20261110/logs/pipeline.log 2>&1
wait "$annotation_pid"

