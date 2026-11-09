#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
TV=reports/aster_cancel_story_20261109/meeting/trace_validation
python -u "$TV/shared.py" > "$TV/prepare.log" 2>&1
python "$TV/exp12/audit_reference.py" > "$TV/exp12/audit_reference.log" 2>&1
python "$TV/reference_audit.py" > "$TV/reference_audit.log" 2>&1
CUDA_VISIBLE_DEVICES='' python -u "$TV/exp12/run.py" > "$TV/exp12/run.log" 2>&1
python "$TV/exp12/replay.py" > "$TV/exp12/replay.log" 2>&1
CUDA_VISIBLE_DEVICES='' python -u "$TV/exp3/run_cv.py" > "$TV/exp3/run_cv.log" 2>&1
python "$TV/exp3/source_diagnostics.py" > "$TV/exp3/source_diagnostics.log" 2>&1
CUDA_VISIBLE_DEVICES='' python -u "$TV/exp4/run_repair.py" > "$TV/exp4/run.log" 2>&1
python "$TV/exp4/replay_saved.py" > "$TV/exp4/replay.log" 2>&1
if [[ "${1:-}" == "--full" ]]; then
 python "$TV/exp3/full_cv.py" --prepare > "$TV/exp3/prepare_full.log" 2>&1
 CUDA_VISIBLE_DEVICES=0 python -u "$TV/exp3/full_cv.py" --folds 0 2 4 > "$TV/exp3/full/gpu0.log" 2>&1 &
 cv_gpu0_pid=$!
 CUDA_VISIBLE_DEVICES=1 python -u "$TV/exp3/full_cv.py" --folds 1 3 > "$TV/exp3/full/gpu1.log" 2>&1 &
 cv_gpu1_pid=$!
 wait "$cv_gpu0_pid"
 wait "$cv_gpu1_pid"
 python "$TV/exp3/full_cv.py" --aggregate > "$TV/exp3/full/aggregate.log" 2>&1
 python "$TV/exp3/full_cv.py" --replay --folds 0 1 2 3 4 > "$TV/exp3/full/replay.log" 2>&1
 python "$TV/exp3/verify_replay.py" > "$TV/exp3/verify_replay.log" 2>&1
fi
python "$TV/render_tradeoff.py"
python "$TV/build_report.py"

