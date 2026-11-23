#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source /home/gcx/gcx261014/bin/activate
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
archive=${ASTER_JOINT_ARCHIVE:-/home/gcx/aster-m3-joint-20261115/reports/aster_m3_joint_study_20261115}
output=${ASTER_CONTROLS_OUTPUT:-reports/aster_m3_missing_controls_20261123}
export ASTER_CONTROLS_OUTPUT="$output"
pkg=experiments.aster_m3_missing_controls_20261123
mkdir -p "$output"
stage=replay_a
# A failed rerun must not inherit a completed status from a previous run.
status() {
  python - "$output" "$1" "$stage" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

directory, state, stage = sys.argv[1:]
path = Path(directory) / "status.json"
record = dict(state=state, stage=stage, code_commit=subprocess.check_output(
    ["git", "rev-parse", "HEAD"], text=True).strip(), gpu_devices=[], gpu_memory_gib=0,
    model_forwards=0, backbone_gradients=0, activation_patching=0, m1_m2_refits=0,
    exploratory=True, independent_confirmation=False)
if state == "complete":
    record.update(outer_folds=5, historical_sources=80, new_inner_fits=15,
                  new_outer_fits=10, archived_cell_refits=0, heldout_scores_per_method=158161)
path.with_suffix(".tmp").write_text(json.dumps(record, indent=2) + "\n")
path.with_suffix(".tmp").replace(path)
PY
}
trap 'status partial' ERR
status partial
python -u -m "$pkg.replay_a" --archive "$archive" --output "$output" > "$output/A_run.log" 2>&1
stage=fit_b
python -u -m "$pkg.fit_b" --archive "$archive" --output "$output" > "$output/b_fit.log" 2>&1
stage=eval_b
/usr/bin/time -f 'seconds=%e max_rss_kib=%M' python -u -m "$pkg.eval_b" --archive "$archive" --output "$output" > "$output/b_evaluation.log" 2>&1
stage=tests
python -m unittest -v "$pkg.test_fit_b" "$pkg.test_results" > "$output/tests.log" 2>&1
stage=plot
python -m "$pkg.plot_results" --output "$output"
stage=completed
status complete
