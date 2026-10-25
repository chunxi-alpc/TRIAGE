#!/usr/bin/env bash
set -u
cd /home/gcx/TRIAGE-worktrees/aster-dynamic-fusion-review
source /home/gcx/gcx261014/bin/activate
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4
exp=experiments/aster_dynamic_fusion_seed0_20261025
# Wait for process exit, not repeated GPU/status monitoring.
tail --pid="$1" --pid="$2" --pid="$3" -f /dev/null
python "$exp/finalize_fresh.py"
result=$?
if [[ "$result" == 0 ]]; then
  git add "$exp/fresh_numeric"
  git add "$exp/balanced_fresh.json" "$exp/fresh_report.zh.md" "$exp/execution.json" "$exp/frozen_development_recheck.json" "$exp/seed0_offline_reproduced.json"
fi
git add "$exp/run_state.json"
git commit -m "Record ASTER matched-control forward outcome (exit $result)"
git push origin HEAD:codex/aster_dynamic_fusion_seed0_20261025
exit "$result"
