#!/usr/bin/env bash
set -euo pipefail
source /home/gcx/gcx261014/bin/activate
cd /home/gcx/TRIAGE-worktrees/aster-validation
export OMP_NUM_THREADS=1
r=/home/gcx/aster_validation_20261022/private/run500/cad200
e=experiments/aster_validation_20261022/mechanism_20261023
trap 'printf "failed at line %s\n" "$LINENO" > "$r/statistics.status"' ERR
printf 'waiting_for_selected200_inference_and_report\n' > "$r/statistics.status"
tail --pid=206119 -f /dev/null
test "$(cat "$r/publish.status")" = complete_and_pushed
python "$e/finish_cad200_statistics.py"
git add -- "$e/results/cad200_option_order"
if ! git diff --cached --quiet; then git commit -m 'Complete paired option-order statistics for the fixed Qwen 200 cases'; fi
git push origin HEAD:refs/heads/codex/aster-validation-20261022
printf 'complete_and_pushed\n' > "$r/statistics.status"
