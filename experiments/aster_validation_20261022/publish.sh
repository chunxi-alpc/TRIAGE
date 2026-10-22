#!/usr/bin/env bash
set -euo pipefail
source /home/gcx/gcx261014/bin/activate
cd /home/gcx/TRIAGE-worktrees/aster-validation
exp=experiments/aster_validation_20261022
run_root=/home/gcx/aster_validation_20261022/private
trap 'printf "failed at line %s\n" "$LINENO" > "$run_root/delivery.status"' ERR
for pid in "$@"; do tail --pid="$pid" -f /dev/null; done
python "$exp/collect.py"
# Stage only this experiment. Never stage the unrelated primary checkout.
git add -- "$exp"
if ! git diff --cached --quiet; then git commit -m 'Record completed ASTER full runs and explicit blocked stages'; fi
git push origin HEAD:refs/heads/codex/aster-validation-20261022
printf 'pushed\n' > "$run_root/delivery.status"
