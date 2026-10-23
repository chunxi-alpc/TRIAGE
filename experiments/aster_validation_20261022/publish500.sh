#!/usr/bin/env bash
set -euo pipefail
source /home/gcx/gcx261014/bin/activate
cd /home/gcx/TRIAGE-worktrees/aster-validation
root=/home/gcx/aster_validation_20261022/private/run500
trap 'printf "failed at line %s\n" "$LINENO" > "$root/delivery.status"' ERR
for pid in "$@"; do tail --pid="$pid" -f /dev/null; done
python experiments/aster_validation_20261022/collect500.py
git add -- experiments/aster_validation_20261022
if ! git diff --cached --quiet; then git commit -m 'Record actual results and stage status for 500-test-case protocol'; fi
git push origin HEAD:refs/heads/codex/aster-validation-20261022
printf 'pushed\n' > "$root/delivery.status"
