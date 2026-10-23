#!/usr/bin/env bash
set -euo pipefail
source /home/gcx/gcx261014/bin/activate
cd /home/gcx/TRIAGE-worktrees/aster-validation
for pid in "$@"; do tail --pid="$pid" -f /dev/null; done
python experiments/aster_validation_20261022/mechanism_20261023/collect.py
git add -- experiments/aster_validation_20261022/mechanism_20261023
if ! git diff --cached --quiet; then git commit -m 'Record completed or blocked mechanism experiment results'; fi
git push origin HEAD:refs/heads/codex/aster-validation-20261022
