#!/usr/bin/env bash
set -euo pipefail
source /home/gcx/gcx261014/bin/activate
cd /home/gcx/TRIAGE-worktrees/aster-validation
tail --pid="$1" -f /dev/null
python experiments/aster_validation_20261022/mechanism_20261023/collect_priority200.py
git add -- experiments/aster_validation_20261022/mechanism_20261023/results/priority200
if ! git diff --cached --quiet; then git commit -m 'Publish requested Qwen 200-case mechanism results'; fi
git push origin HEAD:refs/heads/codex/aster-validation-20261022
