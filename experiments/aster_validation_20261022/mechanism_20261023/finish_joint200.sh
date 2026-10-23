#!/usr/bin/env bash
set -euo pipefail
source /home/gcx/gcx261014/bin/activate
cd /home/gcx/TRIAGE-worktrees/aster-validation
export OMP_NUM_THREADS=1
r=/home/gcx/aster_validation_20261022/private/run500
e=experiments/aster_validation_20261022/mechanism_20261023
trap 'printf "failed at line %s\n" "$LINENO" > "$r/joint200.status"' ERR
printf 'waiting_for_Qwen_modified_CAD_natural_completion\n' > "$r/joint200.status"
tail --pid=87628 -f /dev/null
test "$(cat "$r/qwen.cad_fix_status")" = complete
# Existing publishers finish before this task modifies Git state.
tail --pid=87632 -f /dev/null
printf 'read_only_CAD_and_mechanism_summary\n' > "$r/joint200.status"
python "$e/cad_audit.py" --root "$r" --model qwen --out "$e/results/priority200/qwen/cad_finished_audit_500.json"
python "$e/joint_cad200.py" --root "$r" --out "$e/results/joint200_final"
python - <<'PY'
import json
from pathlib import Path
p=Path('experiments/aster_validation_20261022/mechanism_20261023/results/joint200_final/summary.json');assert json.loads(p.read_text())['CAD_complete']
PY
git add -- "$e/results/joint200_final" "$e/results/priority200/qwen/cad_finished_audit_500.json"
if ! git diff --cached --quiet; then git commit -m 'Combine completed Qwen CAD and mechanism results on the same 200 cases'; fi
git push origin HEAD:refs/heads/codex/aster-validation-20261022
printf 'complete_and_pushed\n' > "$r/joint200.status"
