#!/usr/bin/env bash
set -euo pipefail
source /home/gcx/gcx261014/bin/activate
cd /home/gcx/TRIAGE-worktrees/aster-validation
export OMP_NUM_THREADS=1
base=/home/gcx/aster_validation_20261022/private/run500
r="$base/cad200"
e=experiments/aster_validation_20261022/mechanism_20261023
trap 'printf "failed at line %s\n" "$LINENO" > "$r/publish.status"' ERR
printf 'waiting_for_selected200_original_filtered\n' > "$r/publish.status"
python - <<'PY'
from pathlib import Path
import json,time
p=Path('/home/gcx/aster_validation_20261022/private/run500/cad200/original_filtered.manifest.json')
while True:
 try:
  if p.exists() and json.loads(p.read_text()).get('complete'):break
 except json.JSONDecodeError:pass
 time.sleep(10)
PY
python "$e/joint_cad200.py" --root "$base" --cad-dir "$r" --out "$e/results/cad200_original"
git add -- "$e/results/cad200_original"
if ! git diff --cached --quiet; then git commit -m 'Report original-order filtered CAD restricted to selected 200 cases'; fi
git push origin HEAD:refs/heads/codex/aster-validation-20261022
printf 'original_complete_and_pushed; waiting_for_selected200_orders\n' > "$r/publish.status"
for pid in "$@"; do tail --pid="$pid" -f /dev/null; done
test "$(cat "$r/gpu0.status")" = complete
test "$(cat "$r/gpu1.status")" = complete
python "$e/joint_cad200.py" --root "$base" --cad-dir "$r" --out "$e/results/cad200_final"
git add -- "$e/results/cad200_final"
if ! git diff --cached --quiet; then git commit -m 'Report all remaining Qwen CAD evaluations on selected 200 cases only'; fi
git push origin HEAD:refs/heads/codex/aster-validation-20261022
printf 'complete_and_pushed\n' > "$r/publish.status"
