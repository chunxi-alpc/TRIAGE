#!/usr/bin/env bash
set -euo pipefail
source /home/gcx/gcx261014/bin/activate
cd /home/gcx/TRIAGE-worktrees/aster-validation
export OMP_NUM_THREADS=1
r=/home/gcx/aster_validation_20261022/private/run500
e=experiments/aster_validation_20261022/mechanism_20261023
trap 'printf "failed at line %s\n" "$LINENO" > "$r/first_filtered200.status"' ERR
printf 'waiting_for_original_filtered_CAD_manifest\n' > "$r/first_filtered200.status"
python - <<'PY'
from pathlib import Path
import json,time
p=Path('/home/gcx/aster_validation_20261022/private/run500/cad_fix/qwen_original_filtered.manifest.json')
while True:
 try:
  if p.exists():
   m=json.loads(p.read_text())
   if m.get('complete') and m['rows']==4500:break
 except json.JSONDecodeError:pass
 time.sleep(30)
PY
python "$e/joint_cad200.py" --root "$r" --out "$e/results/joint200_original_filtered"
git add -- "$e/results/joint200_original_filtered"
if ! git diff --cached --quiet; then git commit -m 'Report completed original-order filtered CAD on the selected Qwen 200 cases'; fi
git push origin HEAD:refs/heads/codex/aster-validation-20261022
printf 'complete_and_pushed\n' > "$r/first_filtered200.status"
