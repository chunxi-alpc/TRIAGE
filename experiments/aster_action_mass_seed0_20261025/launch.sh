#!/usr/bin/env bash
set -euo pipefail
cd /home/gcx/TRIAGE-worktrees/aster-action-mass-seed0-20261025
source /home/gcx/gcx261014/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
root=/home/gcx/aster_action_mass_seed0_20261025
exp=experiments/aster_action_mass_seed0_20261025
if [[ ${1:-} == pilot ]]; then
 CUDA_VISIBLE_DEVICES=0 python "$exp/run.py" --name qwen --limit 4 > "$root/qwenpilot.log" 2>&1 &
 a=$!
 CUDA_VISIBLE_DEVICES=1 python "$exp/run.py" --name llama --limit 4 > "$root/llamapilot.log" 2>&1 &
 b=$!
 wait "$a"
 wait "$b"
 exit
fi
# GPU1 showed power capping; same Llama probe recovered on GPU0.
# Sequential GPU0 avoids both wasted time and occupying a degraded device.
exec 9>"$root/full.lock"
flock -n 9 || exit 2
outcome=0
CUDA_VISIBLE_DEVICES=0 python "$exp/run.py" --name qwen > "$root/qwenfull.log" 2>&1 || outcome=$?
if [[ $outcome == 0 ]]; then
 CUDA_VISIBLE_DEVICES=0 python "$exp/run.py" --name llama > "$root/llamafull.log" 2>&1 || outcome=$?
fi
if [[ $outcome == 0 ]]; then
 python "$exp/report.py" --root "$root" --output "$exp/results/final_v1" > "$root/report.log" 2>&1 || outcome=$?
fi
if [[ $outcome == 0 ]]; then
 python "$exp/verify.py" --root "$root" --report "$exp/results/final_v1" > "$root/verify.log" 2>&1 || outcome=$?
fi
python - "$outcome" <<'PY'
import json,sys,time
from pathlib import Path
code=int(sys.argv[1]);p=Path('experiments/aster_action_mass_seed0_20261025/results/run_state.json')
p.write_text(json.dumps({'state':'complete_verified' if code==0 else 'failed','exit_code':code,'updated_unix':time.time(),'private_logs':'/home/gcx/aster_action_mass_seed0_20261025','complete_report':'final_v1/REPORT.zh.md' if code==0 else None},indent=2))
PY
if [[ $outcome == 0 ]]; then
 git add "$exp/results/final_v1"
fi
git add "$exp/results/run_state.json"
git commit -m "Archive ASTER seed0 execution outcome (exit $outcome)"
git push origin HEAD:codex/aster-action-mass-seed0-20261025
exit "$outcome"
