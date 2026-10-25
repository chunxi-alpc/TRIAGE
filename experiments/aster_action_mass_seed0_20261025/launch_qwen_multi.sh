#!/usr/bin/env bash
set -euo pipefail
cd /home/gcx/TRIAGE-worktrees/aster-action-mass-seed0-20261025
source /home/gcx/gcx261014/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
exp=experiments/aster_action_mass_seed0_20261025
root=/home/gcx/aster_action_mass_seed0_20261025/qwen_dual_v1
exec 9>"$root/run.lock"
flock -n 9 || exit 2
pids=()
for gpu in 0 1 6; do
 CUDA_VISIBLE_DEVICES="$gpu" python "$exp/run_dual.py" --name qwen --root "$root" --worker "$gpu" > "$root/worker$gpu.log" 2>&1 &
 pids+=("$!")
done
outcome=0
for pid in "${pids[@]}"; do wait "$pid" || outcome=$?; done
if [[ $outcome == 0 ]]; then python "$exp/merge_qwen.py" > "$root/merge.log" 2>&1 || outcome=$?; fi
if [[ $outcome == 0 ]]; then python "$exp/report.py" --root "$root" --models qwen --output "$exp/results/qwen_multi_v1" > "$root/report.log" 2>&1 || outcome=$?; fi
if [[ $outcome == 0 ]]; then python "$exp/verify.py" --root "$root" --models qwen --report "$exp/results/qwen_multi_v1" > "$root/verify.log" 2>&1 || outcome=$?; fi
python - "$outcome" <<'PY'
import json,sys,time
from pathlib import Path
code=int(sys.argv[1]);p=Path('experiments/aster_action_mass_seed0_20261025/results/run_state.json')
p.write_text(json.dumps({'state':'qwen_complete_verified' if code==0 else 'qwen_failed','exit_code':code,'updated_unix':time.time(),'gpus':[0,1,6],'llama':'paused per user; not launched','report':'qwen_multi_v1/REPORT.zh.md' if code==0 else None},indent=2))
PY
if [[ $outcome == 0 ]]; then git add "$exp/results/qwen_multi_v1"; fi
git add "$exp/results/run_state.json"
git commit -m "Archive three-GPU Qwen seed0 outcome (exit $outcome)"
git push origin HEAD:codex/aster-action-mass-seed0-20261025
exit "$outcome"
