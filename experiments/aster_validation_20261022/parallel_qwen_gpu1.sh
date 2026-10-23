#!/usr/bin/env bash
set -euo pipefail
cd /home/gcx/TRIAGE-worktrees/aster-validation
source /home/gcx/gcx261014/bin/activate
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRIAGE_JOB_OWNER=qwen_parallel_gpu1
r=/home/gcx/aster_validation_20261022/private/run500
e=experiments/aster_validation_20261022
m=/home/gcx/TRIAGE/models/modelscope/Qwen3-8B
s="$r/parallel_qwen/gpu1.status"
trap 'printf "failed at line %s\n" "$LINENO" > "$s"' ERR
printf 'waiting_for_existing_left_and_unsigned\n' > "$s"
python - <<'PY'
import json,time
from pathlib import Path
p=Path('/home/gcx/aster_validation_20261022/private/run500/option_order/qwen_left_unsigned.manifest.json')
while True:
 try:
  if p.exists() and json.loads(p.read_text()).get('complete'):break
 except json.JSONDecodeError:pass
 time.sleep(10)
PY
printf 'left_filtered_CAD\n' > "$s"
python "$e/cad_filtered.py" --model "$m" --input "$r/option_order/left.jsonl" --base "$r/option_order/qwen_left.jsonl" --output "$r/cad_fix/qwen_left_filtered.jsonl"
printf 'complete\n' > "$s"
