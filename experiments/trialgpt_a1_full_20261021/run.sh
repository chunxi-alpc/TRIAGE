#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source /home/gcx/gcx261014/bin/activate
export PYTHONPATH=/home/gcx/care_steering_20261019/.deps
export TOKENIZERS_PARALLELISM=false
EXPERIMENT=experiments/trialgpt_a1_full_20261021
mkdir -p "$EXPERIMENT/private" "$EXPERIMENT/results"

# Closing GitHub PR19 does not stop remote nohup. Reject GPU overlap.
python - <<'PY'
from pathlib import Path
old=(b"experiments/medguide_full_20261021/run.sh",
     b"experiments/medguide_full_20261021/qwen_run.py",
     b"experiments/medguide_full_20261021/private/input_")
found=[]
for p in Path("/proc").glob("[0-9]*/cmdline"):
    try: cmd=p.read_bytes()
    except (OSError, PermissionError): continue
    if any(x in cmd for x in old):
        found.append((p.parent.name,cmd.replace(b"\0",b" ")[:240]))
if found:
    for pid,cmd in found:
        print("OLD MEDGUIDE PROCESS STILL RUNNING",pid,cmd.decode(errors="replace"))
    raise SystemExit("Refusing TrialGPT startup; stop PR19 workers first")
print("Old MedGUIDE processes: none detected")
PY

python "$EXPERIMENT/prepare.py"
python "$EXPERIMENT/tests_cpu.py"

# Four independent A1 model instances, one case partition per GPU.
# This script is interactive/supervised; no nohup or background monitoring.
pids=""
for gpu in 0 1 2 3; do
    (
        export CUDA_VISIBLE_DEVICES="$gpu"
        python experiments/care_module1_semantic_v8/semantic_runner.py \
          --cases "$EXPERIMENT/private/inputs_$gpu.jsonl" \
          --arm A1 --output "$EXPERIMENT/private/a1_$gpu" \
          --batch-size 8 --max-new-tokens 384 \
          > "$EXPERIMENT/private/a1_$gpu.log" 2>&1
    ) &
    pids="$pids $!"
done
failed=0
for pid in $pids; do
    if ! wait "$pid"; then failed=1; fi
done
if ! python "$EXPERIMENT/evaluate.py"; then failed=1; fi
exit "$failed"

