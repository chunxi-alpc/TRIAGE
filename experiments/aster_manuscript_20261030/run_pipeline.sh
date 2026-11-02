#!/usr/bin/env bash
# Run on the authorized server after activating the recorded environment.
set -euo pipefail
if [ "$#" -ne 1 ]; then echo "usage: bash run_pipeline.sh /absolute/new/run-directory" >&2; exit 2; fi
ASTER_RUN_ROOT="$1"
mkdir "$ASTER_RUN_ROOT"
git rev-parse HEAD > "$ASTER_RUN_ROOT/evaluation_code_commit.txt"
python - "$ASTER_RUN_ROOT" <<'PY'
import json,subprocess,sys
from pathlib import Path
result=subprocess.check_output(['nvidia-smi','--id=0,1','--query-gpu=index,memory.used,memory.free','--format=csv,noheader,nounits'],text=True)
Path(sys.argv[1],'gpu_before.txt').write_text(result)
if any(int(line.split(',')[1])>0 for line in result.strip().splitlines()):
 raise RuntimeError('GPU 0 or 1 is occupied; no model jobs started')
PY
export OMP_NUM_THREADS=4
python -m experiments.aster_manuscript_20261030.prepare --original-actions --output "$ASTER_RUN_ROOT/data" > "$ASTER_RUN_ROOT/prepare.log" 2>&1
CUDA_VISIBLE_DEVICES=0 python -m experiments.aster_manuscript_20261030.collect --data "$ASTER_RUN_ROOT/data" --output "$ASTER_RUN_ROOT/collect0" --shard 0 > "$ASTER_RUN_ROOT/collect0.log" 2>&1 & p0=$!
CUDA_VISIBLE_DEVICES=1 python -m experiments.aster_manuscript_20261030.collect --data "$ASTER_RUN_ROOT/data" --output "$ASTER_RUN_ROOT/collect1" --shard 1 > "$ASTER_RUN_ROOT/collect1.log" 2>&1 & p1=$!
wait "$p0"; wait "$p1"
CUDA_VISIBLE_DEVICES="" python -m experiments.aster_manuscript_20261030.analyze --run "$ASTER_RUN_ROOT" > "$ASTER_RUN_ROOT/analyze.log" 2>&1
CUDA_VISIBLE_DEVICES=0 python -m experiments.aster_manuscript_20261030.evaluate --run "$ASTER_RUN_ROOT" --mode gain --shard 0 > "$ASTER_RUN_ROOT/gain0.log" 2>&1 & p0=$!
CUDA_VISIBLE_DEVICES=1 python -m experiments.aster_manuscript_20261030.evaluate --run "$ASTER_RUN_ROOT" --mode gain --shard 1 > "$ASTER_RUN_ROOT/gain1.log" 2>&1 & p1=$!
wait "$p0"; wait "$p1"
python -m experiments.aster_manuscript_20261030.report freeze --run "$ASTER_RUN_ROOT" > "$ASTER_RUN_ROOT/freeze.log" 2>&1
CUDA_VISIBLE_DEVICES=0 python -m experiments.aster_manuscript_20261030.evaluate --run "$ASTER_RUN_ROOT" --mode evaluate --shard 0 > "$ASTER_RUN_ROOT/evaluate0.log" 2>&1 & p0=$!
CUDA_VISIBLE_DEVICES=1 python -m experiments.aster_manuscript_20261030.evaluate --run "$ASTER_RUN_ROOT" --mode evaluate --shard 1 > "$ASTER_RUN_ROOT/evaluate1.log" 2>&1 & p1=$!
wait "$p0"; wait "$p1"
python -m experiments.aster_manuscript_20261030.report summarize --run "$ASTER_RUN_ROOT" > "$ASTER_RUN_ROOT/summarize.log" 2>&1
python -m experiments.aster_manuscript_20261030.supplement --run "$ASTER_RUN_ROOT" > "$ASTER_RUN_ROOT/supplement.log" 2>&1

