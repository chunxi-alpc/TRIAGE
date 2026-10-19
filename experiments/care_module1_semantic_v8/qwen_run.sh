#!/usr/bin/env bash
set -euo pipefail
cd /home/gcx/TRIAGE-worktrees/care-module1-semantic-v8-20261019
source /home/gcx/gcx261014/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=3
python - <<'PY'
import json
from pathlib import Path
root=Path("experiments/care_module1_semantic_v8")
selection=json.loads((root/"selection.json").read_text())
assert selection["selected"]=="A1"
manifest=json.loads((root/"results/confirmation_A1/manifest.json").read_text())
assert manifest["status"]=="completed", "Confirmation is still running; no Qwen run started"
rows=[json.loads(line) for line in (root/"results/confirmation_A1/predictions.jsonl").read_text().splitlines()]
assert len(rows)==192
PY
python experiments/care_module1_semantic_v8/qwen_smoke.py \
  --predictions experiments/care_module1_semantic_v8/results/confirmation_A1/predictions.jsonl \
  --variant A1
