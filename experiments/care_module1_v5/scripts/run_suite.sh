#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/gcx/TRIAGE-v5-pr4
EXP="$ROOT/experiments/care_module1_v5"
PYTHON=/home/gcx/n/.venv/bin/python
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "$ROOT"
# One GPU is shared sequentially by the frozen regression and the new probes.
bash "$EXP/scripts/run_all_v5.sh"
"$PYTHON" "$EXP/exploratory/validate.py" > "$EXP/exploratory/validation_run.log" 2>&1
"$PYTHON" "$EXP/scripts/run_experiment.py" --expected-count 24 \
  --inputs "$EXP/exploratory/data/inputs_public.jsonl" \
  --rules "$EXP/exploratory/data/private/rules.jsonl" \
  --device cuda:6 --output "$EXP/exploratory/runs/frozen_v1" > "$EXP/exploratory/inference.log" 2>&1
"$PYTHON" "$EXP/exploratory/evaluate.py" \
  --predictions "$EXP/exploratory/runs/frozen_v1/predictions_public.jsonl" \
  --output "$EXP/exploratory/results" > "$EXP/exploratory/evaluate.log" 2>&1
