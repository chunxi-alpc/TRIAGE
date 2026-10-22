#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
E=experiments/aster_medguide_20261022
: "${ASTER_MODEL:?Set ASTER_MODEL to local Llama-3.1-8B-Instruct files}"
: "${MEDGUIDE_PARQUET:?Set MEDGUIDE_PARQUET to private parquet}"
: "${MEDGUIDE_PATHS:?Set MEDGUIDE_PATHS to private all_paths.json}"
mkdir -p "$E/private"
python "$E/prepare.py" --parquet "$MEDGUIDE_PARQUET" --paths "$MEDGUIDE_PATHS" \
  --output "$E/private/cases.jsonl" --max-cases "${ASTER_CASES:-300}" --seed "${ASTER_SEED:-42}"
python "$E/run.py" --input "$E/private/cases.jsonl" --output "$E/private/predictions.jsonl" \
  --model "$ASTER_MODEL" --device "${ASTER_DEVICE:-cuda:0}" \
  --layers "${ASTER_LAYERS:-16-21}" --strengths "${ASTER_STRENGTHS:-0.5,1,2,4}" \
  --max-cases "${ASTER_CASES:-300}"
python "$E/analyze.py" --cases "$E/private/cases.jsonl" \
  --predictions "$E/private/predictions.jsonl" --output "$E/private/analysis"
