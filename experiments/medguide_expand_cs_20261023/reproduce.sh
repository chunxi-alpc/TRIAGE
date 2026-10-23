#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root, in the existing server environment.
EXP=experiments/medguide_expand_cs_20261023
SOURCE=${SOURCE:-/home/gcx/aster_medguide_20261022/private}
PRIVATE=${PRIVATE:-/home/gcx/medguide_expand_cs_20261023/private}
DEV=${DEV:-$EXP/results/dev_exclusions.jsonl}
MODEL=${MODEL:-/home/gcx/TRIAGE/models/modelscope/Qwen3-8B}
RESULTS=${RESULTS:-$EXP/results}
mkdir -p "$RESULTS" "$PRIVATE"
python "$EXP/prepare.py" --source "$SOURCE" --output "$RESULTS" --private "$PRIVATE" --dev "$DEV"
CUDA_VISIBLE_DEVICES=0 python "$EXP/closed_book.py" --input "$PRIVATE/cohort.jsonl" --output "$RESULTS/closed_book_0.jsonl" --model "$MODEL" --shard 0 &
FIRST_PID=$!
CUDA_VISIBLE_DEVICES=1 python "$EXP/closed_book.py" --input "$PRIVATE/cohort.jsonl" --output "$RESULTS/closed_book_1.jsonl" --model "$MODEL" --shard 1 &
SECOND_PID=$!
wait "$FIRST_PID"
wait "$SECOND_PID"
python "$EXP/acquire_external.py" --private "$PRIVATE/external" --output "$RESULTS/external_sources.json"
python "$EXP/external.py" --private "$PRIVATE/external" --output "$RESULTS"
python -m unittest discover -s "$EXP" -p 'test_*.py' > "$RESULTS/tests.txt" 2>&1
python "$EXP/finalize.py" --source "$SOURCE" --private "$PRIVATE" --output "$RESULTS" --dev "$DEV"
