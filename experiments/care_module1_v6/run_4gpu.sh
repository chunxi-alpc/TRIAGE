#!/bin/bash
source /home/gcx/gcx261014/bin/activate
set -euo pipefail
ROOT=/home/gcx/TRIAGE
OUT=$ROOT/experiments/care_module1_v6/runs/new100
SCRIPT=$ROOT/experiments/care_module1_v6/scripts_run_new_v6.py
for spec in "0 25 cuda:0" "25 50 cuda:1" "50 75 cuda:2" "75 100 cuda:3"; do
  set -- $spec
  nohup /home/gcx/n/.venv/bin/python "$SCRIPT" --inputs "$ROOT/experiments/care_module1_v6/data/new_inputs_100.jsonl" --rules "$ROOT/experiments/care_module1_v6/data/new_rules_100.jsonl" --output "$OUT" --device "$3" --start "$1" --end "$2" > "$OUT/log_${1}_${2}.txt" 2>&1 &
  echo $! >> "$OUT/pids.txt"
done
wait
