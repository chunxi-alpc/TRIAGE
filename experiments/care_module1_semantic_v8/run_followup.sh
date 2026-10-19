#!/usr/bin/env bash
set -euo pipefail
cd /home/gcx/TRIAGE-worktrees/care-module1-semantic-v8-20261019
root=experiments/care_module1_semantic_v8
arm=${1:?A0 or A1 required}
new_cases=${2:?new72 case path required}
test -f "$root/selection.json"
test -f "$new_cases"
mkdir -p "$root/results/confirmation_$arm" "$root/results/new72_$arm"
/home/gcx/n/.venv/bin/python "$root/semantic_runner.py" --cases "$root/data/semantic_confirmation_inputs.jsonl" --arm "$arm" --output "$root/results/confirmation_$arm" > "$root/results/confirmation_$arm/run.log" 2>&1
/home/gcx/n/.venv/bin/python "$root/semantic_runner.py" --cases "$new_cases" --arm "$arm" --output "$root/results/new72_$arm" > "$root/results/new72_$arm/run.log" 2>&1
