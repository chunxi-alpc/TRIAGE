#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source /home/gcx/gcx261014/bin/activate
mkdir -p results/logs
python scripts/build_minimal_pairs.py --n-groups 60 --seed 42 --out data/derived/pilot.jsonl > results/logs/build_data.log 2>&1
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_experiments.py > results/logs/experiment.log 2>&1
python -u scripts/analyze_results.py > results/logs/analysis.log 2>&1
python -m pytest -q > results/logs/pytest_final.log 2>&1
python scripts/validate_deliverables.py > results/logs/deliverables.log 2>&1
