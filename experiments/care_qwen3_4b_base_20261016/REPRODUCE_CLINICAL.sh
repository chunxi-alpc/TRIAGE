#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source /home/gcx/gcx261014/bin/activate
mkdir -p results/obs1_clinical/logs
python scripts/build_clinical_pairs.py > results/obs1_clinical/logs/build_data.log 2>&1
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_clinical_o1.py > results/obs1_clinical/logs/experiment.log 2>&1
python scripts/analyze_clinical_o1.py > results/obs1_clinical/logs/analysis.log 2>&1
python -m pytest -q > results/obs1_clinical/logs/pytest_final.log 2>&1
python scripts/integrate_clinical_results.py > results/obs1_clinical/logs/integration.log 2>&1
