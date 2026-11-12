#!/usr/bin/env bash
set -euo pipefail
cd /home/gcx/aster-e9-e11-20261112
source /home/gcx/gcx261014/bin/activate
bash reports/aster_e9_e11_20261112/run_e9.sh
bash reports/aster_e9_e11_20261112/run_e11.sh
python -m experiments.aster_e9_e11_20261112.summarize
