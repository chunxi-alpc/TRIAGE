#!/usr/bin/env bash
set -euo pipefail
cd /home/gcx/aster-e9-e11-20261112
source /home/gcx/gcx261014/bin/activate
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python -m experiments.aster_e9_e11_20261112.e11
