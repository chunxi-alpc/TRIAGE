#!/usr/bin/env bash
set -euo pipefail
cd /home/gcx/TRIAGE-worktrees/aster-e0-e8-20261024
source /home/gcx/gcx261014/bin/activate
flex_root=/home/gcx/aster_e0_e8_20261024/private/flex_validation_v1
mkdir -p "$flex_root/deps"
if [[ ! -f "$flex_root/deps/extracted/usr/include/python3.12/Python.h" ]]; then
 (cd "$flex_root/deps" && apt-get download libpython3.12-dev && dpkg-deb -x libpython3.12-dev*.deb extracted)
fi
export CPATH="$flex_root/deps/extracted/usr/include:$flex_root/deps/extracted/usr/include/python3.12"
export CUDA_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1 TORCHINDUCTOR_COMPILE_THREADS=4
python -u experiments/aster_e0_e8_20261024/validate_flex.py
