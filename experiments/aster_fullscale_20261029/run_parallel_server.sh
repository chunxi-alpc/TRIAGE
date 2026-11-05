#!/usr/bin/env bash
set -euo pipefail
cd /home/gcx/TRIAGE-worktrees/aster-fullscale
source /home/gcx/gcx261014/bin/activate
run="${1:?pass a prepared run directory}"
script=experiments/aster_fullscale_20261029/scripts
if [ ! -f "$run/direction_catalog.json" ]; then
 python "$script/parallel_directions.py" catalog --run "$run"
fi
python "$script/serial_controller.py" --run "$run"
