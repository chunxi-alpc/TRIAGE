#!/usr/bin/env bash
set -euo pipefail
diagnostic_output="${1:?usage: bash run_cpu.sh /home/gcx/aster_abc_new_run}"
source /home/gcx/gcx261014/bin/activate
source /home/gcx/aster_manuscript_20261030/venv-overlay/bin/activate
cd /home/gcx/TRIAGE-worktrees/aster-abc-diagnostics-20261030
python -m experiments.aster_abc_diagnostics_20261030.run_cpu --output "$diagnostic_output"
