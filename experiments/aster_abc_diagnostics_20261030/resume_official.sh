#!/usr/bin/env bash
set -euo pipefail
source /home/gcx/gcx261014/bin/activate
source /home/gcx/aster_manuscript_20261030/venv-overlay/bin/activate
cd /home/gcx/TRIAGE-worktrees/aster-abc-diagnostics-20261030
resume_root="${1:-/home/gcx/aster_abc_diagnostics_20261030/official_resume_$(date +%Y%m%d_%H%M%S)}"
python -u -m experiments.aster_abc_diagnostics_20261030.resume_official --output "$resume_root"
