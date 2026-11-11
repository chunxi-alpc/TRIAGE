"""Independent diagnostics; historical artifacts remain read-only."""
from pathlib import Path
from experiments.aster_cr_retrain_paired_v2_20261111.common import read, write, rows, lines, append, Store
REPO=Path(__file__).resolve().parents[2]
OUT=REPO/"reports/aster_cr_diagnostics_e3cv_20261111"
BASE=Path("/home/gcx/aster-cr-retrain-paired-v2-20261111/reports/aster_cr_retrain_paired_v2_20261111")
VERSION="aster_cr_diagnostics_e3cv_20261111_v1"
GROUPS=("E0","E1","E2","E3")
def model_root(group,fold="full"):
    return OUT if group=="E3" and fold!="full" else BASE
