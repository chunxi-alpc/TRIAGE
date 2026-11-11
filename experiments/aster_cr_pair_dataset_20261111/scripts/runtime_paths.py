"""Server locations for the published Sol workflow; no work runs on import."""
import os
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
SCRIPTS = PACKAGE / "scripts"
SCHEMA = PACKAGE / "pair.schema.json"
DEFAULT_ROOT = Path("/home/gcx/aster_cr_pair_dataset_20261111")
DEFAULT_DATA = Path("/home/gcx/aster_cr_repair_v2_run_20261108/data")
DEFAULT_RAW = Path("/home/gcx/TRIAGE/data/MEGA-CDP/train_val")
DEFAULT_REPO = Path("/home/gcx/aster-final-two-20261111")


def configured(name, default):
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


ROOT = configured("ASTER_PAIR_ROOT", DEFAULT_ROOT)
DATA = configured("ASTER_PAIR_DATA", DEFAULT_DATA)
RAW = configured("ASTER_PAIR_RAW", DEFAULT_RAW)
REPO = configured("ASTER_PAIR_REPO", DEFAULT_REPO)


def source_document_path(value):
    """Resolve retained original document paths when the raw directory is moved."""
    path = Path(value)
    try:
        relative = path.relative_to(DEFAULT_RAW)
    except ValueError:
        return path
    return RAW / relative
