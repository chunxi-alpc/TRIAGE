"""Server-only CSV loading; test access requires coordinator freeze first."""
from __future__ import annotations
import csv
import hashlib
from pathlib import Path

from medcalc_native import digest


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(root, split, lock, *, evaluation=False):
    if split not in {"train", "test"}:
        raise ValueError("Unknown MedCalc split")
    if split == "test" and (lock.get("frozen") is not True or not lock.get("frozen_at") or lock.get("seeds") != [0, 1, 2]):
        raise ValueError("Formal MedCalc test access requires the coordinator's full 0/1/2 freeze")
    path = Path(root) / "dataset" / (split + "_data.csv")
    sha = sha_file(path)
    if sha != lock.get("data", {}).get(split + "_sha256"):
        raise ValueError("CSV differs from the frozen/declared input file hash")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = 1100 if split == "test" else 10538
    if len(rows) != expected:
        raise ValueError("Unexpected actual MedCalc-Verified split size")
    ids = [str(row["Row Number"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Row Number is not unique; do not deduplicate by Note ID")
    if split == "train":
        selected = lock.get("train_row_numbers")
        if not isinstance(selected, list) or not selected or len(set(map(str, selected))) != len(selected):
            raise ValueError("Declare explicit training rows before a development run")
        if not set(map(str, selected)) <= set(ids):
            raise ValueError("Unknown training row numbers")
        by_id = dict(zip(ids, rows))
        rows = [by_id[str(key)] for key in selected]
    if evaluation:
        return rows, sha
    # Dataset loader retains grouping metadata outside the model input object.
    projected = []
    for row in rows:
        projected.append({"sample_id": str(row["Row Number"]),
            "source_cluster_sha256": digest([row.get("Note Type"), row.get("Note ID")]),
            "note_text_sha256": digest(row["Patient Note"]),
            "Patient Note": row["Patient Note"], "Question": row["Question"]})
    return projected, sha


def synthetic_rows():
    return [{"sample_id": "synthetic_sum", "Patient Note": "For a symbolic exercise, quantity x is 2 and quantity y is 3.",
             "Question": "What is x plus y?"},
            {"sample_id": "synthetic_unknown", "Patient Note": "For a symbolic exercise, quantity x is not recorded.",
             "Question": "What is quantity x?"}]
