#!/usr/bin/env python3
"""Prepare a frozen MEGA train_val batch; never open reference or test files.

Run only on the experiment server. The manifest contains IDs and fingerprints.
Optional staged JSONL contains private vignette text and MUST NOT be committed.
No model inference or tokenizer-dependent selection is performed here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Iterator


DATASET_REVISION = "8f058c88ce1ed43693785b435ad0ff99907a2890"
OFFICIAL_COMMIT = "15a3a12e73a9169a870bff2ac431ec2e9968b8c7"
TEMPLATES = {
    "en": "mega_cdp/template/template_en/DECISION_MAKING_PREDICTION_TEMPLATE.txt",
    "zh": "mega_cdp/template/template_zh/ZH_DECISION_MAKING_PREDICTION_TEMPLATE.txt",
}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def value_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class MethodInput:
    """The entire method-facing data contract. No reference or row metadata."""
    guideline: str
    vignette: str
    language: str


@dataclass(frozen=True)
class TaskIdentity:
    """Logging/evaluation identity, kept outside MethodInput and the prompt."""
    id: str | int
    case_path_id: str | int
    path_index: str | int
    filename: str

    @property
    def sample_id(self) -> str:
        return value_digest([self.case_path_id, self.path_index, self.filename])


def train_paths(root: Path) -> tuple[Path, Path]:
    root = root.resolve()
    # No user-selected split or filename: development can only resolve train_val.
    if root.name in {"test", "train_val"}:
        raise ValueError("Pass the dataset snapshot root, not a split directory")
    split = root / "train_val"
    return split / "vignette_cases.jsonl", split / "guideline_md"


def safe_snapshot_path(root: Path, path: Path) -> str | None:
    """Permit train_val entries and the pinned snapshot's own HF blob links.

    User/data-derived relative paths cannot escape train_val. HF's own relative
    link targets legitimately contain '..', but may only reach this cache's blobs.
    Returns a blob object ID when full content verification is required.
    """
    root = root.resolve()
    path = Path(os.path.abspath(path))
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Path is outside the dataset snapshot") from exc
    if not relative.parts or relative.parts[0] != "train_val":
        raise ValueError("Only train_val snapshot entries are allowed")
    split = root / "train_val"
    if split.is_symlink() or not path.parent.resolve(strict=True).is_relative_to(split):
        raise ValueError("Snapshot entry parent resolves outside train_val")
    resolved = path.resolve(strict=True)
    if resolved.is_relative_to(split):
        if path.is_symlink() or not resolved.is_file():
            raise ValueError("Local snapshot entries must be regular files; links are only allowed to HF blobs")
        return None
    if not path.is_symlink() or root.parent.name != "snapshots" or root.name != DATASET_REVISION:
        raise ValueError("Only the fixed Hugging Face snapshot may link outside train_val")
    blob_dir = root.parent.parent / "blobs"
    link = path.readlink()
    direct_target = Path(os.path.abspath(path.parent / link))
    if (link.is_absolute() or blob_dir.is_symlink() or resolved.parent != blob_dir
            or direct_target != resolved or direct_target.is_symlink()
            or not resolved.is_file() or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", resolved.name)):
        raise ValueError("HF link must target a regular content-addressed file in the same cache blobs")
    return resolved.name


def snapshot_digest(root: Path, path: Path) -> str:
    """Hash full bytes, also validating HF SHA256/LFS or git-blob SHA1 names."""
    object_id = safe_snapshot_path(root, path)
    sha256 = hashlib.sha256()
    git_sha1 = hashlib.sha1(f"blob {path.stat().st_size}\0".encode()) if object_id and len(object_id) == 40 else None
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha256.update(block)
            if git_sha1 is not None:
                git_sha1.update(block)
    actual = sha256.hexdigest()
    if object_id and object_id != (git_sha1.hexdigest() if git_sha1 else actual):
        raise ValueError("HF blob content does not match its content-addressed filename")
    return actual


def safe_guideline_path(directory: Path, filename: str) -> Path:
    if (not filename or Path(filename).name != filename or filename in {".", ".."}
            or "/" in filename or "\\" in filename or "\0" in filename):
        raise ValueError("Invalid guideline filename")
    path = directory / filename / (filename + ".md")
    safe_snapshot_path(directory.parent.parent, path)
    return path


def records(root: Path):
    """Yield whitelisted official generation inputs, dropping all other fields."""
    source, _ = train_paths(root)
    snapshot_digest(root, source)
    seen = set()
    with source.open(encoding="utf-8") as handle:
        index = 0
        for line in handle:
            if not line.strip():
                continue
            index += 1
            raw = json.loads(line)
            case_id = raw.get("case_path_id", raw.get("guideline_id"))
            path_id = raw.get("path_index", raw.get("path_id"))
            filename, vignette = raw.get("filename"), raw.get("text")
            if (not isinstance(filename, str) or not isinstance(vignette, str)
                    or case_id is None or path_id is None):
                raise ValueError(f"Invalid official task schema at nonempty row {index}")
            identity = TaskIdentity(raw.get("id", index), case_id, path_id, filename)
            if identity.sample_id in seen:
                raise ValueError("Duplicate official task identity")
            seen.add(identity.sample_id)
            language = raw.get("language", "en")
            if not isinstance(language, str):
                raise ValueError("Language must be a string")
            yield identity, vignette, language


def build_manifest(root: Path, batch_size: int = 16,
                   partition: str = "development", calibration_fraction: float = .8,
                   official_root: Path | None = None) -> dict:
    """Freeze document groups before generation; split seed is always zero."""
    if batch_size < 1 or partition not in {"calibration", "development"}:
        raise ValueError("Invalid development batch selection")
    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must be between zero and one")
    source, guideline_dir = train_paths(root)
    document_hashes, filenames, entries, snapshot_files = {}, {}, [], {}
    for identity, vignette, language in records(root):
        if identity.filename not in document_hashes:
            path = safe_guideline_path(guideline_dir, identity.filename)
            document_hashes[identity.filename] = snapshot_digest(root, path)
            snapshot_files[str(path.relative_to(root.resolve()))] = document_hashes[identity.filename]
        document_sha = document_hashes[identity.filename]
        filenames.setdefault(document_sha, []).append(identity.filename)
        entries.append({"identity": asdict(identity), "sample_id": identity.sample_id,
                        "source_cluster": document_sha,
                        "input_sha256": value_digest([document_sha, vignette, language])})
    groups = sorted(set(document_hashes.values()), key=lambda key: value_digest([0, "split", key]))
    if len(groups) < 2:
        raise ValueError("At least two source documents are required for a disjoint split")
    count_calibration = min(len(groups) - 1, max(1, int(len(groups) * calibration_fraction)))
    membership = {key: "calibration" if index < count_calibration else "development"
                  for index, key in enumerate(groups)}
    selected = sorted((row for row in entries if membership[row["source_cluster"]] == partition),
                      key=lambda row: value_digest([0, "batch", row["sample_id"]]))[:batch_size]
    if len(selected) < batch_size:
        raise ValueError("Requested batch exceeds the chosen partition; no silent reduction")
    counts = Counter(row["source_cluster"] for row in entries)
    config = {"split_seed": 0, "method_development_seed": 0,
              "final_method_seeds": [0, 1, 2], "calibration_fraction": calibration_fraction,
              "batch_size": batch_size, "partition": partition,
              "selection": "SHA256(seed, purpose, identity); no label/length/performance selection"}
    templates = {} if official_root is None else {
        language: {"relative_path": path, "sha256": digest(official_root / path)}
        for language, path in TEMPLATES.items()
    }
    return {"schema_version": "aster.mega_trainval_manifest.v1",
            "status": "complete_verified", "scope": "data split and input manifest only; no model run",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "expected_dataset_revision": DATASET_REVISION, "expected_official_code_commit": OFFICIAL_COMMIT,
            "revision_note": "Expected registered revisions; actual file bytes are bound by SHA256 below. Coordinator records checkout HEAD separately.",
            "split": "train_val", "source_file_sha256": snapshot_digest(root, source),
            "snapshot_guideline_files": snapshot_files,
            "snapshot_link_policy": "fixed revision; same repository HF blobs only; complete SHA256 and content-address validation",
            "reader_sha256": digest(Path(__file__)), "config": config,
            "config_sha256": value_digest(config), "templates": templates,
            "group_definition": "released guideline document; identical SHA256 documents merged",
            "canonical_external_source_overlap": "unknown; content hashes cannot resolve all editions or translations",
            "total_rows": len(entries), "total_groups": len(groups),
            "groups": [{"source_cluster": group, "partition": membership[group],
                        "rows": counts[group], "filenames": sorted(set(filenames[group]))}
                       for group in groups], "selected": selected,
            "test_files_read": False, "reference_files_read": False,
            "method_input_fields": ["guideline", "vignette", "language"]}


def iter_method_tasks(root: Path, manifest: dict) -> Iterator[tuple[TaskIdentity, MethodInput]]:
    """Read only frozen selected generation inputs; method receives MethodInput."""
    if manifest.get("split") != "train_val":
        raise ValueError("Only train_val manifests are accepted")
    source, guideline_dir = train_paths(root)
    if manifest.get("expected_dataset_revision") != DATASET_REVISION:
        raise ValueError("Manifest dataset revision differs from the fixed snapshot")
    if snapshot_digest(root, source) != manifest["source_file_sha256"]:
        raise ValueError("Vignette source fingerprint changed after freezing")
    wanted = {entry["sample_id"]: entry for entry in manifest["selected"]}
    if len(wanted) != len(manifest["selected"]):
        raise ValueError("Duplicate frozen sample identity")
    tasks, cache = {}, {}
    for identity, vignette, language in records(root):
        entry = wanted.get(identity.sample_id)
        if entry is None:
            continue
        if asdict(identity) != entry["identity"]:
            raise ValueError("Frozen official identity mismatch")
        if identity.filename not in cache:
            path = safe_guideline_path(guideline_dir, identity.filename)
            frozen_sha = manifest.get("snapshot_guideline_files", {}).get(str(path.relative_to(root.resolve())))
            if frozen_sha != entry["source_cluster"]:
                raise ValueError("Guideline entry is absent from the frozen train_val file manifest")
            cache[identity.filename] = (snapshot_digest(root, path), path.read_text(encoding="utf-8"))
        document_sha, guideline = cache[identity.filename]
        if (document_sha != entry["source_cluster"]
                or value_digest([document_sha, vignette, language]) != entry["input_sha256"]):
            raise ValueError("Frozen model input fingerprint mismatch")
        tasks[identity.sample_id] = (identity, MethodInput(guideline, vignette, language))
    if set(tasks) != set(wanted):
        raise ValueError("Frozen selected tasks are missing")
    for entry in manifest["selected"]:
        yield tasks[entry["sample_id"]]


def render_official_prompt(task: MethodInput, official_root: Path, manifest: dict) -> str:
    language = "zh" if task.language == "zh" else "en"
    template_path = official_root / TEMPLATES[language]
    frozen = manifest.get("templates", {}).get(language)
    if not frozen or digest(template_path) != frozen["sha256"]:
        raise ValueError("Original template must be fingerprinted at split freeze")
    template = template_path.read_text(encoding="utf-8")
    return template.replace("{Guideline}", task.guideline).replace("{Case}", task.vignette)


def write_new_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    location = parser.add_mutually_exclusive_group(required=True)
    location.add_argument("--dataset-root", type=Path)
    location.add_argument("--config", type=Path, help="TRIAGE configs/datasets.json")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--partition", choices=["calibration", "development"], default="development")
    parser.add_argument("--calibration-fraction", type=float, default=.8)
    parser.add_argument("--stage-jsonl", type=Path,
                        help="Optional server-private vignette file; NEVER commit it")
    args = parser.parse_args()
    root = args.dataset_root
    if root is None:
        config = json.loads(args.config.read_text(encoding="utf-8"))["MEGA-CDP"]
        if config.get("revision") != DATASET_REVISION:
            raise ValueError("Dataset config revision differs from audited revision")
        root = Path(config["path"])
    manifest = build_manifest(root, args.batch_size, args.partition,
                              args.calibration_fraction, args.official_root)
    write_new_json(args.manifest, manifest)
    if args.stage_jsonl:
        args.stage_jsonl.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(args.stage_jsonl, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for identity, task in iter_method_tasks(root, manifest):
                row = {**asdict(identity), "text": task.vignette, "language": task.language}
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "complete_verified", "scope": "train_val input freeze only",
                      "selected": len(manifest["selected"]), "groups": manifest["total_groups"],
                      "manifest_sha256": digest(args.manifest), "inference": False}))


if __name__ == "__main__":
    main()
