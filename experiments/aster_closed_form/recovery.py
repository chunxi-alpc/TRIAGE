"""Prepare an independent corrected run and launch its definition pilot first.

Preparation performs CPU file operations only. Explicit --stage run starts
the six-node definition pilot on GPUs 0/1, then the existing complete pipeline
only when at least one definition construction passes its recorded validation.
"""
import argparse
import json
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .common import DEFAULT_ROOT, rows, status, write_json, write_rows

REPAIRED_ROOT = Path("/home/gcx/aster_closed_form_20261106_repaired")
INPUT_FILES = (
    "data/groups_main.jsonl",
    "data/groups_definition.jsonl",
    "data/path_cases.jsonl",
    "data/constructions.jsonl",
    "data/semantic_masks.jsonl",
    "data/path_semantic_masks.jsonl",
    "data/manifest.json",
    "configuration.json",
)


def timestamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def definition_pilot(groups):
    """Choose two nodes per original split using ancestors and ID, never scores."""
    chosen = []
    for split in ("fit", "dev", "test"):
        candidates = sorted(
            (g for g in groups if g["split"] == split),
            key=lambda g: (len(g.get("ancestors", [])), g["group_id"]),
        )
        if len(candidates) < 2:
            raise ValueError("The definition pilot needs two nodes in split " + split)
        chosen.extend(candidates[:2])
    return chosen


def _existing_preparation(root, source, mode):
    marker = root / "recovery_manifest.json"
    if marker.exists():
        previous = json.loads(marker.read_text())
        if previous["source_root"] != str(source) or previous["mode"] != mode:
            raise ValueError("Existing recovery directory belongs to different inputs: " + str(root))
        return previous
    if root.exists() and any(root.iterdir()):
        raise ValueError("Use an empty recovery directory; existing files will not be overwritten: " + str(root))
    return None


def _prepare_one(source, root, pilot_groups=None):
    mode = "definition_pilot" if pilot_groups is not None else "full"
    previous = _existing_preparation(root, source, mode)
    if previous is not None:
        return previous
    root.mkdir(parents=True, exist_ok=True)
    counts = {}
    for relative in INPUT_FILES:
        original = source / relative
        if not original.exists():
            raise FileNotFoundError(original)
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if relative.endswith(".jsonl"):
            values = rows(original)
            if mode == "definition_pilot":
                if relative == "data/groups_definition.jsonl":
                    values = pilot_groups
                else:
                    # Each pilot definition is reconstructed with the corrected
                    # prompt; no old construction or mask can suppress this run.
                    values = []
            elif relative in ("data/semantic_masks.jsonl", "data/path_semantic_masks.jsonl"):
                values = [v for v in values if v.get("status") == "complete"]
            write_rows(destination, values)
            counts[relative] = len(values)
        else:
            value = json.loads(original.read_text())
            if mode == "definition_pilot":
                value.update(
                    main_nodes={s: 0 for s in ("fit", "dev", "test")},
                    definition_targets={s: 2 for s in ("fit", "dev", "test")},
                    path_cases=0,
                )
                if relative == "data/manifest.json":
                    value["main_sources"] = {s: 0 for s in ("fit", "dev", "test")}
            write_json(destination, value)
    manifest = dict(
        source_root=str(source), mode=mode, prepared_at=timestamp(),
        copied_rows=counts, automatic_pilot=True, clinical_verified=False,
        downstream_caches_copied=False,
    )
    if pilot_groups is not None:
        manifest["selected_group_ids"] = [g["group_id"] for g in pilot_groups]
        manifest["selection"] = "Two per original split, ordered by ancestor count then group_id."
    write_json(root / "recovery_manifest.json", manifest)
    return manifest


def prepare_recovery(source=DEFAULT_ROOT, root=REPAIRED_ROOT, pilot_root=None):
    """Copy necessary inputs once; repeated preparation preserves new progress."""
    source, root = Path(source).resolve(), Path(root).resolve()
    pilot_root = Path(pilot_root).resolve() if pilot_root is not None else root / "definition_pilot"
    if source == root or source == pilot_root or root == pilot_root:
        raise ValueError("Source, repaired root and pilot root must be distinct.")
    selected = definition_pilot(rows(source / "data/groups_definition.jsonl"))
    main = _prepare_one(source, root)
    pilot = _prepare_one(source, pilot_root, selected)
    write_json(root / "recovery_preparation.json", dict(
        source_root=str(source), repaired_root=str(root), pilot_root=str(pilot_root),
        main=main, pilot=pilot,
    ))
    return dict(root=str(root), pilot_root=str(pilot_root), main=main, pilot=pilot)


def _merge_pilot(root, pilot_root):
    from .construct import _combine_records
    before = {r["group_id"]: r for r in rows(root / "data/constructions.jsonl")}
    combined = _combine_records(
        [root / "data/constructions.jsonl", pilot_root / "data/constructions.jsonl"], "group_id"
    )
    # A replaced construction invalidates masks made from the earlier input.
    replaced = {gid for gid, record in combined.items() if record != before.get(gid)}
    masks = {r["key"]: r for r in rows(root / "data/semantic_masks.jsonl")
             if r.get("status") == "complete" and r["group_id"] not in replaced}
    for record in rows(pilot_root / "data/semantic_masks.jsonl"):
        if record.get("status") == "complete":
            masks[record["key"]] = record
    write_rows(root / "data/constructions.jsonl", combined.values())
    write_rows(root / "data/semantic_masks.jsonl", masks.values())
    return dict(replaced_constructions=len(replaced), constructions=len(combined),
                semantic_masks=len(masks))


def _finish_launch(root, state):
    path = root / "launch.json"
    if path.exists():
        launch = json.loads(path.read_text())
        launch.update(state=state, finished_at=timestamp())
        write_json(path, launch)


def run_recovery(source=DEFAULT_ROOT, root=REPAIRED_ROOT, pilot_root=None):
    """Run pilot then full pipeline; an unavailable pilot is blocked, not a crash."""
    prepared = prepare_recovery(source, root, pilot_root)
    root, pilot_root = Path(prepared["root"]), Path(prepared["pilot_root"])
    from . import pipeline
    from .construct import REVISION, merge_construction_shards
    started = time.time()
    write_json(root / "recovery_run.json", dict(
        started_at=timestamp(), source_root=str(Path(source).resolve()),
        root=str(root), pilot_root=str(pilot_root), construction_revision=REVISION,
        gpus=[0, 1], medgemma_GiB_per_gpu=[55, 65], qwen_GiB_per_gpu=[17, 30],
        definition_pilot_nodes=6, definition_pilot_minutes_estimate=[15, 30],
        total_hours_estimate=[3, 6], estimates_require_pilot_timing=True,
    ))
    phase = "definition_pilot"
    try:
        workers = [pipeline.spawn(pilot_root, "construct", gpu, shard=gpu, shards=2)
                   for gpu in (0, 1)]
        write_json(pilot_root / "construction_workers.json",
                   dict(pids=[p.pid for p in workers], gpus=[0, 1], shards=2))
        status(root, "recovery", state="definition_pilot", updated_at=timestamp(),
               pilot_root=str(pilot_root), worker_pids=[p.pid for p in workers], gpus=[0, 1])
        pipeline.join(workers)
        merge_construction_shards(pilot_root, shards=2)
        constructions = rows(pilot_root / "data/constructions.jsonl")
        valid = [r for r in constructions
                 if r.get("status") == "complete"
                 and r.get("value", {}).get("definition_valid") is True
                 and r.get("validation", {}).get("available") is True]
        pilot_result = dict(
            completed_at=timestamp(), elapsed_seconds=time.time() - started,
            nodes=len(rows(pilot_root / "data/groups_definition.jsonl")),
            records=len(constructions), valid_definitions=len(valid),
            valid_group_ids=[r["group_id"] for r in valid],
            construction_status_counts=dict(Counter(r.get("status") for r in constructions)),
        )
        write_json(pilot_root / "pilot_result.json", pilot_result)
        if not valid:
            result = dict(state="blocked", reason="No pilot definition passed construction validation.",
                          full_pipeline_started=False, pilot=pilot_result)
            status(root, "recovery", updated_at=timestamp(), **result)
            _finish_launch(root, "blocked")
            return result
        merged = _merge_pilot(root, pilot_root)
        phase = "full_pipeline"
        status(root, "recovery", state=phase, updated_at=timestamp(),
               pilot=pilot_result, merged=merged, gpus=[0, 1])
        pipeline.all_stages(root)
        result = dict(state="complete", full_pipeline_started=True, pilot=pilot_result,
                      elapsed_seconds=time.time() - started)
        status(root, "recovery", updated_at=timestamp(), **result)
        _finish_launch(root, "complete")
        return result
    except Exception as exc:
        status(root, "recovery", state="failed", phase=phase,
               updated_at=timestamp(), error=str(exc), elapsed_seconds=time.time() - started)
        _finish_launch(root, "failed")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--root", type=Path, default=REPAIRED_ROOT)
    parser.add_argument("--pilot-root", type=Path)
    parser.add_argument("--stage", choices=("prepare", "run"), default="prepare")
    args = parser.parse_args()
    if args.stage == "prepare":
        result = prepare_recovery(args.source, args.root, args.pilot_root)
        print(json.dumps(dict(state="prepared", root=result["root"], pilot_root=result["pilot_root"])))
    else:
        result = run_recovery(args.source, args.root, args.pilot_root)
        print(json.dumps(dict(state=result["state"], full_pipeline_started=result["full_pipeline_started"])))
        # Natural construction unavailability is recorded explicitly, without
        # raising an exception or launching the full computation.


if __name__ == "__main__":
    main()
