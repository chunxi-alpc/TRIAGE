"""Reuse frozen E1 data and historical MEGA evaluation metadata on the server.

This module only prepares data. It does not train, score, select methods, or
claim experiment results. Prompt strings remain in the existing read-only
input table, addressed by UID without normalization or reconstruction.
"""
from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import shutil
import time
from typing import Any

REPO = Path(__file__).resolve().parents[2]
PREPARED = REPO / "reports/aster_codex03_external_20261112/prepared"
RAW_SOURCE_OUT = Path(
    "/home/gcx/aster-cr-retrain-paired-v2-20261111/"
    "reports/aster_cr_retrain_paired_v2_20261111"
)
FROZEN_DATA = Path("/home/gcx/aster_cr_pair_dataset_20261111")
HISTORICAL_DATA = Path("/home/gcx/aster_cr_repair_v2_run_20261108/data")
SPLITS = ("fit", "dev", "test")
BUNDLE_KEYS = ("views", "pairs", "cases", "path_views")
PATH_VARIANTS = ("original", "action_presence", "action_neutral", "action_swap")


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_rows(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_bundle(split: str, prepared: str | Path = PREPARED) -> dict[str, list[dict[str, Any]]]:
    """Return views, pairs, cases, path_views, with unchanged UID metadata.

    fit contains all 2,540 E1 pairs and their 5,080 endpoint records. Repeated
    endpoint UIDs are intentionally retained. fit has no evaluation paths.
    dev/test contain fixed historical pair records (including separate F0
    diagnostics), natural cases, and all natural/interference tree-node views.
    Prompt strings are obtained separately with get_input/get_prompt.
    """
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}")
    folder = Path(prepared) / split
    return {key: read_rows(folder / (key + ".jsonl")) for key in BUNDLE_KEYS}


@lru_cache(maxsize=1)
def load_inputs():
    """Open the existing UID-to-input mapping without copying the prompt table."""
    from experiments.aster_cr_retrain_paired_v2_20261111.common import load_inputs as existing_inputs
    return existing_inputs(RAW_SOURCE_OUT)


def get_input(uid: str):
    """Read one complete input record, including its exact prompt and labels."""
    return load_inputs()[uid]


def get_prompt(uid: str) -> str:
    return get_input(uid)["prompt"]


def prepare(prepared: str | Path = PREPARED) -> dict[str, Any]:
    """Copy existing metadata and record small count/source/join checks only."""
    started = time.perf_counter()
    destination = Path(prepared)
    destination.mkdir(parents=True, exist_ok=True)
    _write_prepared_gitignore(destination)
    source_data = RAW_SOURCE_OUT / "data"
    split_path = FROZEN_DATA / "source_splits.json"
    source_splits = json.loads(split_path.read_text(encoding="utf-8"))
    source_sets = {split: set(source_splits[split]) for split in SPLITS}
    if any(source_sets[a] & source_sets[b]
           for i, a in enumerate(SPLITS) for b in SPLITS[i + 1:]):
        raise ValueError("Existing fit/dev/test source lists overlap")

    fit_pairs = read_rows(source_data / "E1_pairs.jsonl")
    fit_views = read_rows(source_data / "E1_views.jsonl")
    kinds = Counter(pair["kind"] for pair in fit_pairs)
    if len(fit_pairs) != 2540 or dict(kinds) != dict.fromkeys(("A", "P", "C", "F"), 635):
        raise ValueError("Existing E1 file does not contain the requested 2,540 balanced pairs")
    historical_pairs = read_rows(source_data / "historical_pairs.jsonl")
    historical_views = read_rows(source_data / "historical_views.jsonl")
    historical_cases = read_rows(HISTORICAL_DATA / "path_cases.jsonl")
    path_views = read_rows(source_data / "path_features.jsonl")
    path_views += read_rows(source_data / "path_interference_views.jsonl")

    counts = {}
    for split in SPLITS:
        bundle = dict(
            views=fit_views if split == "fit" else [r for r in historical_views if r["split"] == split],
            pairs=fit_pairs if split == "fit" else [r for r in historical_pairs if r["split"] == split],
            cases=[] if split == "fit" else [r for r in historical_cases if r["split"] == split],
            path_views=[] if split == "fit" else [r for r in path_views if r["split"] == split],
        )
        for key, records in bundle.items():
            if any(r["source_id"] not in source_sets[split] for r in records):
                raise ValueError(f"Unexpected source in existing {split}/{key}")
        uids = {row["uid"] for row in bundle["views"]}
        if any(p[side] not in uids for p in bundle["pairs"] for side in ("left", "right")):
            raise ValueError(f"Pair endpoint missing from existing {split} views")
        for key, records in bundle.items():
            _write_rows(destination / split / (key + ".jsonl"), records)
        kind_counts = Counter(pair["kind"] for pair in bundle["pairs"])
        variants = Counter(row.get("variant", "original") for row in bundle["path_views"])
        path_keys = {(r["case_id"], r.get("variant", "original")) for r in bundle["path_views"]}
        sample_uid = bundle["views"][0]["uid"]
        if not isinstance(get_prompt(sample_uid), str) or not get_prompt(sample_uid):
            raise ValueError(f"Existing input lookup failed for {split}")
        counts[split] = dict(
            source_count=len(source_sets[split]),
            pair_count=len(bundle["pairs"]),
            pair_kinds=dict(kind_counts),
            ap_pairs=sum(kind_counts[k] for k in ("A", "P")),
            cf_pairs=sum(kind_counts[k] for k in ("C", "F")),
            f0_pairs=kind_counts["F0"],
            endpoint_records=len(bundle["views"]),
            unique_pair_input_uids=len(uids),
            cases=len(bundle["cases"]),
            path_node_records=len(bundle["path_views"]),
            path_node_variants=dict(variants),
            natural_paths=sum(v == "original" for _, v in path_keys),
            interference_paths=sum(v != "original" for _, v in path_keys),
            unique_scoring_uids=len(uids | {r["uid"] for r in bundle["path_views"]}),
            sample_input_lookup="passed",
        )

    # Keep every original E1 annotation, not only the slim scoring metadata.
    shutil.copyfile(FROZEN_DATA / "balanced_train.jsonl", destination / "E1_pairs_full.jsonl")
    shutil.copyfile(split_path, destination / "source_splits.json")
    index_reference = dict(
        source_output=str(RAW_SOURCE_OUT),
        input_jsonl=str(source_data / "inputs.jsonl"),
        input_index=str(RAW_SOURCE_OUT / "runtime/inputs.sqlite"),
        input_index_manifest=str(RAW_SOURCE_OUT / "runtime/input_index.json"),
        lookup="data.load_inputs()[uid] or data.get_input(uid)",
        prompt_policy="Read exact existing prompt and labels; no normalization, rewriting or reconstruction",
        raw_scores=dict(
            reused=str(source_data / "reused_scores.jsonl"),
            additional_pattern=str(RAW_SOURCE_OUT / "cache/scores_*.jsonl"),
            loader="experiments.aster_cr_retrain_paired_v2_20261111.common.load_records",
        ),
    )
    _write_json(destination / "input_reference.json", index_reference)
    result = dict(
        status="prepared_only",
        actual_experiment_results=False,
        training="E1 / Dataset A, 2,540 pairs from 80 fit sources",
        source_data_version="20261111-sol-agents-v2",
        source_paths=dict(
            complete_e1=str(FROZEN_DATA / "balanced_train.jsonl"),
            metadata=str(source_data),
            path_cases=str(HISTORICAL_DATA / "path_cases.jsonl"),
            source_splits=str(split_path),
        ),
        evaluation_scope="Historical 20-source development and 40-source controlled test; not official 7,458-case test",
        evaluation_history="The historical evaluation sets were previously used; this preparation does not make them new blind data",
        training_rule="Use fit data for fitting and dev for choices; freeze all choices before test scoring",
        f0_policy="Keep F0 diagnostics separate from A/P and C/F",
        fit_path_policy="No path evaluation exported for fitting; fit contains all requested E1 pairs",
        checks="Existing counts, source membership/disjointness, endpoint joins and three prompt lookups; no hashes or repeated data review",
        splits=counts,
        seconds=time.perf_counter() - started,
    )
    _write_json(destination / "summary.json", result)
    return result



def _write_prepared_gitignore(destination: Path) -> None:
    (destination / ".gitignore").write_text(
        "*\n!.gitignore\n!summary.json\n!input_reference.json\n!token_lengths.json\n",
        encoding="utf-8",
    )


def prepare_token_lengths(prepared: str | Path = PREPARED) -> dict[str, Any]:
    """Summarize already recorded E1 token lengths, without tokenization."""
    import math
    import statistics

    destination = Path(prepared)
    _write_prepared_gitignore(destination)
    fit_views = read_rows(destination / "fit/views.jsonl")
    wanted = {view["uid"] for view in fit_views}
    lengths = {}
    files_read = []
    candidates = [RAW_SOURCE_OUT / "data/reused_scores.jsonl"]
    candidates += sorted((RAW_SOURCE_OUT / "cache").glob("scores_*.jsonl"))
    for path in candidates:
        if not path.is_file():
            continue
        scanned = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                scanned += 1
                uid = record.get("uid", record.get("view_id"))
                value = record.get("tokens")
                if (uid in wanted and uid not in lengths
                        and record.get("status", "ok") == "ok"
                        and isinstance(value, (int, float)) and not isinstance(value, bool)
                        and value > 0 and math.isfinite(value) and int(value) == value):
                    lengths[uid] = int(value)
                if len(lengths) == len(wanted):
                    break
        files_read.append(dict(path=str(path), records_read=scanned))
        if len(lengths) == len(wanted):
            break
    values = sorted(lengths.values())
    def percentile(probability):
        if not values:
            return None
        position = (len(values) - 1) * probability
        lower = math.floor(position)
        upper = math.ceil(position)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)
    result = dict(
        status="existing_record_summary_only",
        actual_experiment_results=False,
        scope="E1 fit unique input UIDs",
        endpoint_records=len(fit_views),
        unique_input_uids=len(wanted),
        covered_uids=len(lengths),
        unknown_uids=len(wanted - lengths.keys()),
        coverage=len(lengths) / len(wanted) if wanted else None,
        tokens=dict(
            min=values[0] if values else None,
            median=statistics.median(values) if values else None,
            p95=percentile(0.95),
            max=values[-1] if values else None,
        ),
        percentile_method="linear interpolation of sorted observed token lengths",
        missing_policy="Unknown; no model loading or new tokenization",
        raw_score_files_read=files_read,
    )
    _write_json(destination / "token_lengths.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=PREPARED)
    args = parser.parse_args()
    summary = prepare(args.prepared)
    print(json.dumps(dict(status=summary["status"], splits=summary["splits"]), ensure_ascii=False))
