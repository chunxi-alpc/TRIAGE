#!/usr/bin/env python3
"""Server CPU only: paired efficacy differences with whole-entity bootstrap.

Reads the fixed standard train[5000:10000] and verified anonymous exports.
Never outputs entity text, prompts, answers, or clinical test data. No inference.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np


METHODS = ("seka", "matched_zero", "unsteered", "pasta")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def selected_methods(requested, locations):
    """Default remains four methods; an explicit list never invents missing runs."""
    names = tuple(METHODS if requested is None else requested)
    require(len(names) >= 2 and len(set(names)) == len(names) and "seka" in names,
            "Select SEKA and at least one distinct baseline")
    require(set(names) <= set(METHODS), "Unknown selected method")
    require(all(locations.get(name) is not None for name in names), "A selected method has no actual export directory")
    require(not any(locations.get(name) is not None for name in METHODS if name not in names),
            "An export was supplied for an unselected method; do not silently exclude existing results")
    return tuple(name for name in METHODS if name in names)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def hashed(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def iter_json_objects(path, chunk_size=1024 * 1024):
    """Read JSONL or adjacent JSON objects without rewriting their source bytes."""
    decoder = json.JSONDecoder()
    buffer, position, ended, count = "", 0, False, 0
    with Path(path).open(encoding="utf-8") as handle:
        while True:
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position >= len(buffer):
                if ended:
                    return
                chunk = handle.read(chunk_size)
                buffer, position = chunk, 0
                ended = not chunk
                continue
            try:
                row, next_position = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError as exc:
                if ended:
                    raise ValueError(f"Malformed JSON object at record {count}") from exc
                chunk = handle.read(chunk_size)
                buffer, position = buffer[position:] + chunk, 0
                ended = not chunk
                continue
            require(isinstance(row, dict), "Expected an ordered stream of JSON objects")
            count += 1
            yield row
            position = next_position
            if position > chunk_size:
                buffer, position = buffer[position:], 0


def source_clusters(path):
    """Official object order; exact entity field equality defines a cluster."""
    selected = {}
    source_sha = digest(path)
    for index, row in enumerate(itertools.islice(iter_json_objects(path), 10000)):
        if index < 5000:
            continue
        sid, entity = str(row["id"]), row["entity"]
        require(isinstance(entity, str) and entity.strip(), "Missing official entity")
        require(sid not in selected, "Duplicate standard CounterFact case ID")
        selected[sid] = {"sample_id": hashed(["CounterFact", source_sha, index, sid]),
                         "entity_cluster": hashed(["CounterFact-entity-exact", entity]),
                         "absolute_processed_row_index": index}
    require(len(selected) == 5000, "Official standard slice does not contain 5000 cases")
    return selected, source_sha


def load_export(folder):
    folder = Path(folder)
    audit_path, cases_path = folder / "audit.json", folder / "case_metrics.jsonl"
    prediction_path = folder / "predictions.jsonl"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    require(audit.get("status") == "complete_verified" and audit.get("benchmark") == "efficacy",
            "Requires an independently verified efficacy export")
    require(digest(cases_path) == audit["case_metrics_sha256"], "Case metrics hash changed")
    require(digest(prediction_path) == audit["anonymous_predictions_sha256"], "Prediction hash changed")
    rows, values = jsonl(cases_path), {}
    for row in rows:
        sid, score = str(row["id"]), float(row["score"])
        require(sid not in values, "Duplicate exported case ID")
        require(score in (0., 1.) and int(row["prompt_count"]) == 1,
                "Only native binary efficacy cases are accepted; not paraphrase")
        values[sid] = score
    require(len(values) == 5000, "Missing standard efficacy cases")
    mean = math.fsum(values.values()) / len(values)
    require(math.isclose(mean, float(audit["recomputed_metrics"]["score"]["mean"]),
                         rel_tol=1e-10, abs_tol=1e-12), "Exported efficacy mean differs")
    return values, {"audit_sha256": digest(audit_path), "case_metrics_sha256": digest(cases_path),
                    "predictions_sha256": digest(prediction_path), "efficacy": mean,
                    "raw_file_sha256": audit.get("raw_file_sha256"),
                    "source_commit": audit.get("source_commit"),
                    "input_files_sha256": audit.get("input_files_sha256", {}),
                    "model_files_sha256": audit.get("model_files_sha256", {})}


def bootstrap_effects(values, cluster_ids, replicates=10000, seed=0):
    """Uniformly sample full entities; ratio of sums retains case weighting."""
    require(replicates >= 100, "At least 100 bootstrap samples required")
    ids, inverse = np.unique(np.asarray(cluster_ids), return_inverse=True)
    require(len(ids) >= 2, "At least two entities required for an interval")
    values = np.asarray(values, dtype=float)
    sums = np.zeros((len(ids), values.shape[1]), dtype=float)
    np.add.at(sums, inverse, values)
    sizes = np.bincount(inverse).astype(float)
    rng = np.random.default_rng(seed)
    boot = np.empty((replicates, values.shape[1]), dtype=float)
    for start in range(0, replicates, 128):
        size = min(128, replicates - start)
        drawn = rng.integers(0, len(ids), size=(size, len(ids)))
        denominator = sizes[drawn].sum(axis=1)
        # Shared draws preserve pairing across all methods and all reported rates.
        for col in range(values.shape[1]):
            boot[start:start + size, col] = sums[:, col][drawn].sum(axis=1) / denominator
    return np.quantile(boot, [.025, .975], axis=0).T.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Official counterfact.jsonl")
    for name in METHODS:
        parser.add_argument("--" + name.replace("_", "-"), type=Path,
                            help="Verified anonymous efficacy export directory")
    parser.add_argument("--methods", nargs="+", choices=METHODS,
                        help="Optional explicit actual method list; omission retains the original four-method default")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"status": "running", "new_gpu_forwards": 0,
              "scope": "standard implementation validation; not independent clinical confirmation",
              "date_utc": datetime.now(timezone.utc).isoformat(),
              "split": "train[5000:10000]", "all_negative_results_retained": True,
              "source_format": "ordered JSON objects; supports JSONL and adjacent objects without source rewriting",
              "model_seed_inference": "One existing greedy run per method; no multi-seed result claimed",
              "planned_methods_without_results": ["CAD", "Spotlight"],
              "paraphrase": "not analyzed by this binary efficacy script",
              "method_parameters_changed": False}
    code = 0
    try:
        chosen = selected_methods(args.methods, {name: getattr(args, name) for name in METHODS})
        references = [name for name in chosen if name != "seka"]
        report["selected_methods"] = list(chosen)
        report["planned_methods_without_results"] = [name for name in METHODS if name not in chosen] + ["CAD", "Spotlight"]
        mapping, source_sha = source_clusters(args.data)
        methods, evidence = {}, {}
        for name in chosen:
            methods[name], evidence[name] = load_export(getattr(args, name))
            require(set(methods[name]) == set(mapping), "Method/source standard ID coverage differs")
            require(source_sha in evidence[name]["input_files_sha256"].values(),
                    "Entity source file hash is absent from an inference run's input manifest")
        checkpoint = evidence["seka"]["model_files_sha256"]
        require(checkpoint and all(evidence[name]["model_files_sha256"] == checkpoint for name in chosen),
                "Compared methods do not have identical recorded checkpoint fingerprints")
        report["checkpoint_fingerprints_identical"] = True
        ordered = sorted(mapping, key=lambda sid: mapping[sid]["absolute_processed_row_index"])
        entities = [mapping[sid]["entity_cluster"] for sid in ordered]
        columns, comparisons = [], {}
        for reference in references:
            delta = np.asarray([methods["seka"][sid] - methods[reference][sid] for sid in ordered])
            corrected, regressed = (delta > 0).astype(float), (delta < 0).astype(float)
            columns.extend([delta, corrected, regressed])
            comparisons[reference] = {"candidate": "seka", "reference": reference,
                "difference": float(delta.mean()), "difference_percentage_points": float(100 * delta.mean()),
                "corrected": int(corrected.sum()), "regressed": int(regressed.sum()),
                "unchanged": int((delta == 0).sum()), "denominator_all_cases": len(ordered),
                "correction_rate": float(corrected.mean()), "regression_rate": float(regressed.mean())}
        intervals = bootstrap_effects(np.asarray(columns).T, entities,
                                      args.bootstrap_samples, args.bootstrap_seed)
        for position, reference in enumerate(references):
            item = comparisons[reference]
            item.update(ci95_difference=intervals[3 * position],
                        ci95_difference_percentage_points=[100 * x for x in intervals[3 * position]],
                        ci95_correction_rate=intervals[3 * position + 1],
                        ci95_regression_rate=intervals[3 * position + 2])
        output = args.output_dir / "paired_cases.jsonl"
        with output.open("x", encoding="utf-8") as handle:
            for sid in ordered:
                row = {**mapping[sid], "efficacy": {name: methods[name][sid] for name in chosen}}
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        sizes = Counter(entities)
        report.update(status="complete_verified", source_sha256=source_sha, methods=evidence,
                      paired_comparisons=comparisons, n_cases=len(ordered), n_entity_clusters=len(sizes),
                      repeated_entity_clusters=sum(size > 1 for size in sizes.values()),
                      largest_entity_cluster=max(sizes.values()), paired_cases_sha256=digest(output),
                      bootstrap_samples=args.bootstrap_samples, bootstrap_seed=args.bootstrap_seed,
                      numpy_version=np.__version__, interval="two-sided marginal 95% percentile",
                      bootstrap_unit="exact public entity field; all facts of an entity resampled together",
                      weighting="case-weighted native efficacy: cluster sum / resampled case count",
                      ci_scope="Conditional on this fixed checkpoint, protocol and existing runs; entity independence assumed. Marginal CIs are not multiplicity-adjusted or a clinical confirmation.",
                      input_equivalence="Existing forward-trace comparison provides matched-input evidence; this script verifies IDs, export hashes and scores only")
    except Exception as exc:
        code = 1
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        report.update(elapsed_seconds=time.monotonic() - started, checker_sha256=digest(Path(__file__)))
        (args.output_dir / "entity_ci.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "new_gpu_forwards": 0}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
