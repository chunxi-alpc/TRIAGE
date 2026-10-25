#!/usr/bin/env python3
"""Read server datasets; write only counts, IDs, field names, hashes and license excerpts.

Never emits patient, guideline, question, option or answer content. No inference.
Run on server, e.g. python audit_server_metadata.py --config /home/gcx/TRIAGE/configs/datasets.json --output audit.json
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata


SOURCE_KEYS = {"source_url", "guideline_url", "document_url", "doi", "source_id", "source_document_id", "guideline_title", "document_title"}
PAIR_KEYS = ("pair_id", "counterfactual_pair_id", "linked_pair_id")
LINK_KEYS = ("paired_instance_id", "paired_with", "counterfactual_of", "linked_instance_id")
CLUSTER_KEYS = ("source_rule_id", "rule_id", "source_guideline_id", "patient_id", "case_group_id")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def encoded_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def normalized_source(value):
    value = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    value = re.sub(r"^https?://(?:dx\.)?doi\.org/", "doi:", value)
    return re.sub(r"\s+", " ", value).rstrip("/")


def license_excerpt(root):
    path = root / "README.md"
    if not path.exists():
        return {"readme_present": False}
    text = path.read_text(encoding="utf-8")
    excerpts = []
    yaml = re.search(r"(?mi)^license:\s*([^\n]+)", text)
    if yaml:
        excerpts.append(yaml.group(0).strip())
    section = re.search(r"(?mi)^#{1,4}\s+License\s*$", text)
    if section:
        rest = text[section.end():]
        line = next((line.strip() for line in rest.splitlines() if line.strip()), "")
        # License section only; no README example is copied.
        if len(line) <= 250:
            excerpts.append(line)
    return {"readme_present": True, "readme_sha256": digest(path), "license_excerpts": excerpts}


def jsonl_metadata(path):
    count, keys, unique_ids, duplicates = 0, set(), set(), 0
    fields = Counter()
    sources = defaultdict(set)
    filenames, guidelines = set(), set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            count += 1
            fields.update(row.keys())
            keys.update(row)
            record_id = (row.get("guideline_id"), row.get("filename"), row.get("path_id"))
            key = encoded_hash(record_id)
            duplicates += int(key in unique_ids)
            unique_ids.add(key)
            if row.get("filename") is not None:
                filenames.add(normalized_source(row["filename"]))
            if row.get("guideline_id") is not None:
                guidelines.add(str(row["guideline_id"]))
            for field in SOURCE_KEYS:
                value = row.get(field)
                if isinstance(value, (str, int)) and str(value).strip():
                    sources[field].add(normalized_source(value))
    return ({"file": path.name, "sha256": digest(path), "rows": count, "fields": sorted(keys),
             "field_nonmissing_counts": dict(fields), "duplicate_record_keys": duplicates,
             "unique_guideline_ids": len(guidelines), "record_key_sha256": sorted(unique_ids),
             "source_value_counts": {k: len(v) for k, v in sources.items()}}, sources, filenames)


def extract_medpic_pairs(rows):
    ids = [str(row["instance_id"]) for row in rows]
    id_set = set(ids)
    if len(ids) != len(id_set):
        raise ValueError("MedPIC duplicate instance_id")
    pairs, errors, evidence = set(), [], []
    for field in PAIR_KEYS:
        groups = defaultdict(list)
        for row in rows:
            if row.get(field) not in (None, ""):
                groups[str(row[field])].append(str(row["instance_id"]))
        if groups:
            evidence.append(field)
        for members in groups.values():
            if len(set(members)) == 2:
                pairs.add(tuple(sorted(set(members))))
            else:
                errors.append({"field": field, "member_count": len(members)})
    for field in LINK_KEYS:
        for row in rows:
            value = row.get(field)
            if value in (None, ""):
                continue
            evidence.append(field)
            links = value if isinstance(value, list) else [value]
            for other in links:
                here, other = str(row["instance_id"]), str(other)
                if other not in id_set or other == here:
                    errors.append({"field": field, "invalid_link_from": here})
                else:
                    pairs.add(tuple(sorted((here, other))))
    verified = bool(evidence) and not errors and len(pairs) == 89
    return {"status": "complete_verified" if verified else "blocked",
            "evidence_fields": sorted(set(evidence)), "pair_count": len(pairs), "expected_pair_count": 89,
            "pairs": [list(x) for x in sorted(pairs)] if verified else None,
            "errors": errors, "reason": None if verified else "Explicit official links missing, malformed or not matching 89 pairs; no inferred pairs."}


def audit(config):
    result = {"schema_version": "aster.server_data_audit.v1", "new_inference": False, "datasets": {}, "source_overlap": {}}
    source_sets, document_hashes = {}, {}
    for name in ("MEGA-CDP", "MedPIC-Bench", "MedGUIDE"):
        root = Path(config[name]["path"])
        info = {"registered_revision": config[name].get("revision"), "exists": root.is_dir()}
        result["datasets"][name] = info
        if not root.is_dir():
            info["status"] = "blocked"
            continue
        info.update(license_excerpt(root))
        source_sets[name] = defaultdict(set)
        document_hashes[name] = set()
        if name == "MEGA-CDP":
            info["splits"] = {}
            split_source_names, split_hashes = {}, {}
            for split in ("train_val", "test"):
                record_reports = {}
                names = set()
                for filename in ("case_paths.jsonl", "decision_trees.jsonl", "gt_paths.jsonl", "vignette_cases.jsonl"):
                    path = root / split / filename
                    report, sources, fnames = jsonl_metadata(path)
                    record_reports[filename] = report
                    names |= fnames
                    for k, values in sources.items():
                        source_sets[name][k] |= values
                hashes = []
                for path in sorted((root / split / "guideline_md").rglob("*.md")):
                    sha = digest(path)
                    hashes.append({"path_id_sha256": encoded_hash(str(path.relative_to(root))), "sha256": sha, "bytes": path.stat().st_size})
                split_source_names[split] = names
                split_hashes[split] = {x["sha256"] for x in hashes}
                document_hashes[name] |= split_hashes[split]
                info["splits"][split] = {"records": record_reports, "guideline_markdown_count": len(hashes), "guideline_fingerprints": hashes}
            info["cross_split_same_filename_count"] = len(split_source_names["train_val"] & split_source_names["test"])
            info["cross_split_identical_document_sha256"] = sorted(split_hashes["train_val"] & split_hashes["test"])
        elif name == "MedPIC-Bench":
            path = root / "questions.json"
            rows = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(rows, list):
                raise ValueError("MedPIC questions.json must be a list; inspect schema without emitting text")
            fields = sorted(set().union(*(row.keys() for row in rows)))
            info.update({"file_sha256": digest(path), "rows": len(rows), "fields": fields,
                         "task_family_counts": dict(Counter(row.get("benchmark_task_family", "missing") for row in rows)),
                         "instance_ids": [row["instance_id"] for row in rows],
                         "pair_mapping": extract_medpic_pairs(rows),
                         "available_cluster_fields": [key for key in CLUSTER_KEYS if any(key in r for r in rows)]})
        else:
            import pyarrow.parquet as pq
            info["parquet_files"] = []
            for path in sorted(root.rglob("*.parquet")):
                pf = pq.ParquetFile(path)
                names = pf.schema_arrow.names
                selected = sorted(set(names) & SOURCE_KEYS)
                info["parquet_files"].append({"relative_path": str(path.relative_to(root)), "sha256": digest(path),
                                             "rows": pf.metadata.num_rows, "fields": names, "source_fields": selected})
                if selected:
                    for batch in pf.iter_batches(columns=selected):
                        for row in batch.to_pylist():
                            for key, value in row.items():
                                if isinstance(value, (str, int)) and str(value).strip():
                                    source_sets[name][key].add(normalized_source(value))
        info["status"] = "complete_verified"
        info["meaning_of_status"] = "metadata audit executed, not model inference, license clearance or independent-test clearance"
    mega, med = source_sets.get("MEGA-CDP", {}), source_sets.get("MedGUIDE", {})
    shared = sorted(set(mega) & set(med))
    comparisons = {key: {"mega_values": len(mega[key]), "medguide_values": len(med[key]),
                         "exact_overlap_count": len(mega[key] & med[key]),
                         "overlap_value_sha256": sorted(encoded_hash(v) for v in mega[key] & med[key])} for key in shared}
    result["source_overlap"] = {"MEGA_MedGUIDE": {"status": "partial_metadata_comparison" if shared else "unknown",
                                                    "comparisons": comparisons,
                                                    "independent_sources_verified": False,
                                                    "reason": "Missing source-document fields cannot establish disjointness; disease names and dataset-local IDs are insufficient."}}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(json.loads(args.config.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"written": str(args.output), "new_inference": False, "status": {k:v.get("status") for k,v in result["datasets"].items()}}))


if __name__ == "__main__":
    main()
