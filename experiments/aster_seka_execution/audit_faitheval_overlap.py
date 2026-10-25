#!/usr/bin/env python3
"""Server CPU provenance audit. Emits only ID/text hashes, metadata and counts.

FaithEval reads only subset/qid/id/question, never context, choices or gold.
HotpotQA context stays in process solely to reproduce logged builder prompt hashes.
No test rows are removed and no method receives test strings from this audit.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import types
import unicodedata


SEKA_PIN = "aab9cfed80dc012fb9e83f8bdefb8675c33f40e7"
BUILDER_BLOBS = {
    "src/custom_builders/adaptive/hotpot_qa_builder.py": "350f795c868306ded2c893c5356e68d013b82cc4",
    "src/model/projection_builder_base_adapsvd.py": "4a6e5640b40a4f773783915fe7900a22121b6326",
}
HOTPOT_SHA256 = "26650cf50234ef5fb2e664ed70bbecdfd87815e6bffc257e068efea5cf7cd316"
HOTPOT_MODULE = "src.custom_builders.adaptive.hotpot_qa_builder"


def text_sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalized_question(text):
    # Case/compatibility/whitespace only. Do not remove negation, digits or units.
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def question_hashes(text):
    if not isinstance(text, str) or not text.strip():
        return {"question_exact_sha256": None, "question_normalized_sha256": None}
    return {"question_exact_sha256": text_sha(text),
            "question_normalized_sha256": text_sha(normalized_question(text))}


def prefix_rows(path, count):
    """Read the original JSON array prefix in original order without loading 566MB."""
    decoder, buffer, position, began = json.JSONDecoder(), "", 0, False
    with Path(path).open() as handle:
        for index in range(count):
            while True:
                if not began:
                    buffer = buffer.lstrip()
                    if buffer:
                        if not buffer.startswith("["):
                            raise ValueError("Hotpot source must be the original JSON array")
                        position, began = 1, True
                if began:
                    while position < len(buffer) and buffer[position] in " \r\n\t,":
                        position += 1
                    if position < len(buffer) and buffer[position] == "]":
                        raise ValueError("Hotpot source contains fewer than the requested first 200 rows")
                    try:
                        value, end = decoder.raw_decode(buffer, position)
                    except json.JSONDecodeError:
                        pass
                    else:
                        if not isinstance(value, dict):
                            raise ValueError("Hotpot source row must be an object")
                        yield index, value
                        buffer, position = buffer[end:], 0
                        break
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    raise ValueError("Incomplete original Hotpot array prefix")
                if began and position:
                    buffer, position = buffer[position:] + chunk, 0
                else:
                    buffer += chunk


def option(arguments, name, default=None):
    for index, value in enumerate(arguments):
        if value == name and index + 1 < len(arguments):
            return arguments[index + 1]
        if value.startswith(name + "="):
            return value.split("=", 1)[1]
    return default


def pinned_assembly(seka_root):
    hashes, method = {}, None
    for relative, expected in BUILDER_BLOBS.items():
        path = Path(seka_root) / relative
        data = path.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if blob != expected:
            raise ValueError("SEKA builder source differs from the pinned implementation")
        hashes[relative] = hashlib.sha256(data).hexdigest()
        if relative.endswith("projection_builder_base_adapsvd.py"):
            tree = ast.parse(data.decode())
            base = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ProjectionBuilderBase")
            selected = next(n for n in base.body if isinstance(n, ast.FunctionDef) and n.name == "assemble_texts")
            namespace = {}
            exec(compile(ast.fix_missing_locations(ast.Module(body=[selected], type_ignores=[])), str(path), "exec"), namespace)
            method = namespace["assemble_texts"]
    return method, hashes


def calibration_signatures(rows):
    result = []
    for index, row in rows:
        source_id = row.get("_id")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("Original Hotpot _id is missing")
        item = {"source_index": index, "source_id_sha256": text_sha(source_id),
                **question_hashes(row.get("question"))}
        result.append(item)
    return result


def map_trace(rows, trace_rows, assemble, *, chat=False, tokenizer=None):
    by_pair = defaultdict(list)
    for index, row in rows:
        context = " ".join(f"{title} {''.join(sentences)}" for title, sentences in row["context"])
        h, hp = assemble(types.SimpleNamespace(chat=chat, tokenizer=tokenizer), context, row["question"])
        by_pair[(text_sha(h), text_sha(hp))].append(index)
    observed, ambiguous, outside = defaultdict(list), [], []
    for order, row in enumerate(trace_rows):
        pair = (row.get("text_h_sha256"), row.get("text_hp_sha256"))
        indices = by_pair.get(pair, [])
        if len(indices) == 1:
            observed[indices[0]].append({"trace_index": order, "included": row.get("included")})
        elif indices:
            ambiguous.append({"trace_index": order, "candidate_source_indices": indices})
        else:
            outside.append(order)
    memberships = {}
    for index, _row in rows:
        values = observed.get(index, [])
        memberships[index] = ("included" if len(values) == 1 and values[0]["included"] is True else
                              "rejected" if len(values) == 1 and values[0]["included"] is False else
                              "not_observed_in_trace" if not values else "ambiguous_trace")
    for item in ambiguous:
        for index in item["candidate_source_indices"]:
            memberships[index] = "ambiguous_trace"
    return memberships, {"trace_rows": len(trace_rows), "uniquely_mapped_trace_rows": sum(map(len, observed.values())),
                         "ambiguous_trace_rows": ambiguous, "unmapped_trace_indices": outside,
                         "membership_counts": dict(Counter(memberships.values()))}


def audit_build(name, report_path, rows, hotpot_path, hotpot_sha, assemble, builder_hashes):
    report = json.loads(Path(report_path).read_text())
    outcome = {"name": name, "build_report_sha256": file_sha(report_path), "status": "unknown"}
    try:
        if report.get("mode") != "build" or report.get("status") not in {"complete_pending_projection_audit", "complete_verified"}:
            raise ValueError("Report is not a completed calibration build")
        if report.get("upstream_module") != HOTPOT_MODULE or report.get("calibration_data_sha256") != hotpot_sha:
            raise ValueError("Calibration source/module does not match the actual Hotpot file")
        if report.get("upstream_builder_sha256") != builder_hashes["src/model/projection_builder_base_adapsvd.py"]:
            raise ValueError("Build report source differs from the pinned base builder")
        arguments = report["upstream_args"]
        if int(option(arguments, "--max_samples", "90447")) != 200:
            raise ValueError("Actual build does not declare the first 200 requests")
        if Path(option(arguments, "--data", "")).resolve() != Path(hotpot_path).resolve():
            raise ValueError("Build report data path differs")
        trace_path = Path(report["selection_trace_path"])
        if file_sha(trace_path) != report["selection_trace_sha256"]:
            raise ValueError("Selection trace does not match its build report")
        trace_rows = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
        tokenizer = None
        chat = "--chat" in arguments
        if chat:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(report["model_path"], local_files_only=True)
        memberships, detail = map_trace(rows, trace_rows, assemble, chat=chat, tokenizer=tokenizer)
        if detail["unmapped_trace_indices"]:
            raise ValueError("Trace contains prompts not attributable to the first 200 source rows")
        outcome.update(status="complete_verified", max_samples=200, original_array_prefix=True,
                       selection_trace_sha256=file_sha(trace_path), chat=chat, **detail)
        return memberships, outcome
    except Exception as error:
        outcome.update(error_type=type(error).__name__, error_sha256=text_sha(str(error)),
                       reason="Actual calibration membership could not be established from pinned source and trace")
        return {}, outcome


def compare_signatures(task, faith_rows, calibration, memberships):
    by_id, by_exact, by_normal = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in calibration:
        by_id[row["source_id_sha256"]].append(row["source_index"])
        if row["question_exact_sha256"]:
            by_exact[row["question_exact_sha256"]].append(row["source_index"])
            by_normal[row["question_normalized_sha256"]].append(row["source_index"])
    results, counts = [], Counter()
    for index, row in enumerate(faith_rows):
        explicit_id = row.get("qid", row.get("id"))
        explicit_id = explicit_id if isinstance(explicit_id, str) else None
        subset = row.get("subset") if isinstance(row.get("subset"), str) else None
        namespace = re_source_namespace(subset)
        hashes = question_hashes(row.get("question"))
        id_matches = by_id.get(text_sha(explicit_id), []) if explicit_id else []
        exact = by_exact.get(hashes["question_exact_sha256"], [])
        normal = by_normal.get(hashes["question_normalized_sha256"], [])
        proven = list(id_matches) if namespace == "hotpotqa" else []
        all_matches = sorted(set(proven + list(exact) + list(normal)))
        record = {"task": task, "row_index": index, "row_id_sha256": text_sha(explicit_id) if explicit_id else None,
                  "subset": subset, "explicit_source_namespace": namespace, **hashes,
                  "confirmed_hotpot_source_id_matches": proven,
                  "unnamespaced_same_id_candidates": list(id_matches) if namespace != "hotpotqa" else [],
                  "exact_question_match_indices": list(exact), "normalized_question_match_indices": list(normal),
                  "source_link_status": "confirmed_explicit_source_id" if proven else
                      "question_overlap_source_identity_unknown" if exact or normal else "unknown_no_observed_match",
                  "calibration_selection": {name: {str(i): values.get(i, "unknown") for i in all_matches}
                                              for name, values in memberships.items()}}
        results.append(record)
        counts["all_rows"] += 1
        counts["explicit_hotpot_subset_rows"] += namespace == "hotpotqa"
        counts["confirmed_source_id_overlap_rows"] += bool(proven)
        counts["exact_question_overlap_rows"] += bool(exact)
        counts["normalized_question_overlap_rows"] += bool(normal)
        counts["unknown_no_observed_match_rows"] += not bool(proven or exact or normal)
    return results, dict(counts)


def re_source_namespace(value):
    if not isinstance(value, str):
        return None
    compact = value.strip().casefold().replace("_", "").replace("-", "").replace(" ", "")
    return "hotpotqa" if compact == "hotpotqa" else None


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--hotpot", type=Path, required=True)
    parser.add_argument("--seka-root", type=Path, required=True)
    parser.add_argument("--build-report", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--faith-manifest", type=Path, required=True)
    for task in ("counterfactual", "inconsistent", "unanswerable"):
        parser.add_argument("--" + task, type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    hotpot_sha = file_sha(args.hotpot)
    if hotpot_sha != HOTPOT_SHA256:
        raise ValueError("Original HotpotQA file hash differs from the registered author calibration source")
    rows = list(prefix_rows(args.hotpot, 200))
    calibration = calibration_signatures(rows)
    assemble, builder_hashes = pinned_assembly(args.seka_root)
    memberships, builds = {}, []
    for spec in args.build_report:
        name, separator, path = spec.partition("=")
        if not separator or not name or name in memberships:
            raise ValueError("Use unique NAME=PATH build reports")
        membership, report = audit_build(name, path, rows, args.hotpot, hotpot_sha, assemble, builder_hashes)
        memberships[name] = membership
        builds.append(report)
    for record in calibration:
        record["calibration_membership"] = {name: values.get(record["source_index"], "unknown") for name, values in memberships.items()}
    with (args.output_dir / "calibration_hashes.jsonl").open("x") as handle:
        for row in calibration:
            handle.write(json.dumps(row) + "\n")
    manifest = json.loads(args.faith_manifest.read_text())
    datasets = []
    import pyarrow.parquet as pq
    for entry in manifest["datasets"]:
        task, path = entry["task"], getattr(args, entry["task"])
        if path.stat().st_size != entry["bytes"] or file_sha(path) != entry["sha256"]:
            raise ValueError("FaithEval file differs from the registered official object")
        parquet = pq.ParquetFile(path)
        columns = [field for field in ("subset", "qid", "id", "question") if field in parquet.schema_arrow.names]
        faith_rows = parquet.read(columns=columns).to_pylist()
        if len(faith_rows) != entry["rows_card"]:
            raise ValueError("FaithEval row count differs from the full frozen collection")
        signatures, counts = compare_signatures(task, faith_rows, calibration, memberships)
        with (args.output_dir / (task + ".hashes.jsonl")).open("x") as handle:
            for record in signatures:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        datasets.append({"task": task, "revision": entry["revision"], "sha256": entry["sha256"],
                         "columns_read": columns, "all_test_rows_retained": True, **counts})
    output = {"status": "complete_verified", "scope": "source hashes/overlap audit only; not an independence certificate",
              "calibration_file_sha256": hotpot_sha, "requested_source_prefix": 200,
              "builder_source_hashes": builder_hashes, "builds": builds, "datasets": datasets,
              "actual_calibration_membership_status": "complete_verified" if all(x["status"] == "complete_verified" for x in builds) else "unknown",
              "source_independence": "unknown; zero matches do not establish independence",
              "test_context_or_gold_columns_read": False, "test_question_text_exported": False,
              "test_rows_excluded": 0,
              "question_normalization": "Unicode NFKC, casefold, whitespace collapse; punctuation/negation/numbers retained"}
    (args.output_dir / "summary.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": output["status"], "datasets": len(datasets),
                      "source_independence": "unknown", "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
