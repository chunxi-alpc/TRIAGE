#!/usr/bin/env python3
"""Run explicit V6 or experimental V7 on new inputs; never modify old runs."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

from common import BASELINE_COMMIT, load_modules, read_jsonl, sha256, source_hashes, write_json


def normalized_input(row):
    """Allowlist the actual model inputs; gold/reference fields are excluded."""
    case_id = row.get("case_id", row.get("sample_id"))
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("Each row requires case_id or sample_id")
    guideline = row.get("guideline", row.get("rules"))
    patient, question = row.get("patient"), row.get("question")
    if not isinstance(patient, str) or not isinstance(question, str) or not question.strip():
        raise ValueError(f"{case_id}: patient must be text and question must be nonempty text")
    if isinstance(guideline, dict) and "rules" in guideline:
        guideline = guideline["rules"]
    if not isinstance(guideline, list):
        raise ValueError(f"{case_id}: guideline/rules must contain structured rules")
    # Only implementation-facing fields are retained within each rule/atom.
    # In particular: no gold atom states, confidence labels or oracle links.
    rule_fields = {"rule_id", "text", "action", "action_id", "action_span", "source_span", "logic", "exception_logic", "explicitly_unconditional"}
    atom_fields = {"condition_id", "text", "field", "numeric", "criterion_kind", "source_span"}
    numeric_fields = {"field", "operator", "value", "unit"}
    rules = []
    for index, original in enumerate(guideline):
        rule = {key: value for key, value in original.items() if key in rule_fields}
        rule["rule_id"] = str(rule.get("rule_id") or f"R{index + 1}")
        for role in ("conditions", "exceptions"):
            rule[role] = []
            for source in original.get(role, []):
                atom = {key: value for key, value in source.items() if key in atom_fields}
                if isinstance(atom.get("numeric"), dict):
                    atom["numeric"] = {key: value for key, value in atom["numeric"].items() if key in numeric_fields}
                rule[role].append(atom)
        rules.append(rule)
    return {"case_id": case_id, "guideline": rules, "patient": patient, "question": question}


def digest_json(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def status_and_compact(graph, rules):
    all_conditions = [c for r in graph["rules"] for c in r["conditions"] + r["exceptions"]]
    technical = any(c.get("technical_error") or c.get("generation_status") == "ERROR" for c in all_conditions)
    parse_failed = any(not c.get("trialmatch", {}).get("raw_label") for c in all_conditions)
    status = "technical_error" if technical else "parse_failure" if parse_failed else "ok"
    sources = {r["rule_id"]: r for r in rules}
    compact = []
    for r in graph["rules"]:
        compact.append({"rule_id": r["rule_id"], "state": r["applicability"],
            "confidence": None, "confidence_is_calibrated": False,
            "action_id": sources[r["rule_id"]].get("action_id", sources[r["rule_id"]].get("action")),
            "support_evidence_ids": sorted({e["sentence_id"] for e in r["support_evidence"]}),
            "refute_evidence_ids": sorted({e["sentence_id"] for e in r["refute_evidence"]}),
            "selected_proof_evidence_ids": sorted({e["sentence_id"] for e in r.get("proof_evidence", [])}),
            "allow_attention_intervention": r["allow_attention_intervention"]})
    return status, compact


def package_versions():
    result = {}
    for package in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "numpy"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def local_model_manifest(model):
    path = Path(model)
    if not path.is_dir():
        return {"identifier": model, "local": False, "resolved_revision": None}
    files = ("config.json", "adapter_config.json", "tokenizer_config.json", "model.safetensors.index.json")
    return {"identifier": str(path.resolve()), "local": True,
            "config_sha256": {name: sha256(path / name) for name in files if (path / name).is_file()},
            "weight_files_hashed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", choices=("v6", "v7"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--base-model", default="microsoft/phi-4")
    parser.add_argument("--adapter-model", default="majdabd33/trialmatchai-phi4-reasoning-lora")
    parser.add_argument("--evidence-model", default="cnut1648/biolinkbert-mednli")
    parser.add_argument("--cache-dir")
    parser.add_argument("--candidate-limit", type=int, default=8)
    parser.add_argument("--nli-batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true", help="Validate schema and save a prepared manifest; do not load/call models")
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("Require 0 <= shard-index < shard-count")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    if min(args.candidate_limit, args.nli_batch_size, args.max_new_tokens) < 1:
        parser.error("candidate-limit, nli-batch-size and max-new-tokens must be positive")
    v6, v7 = load_modules()
    originals = read_jsonl(args.inputs)
    normalized = [normalized_input(row) for row in originals]
    ids = [row["case_id"] for row in normalized]
    if len(ids) != len(set(ids)):
        parser.error("Input case IDs must be unique")
    if args.limit:
        normalized = normalized[:args.limit]
    chosen = [row for index, row in enumerate(normalized) if index % args.shard_count == args.shard_index]
    for row in chosen:
        rule_ids = [rule["rule_id"] for rule in row["guideline"]]
        if len(rule_ids) != len(set(rule_ids)):
            parser.error(f"{row['case_id']}: duplicate rule IDs")
        v6.validate_v6_input(row["guideline"], row["patient"], row["question"])
        for rule in v6._rule_list(row["guideline"]):
            cids = [c["condition_id"] for c in rule["conditions"] + rule["exceptions"]]
            if len(cids) != len(set(cids)):
                parser.error(f"{row['case_id']}: duplicate condition IDs")
            if cids:
                v7.validate_logic(v7.full_rule_logic(rule), set(cids))
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"run_type": "prepared_real_model_inference" if args.dry_run else "real_model_inference",
        "status": "prepared_dry_run" if args.dry_run else "initializing", "version": args.version,
        "created_at": datetime.now(timezone.utc).isoformat(), "baseline_commit": BASELINE_COMMIT,
        "source_hashes": source_hashes(), "inputs_file_sha256": sha256(args.inputs),
        "normalized_input_sha256": digest_json(chosen), "case_ids": [x["case_id"] for x in chosen],
        "cases_total": len(originals), "cases_selected": len(chosen),
        "shard_count": args.shard_count, "shard_index": args.shard_index, "limit": args.limit,
        "device": args.device, "models": {name: local_model_manifest(value) for name, value in
            (("base", args.base_model), ("adapter", args.adapter_model), ("evidence", args.evidence_model))},
        "generation": {"max_new_tokens": args.max_new_tokens, "candidate_limit": args.candidate_limit,
                       "nli_batch_size": args.nli_batch_size, "do_sample": False},
        "environment": {"python": platform.python_version(), "packages": package_versions()},
        "gold_in_model_inputs": False, "input_policy": "allowlist_guideline_patient_question_excludes_reference_structure_and_structured_facts",
        "weights_or_models_called": False, "expert_review": False}
    write_json(args.output / "run_manifest.json", manifest)
    if args.dry_run:
        print({"status": manifest["status"], "cases": len(chosen), "model_calls": 0})
        return
    try:
        engine_type = v6.CAREModule1V6 if args.version == "v6" else v7.CAREModule1V7
        engine = engine_type.from_pretrained(device=args.device, cache_dir=args.cache_dir,
            base_model=args.base_model, adapter_model=args.adapter_model, evidence_model=args.evidence_model,
            nli_batch_size=args.nli_batch_size, candidate_limit=args.candidate_limit)
        engine.trialmatch.MAX_NEW_TOKENS = args.max_new_tokens
    except Exception as exc:
        manifest.update(status="initialization_failed", initialization_error={"type": type(exc).__name__, "message": str(exc)[:2000]})
        write_json(args.output / "run_manifest.json", manifest)
        raise
    manifest.update(status="running", weights_or_models_called=True)
    write_json(args.output / "run_manifest.json", manifest)
    counts = {"ok": 0, "parse_failure": 0, "technical_error": 0}
    with (args.output / "predictions.jsonl").open("x") as output:
        for row in chosen:
            started = time.perf_counter()
            record = {"case_id": row["case_id"], "sample_id": row["case_id"], "version": args.version,
                      "normalized_input_sha256": digest_json(row)}
            try:
                graph = engine.run(guideline=row["guideline"], patient=row["patient"], question=row["question"])
                status, compact = status_and_compact(graph, row["guideline"])
                record.update(status=status, rules=compact, result=graph)
            except Exception as exc:
                record.update(status="technical_error", rules=[], error={"type": type(exc).__name__, "message": str(exc)[:2000]})
            record["elapsed_seconds"] = time.perf_counter() - started
            counts[record["status"]] += 1
            output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            output.flush()
            print({"case_id": row["case_id"], "status": record["status"], "elapsed_seconds": round(record["elapsed_seconds"], 2)}, flush=True)
    manifest.update(status="completed", completed_at=datetime.now(timezone.utc).isoformat(), status_counts=counts,
                    predictions_sha256=sha256(args.output / "predictions.jsonl"))
    write_json(args.output / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()
