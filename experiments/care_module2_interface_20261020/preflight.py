#!/usr/bin/env python3
"""Offline, read-only prerequisite audit; the explicit JSON report is the only write.

No installation, download, network probe, model load, inference, or M1 recomputation
is performed. Exit 0 means prerequisites for limited runtime checks are present;
exit 2 means blocked or an invalid/existing output path. Neither is efficacy evidence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone


SNAPSHOT_COMMIT = "7557d3150214a3894f2d340df6d9964de75efce1"
ALLOWED_PHYSICAL_GPUS = {0, 1, 2, 3, 6, 7}
MODEL_REVISION = "906bfd4b4dc7f14ee4320094d8b41684abff8539"
WEIGHT_HASHES = {
    "model-00001-of-00003.safetensors": "4c807e2503d68ae373d508689d00a41f4b33f33c2536da97ab81a20caddc1241",
    "model-00002-of-00003.safetensors": "f4707585548b2fc75a6b1d732e8465c62040a8699903c32850781beeb9b27826",
    "model-00003-of-00003.safetensors": "c7b1aa8fb672de2e00423c99876926022e50b18d4f0d140670788510a27f9965",
}
RAW_INPUT_HASHES = {
    "inputs_192.jsonl": "32ec4446d80a3f514827da8a5df8a0206aa4eceea63f9940df99876de7c4955b",
    "new_inputs_100.jsonl": "3d0c58bd3cadb0b23038954bee95df58e066936d7de21623f09d39e7e4de8602",
}
EXPECTED_ARCHITECTURE = {
    "model_type": "qwen3", "num_hidden_layers": 36,
    "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def add_check(report, name, passed, detail):
    report["checks"][name] = {"status": "pass" if passed else "blocked", "detail": detail}
    if not passed:
        report["runtime_blockers"].append(name)


def probe_import(name: str):
    """Actually import in an isolated offline child; do not install missing modules."""
    if importlib.util.find_spec(name) is None:
        return {"importable": False, "reason": "module_not_installed"}
    code = (
        "import importlib,json; m=importlib.import_module(" + repr(name) + "); "
        "r={'importable':True,'version':getattr(m,'__version__',None)}; "
    )
    if name == "torch":
        code += (
            "r.update(cuda_available=m.cuda.is_available(),"
            "cuda_device_count=m.cuda.device_count(),cuda_build=m.version.cuda); "
        )
    code += "print(json.dumps(r))"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    try:
        proc = subprocess.run(
            [sys.executable, "-B", "-c", code], capture_output=True, text=True,
            timeout=30, env=env, check=False,
        )
        if proc.returncode:
            return {"importable": False, "reason": "import_failed", "returncode": proc.returncode,
                    "error_tail": proc.stderr[-1600:]}
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError) as exc:
        return {"importable": False, "reason": type(exc).__name__, "detail": str(exc)[:1600]}


def audit_runtime(report, requested_gpus):
    imports = {name: probe_import(name) for name in ("torch", "transformers", "numpy", "scipy")}
    report["runtime"] = {"python": platform.python_version(), "executable": sys.executable,
                         "platform": platform.platform(), "imports": imports}
    for name in ("torch", "transformers"):
        add_check(report, "import_" + name, imports[name]["importable"], imports[name])
    add_check(report, "cuda_available", imports["torch"].get("cuda_available") is True,
              {"cuda_available": imports["torch"].get("cuda_available"),
               "cuda_device_count": imports["torch"].get("cuda_device_count"),
               "note": "No model, tensor allocation, or inference is attempted."})
    add_check(report, "gpu_authorization", set(requested_gpus) <= ALLOWED_PHYSICAL_GPUS,
              {"requested_physical_ids": requested_gpus, "allowed_physical_ids": sorted(ALLOWED_PHYSICAL_GPUS)})
    visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    detail = {"CUDA_VISIBLE_DEVICES": visibility, "requested_physical_ids": requested_gpus}
    if visibility is None:
        detail["interpretation"] = "No explicit visibility mask; future runners must set one explicitly."
        visibility_ok = True
    else:
        try:
            visible = [int(x.strip()) for x in visibility.split(",") if x.strip()]
            detail["visible_physical_ids"] = visible
            visibility_ok = set(requested_gpus) <= set(visible)
        except ValueError:
            visibility_ok = False
            detail["interpretation"] = "UUID/MIG visibility mapping needs explicit physical-GPU verification."
    add_check(report, "gpu_visibility", visibility_ok, detail)
    binary = shutil.which("nvidia-smi")
    inventory = {"nvidia_smi": binary, "devices": []}
    if binary:
        try:
            proc = subprocess.run(
                [binary, "--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10, check=False,
            )
            inventory["returncode"] = proc.returncode
            if proc.returncode == 0:
                for row in csv.reader(io.StringIO(proc.stdout)):
                    if not row:
                        continue
                    inventory["devices"].append({"physical_id": int(row[0].strip()), "name": row[1].strip(),
                                                  "memory_total_MiB": int(row[2].strip()),
                                                  "memory_free_MiB": int(row[3].strip())})
            else:
                inventory["error_tail"] = proc.stderr[-1600:]
        except (OSError, subprocess.TimeoutExpired, ValueError, IndexError) as exc:
            inventory["error"] = type(exc).__name__ + ": " + str(exc)[:1600]
    else:
        inventory["reason"] = "nvidia_smi_not_available"
    report["runtime"]["gpu_inventory"] = inventory
    present = {row["physical_id"] for row in inventory["devices"]}
    add_check(report, "requested_gpu_present", set(requested_gpus) <= present, inventory)


def audit_model(report, repo: Path, model: Path):
    evidence_path = repo / "experiments/care_module2_signed_20261019/results/fold0/provenance.json"
    evidence = {"path": str(evidence_path), "expected_revision": MODEL_REVISION}
    try:
        historical = read_json(evidence_path)
        evidence.update(observed_revision=historical.get("model_revision"), sha256=sha256(evidence_path))
        verified_evidence = historical.get("model_revision") == MODEL_REVISION and historical.get("weights") == WEIGHT_HASHES
    except (OSError, ValueError, AttributeError) as exc:
        verified_evidence = False
        evidence["error"] = type(exc).__name__ + ": " + str(exc)
    add_check(report, "model_revision_evidence", verified_evidence, evidence)

    model_details = {"path": str(model), "model_id": "Qwen/Qwen3-4B-Base", "required_revision": MODEL_REVISION}
    config_path = model / "config.json"
    try:
        config = read_json(config_path)
        actual = {key: config.get(key) for key in EXPECTED_ARCHITECTURE}
        embedded_revision = config.get("_commit_hash")
        config_ok = actual == EXPECTED_ARCHITECTURE and embedded_revision in (None, MODEL_REVISION)
        config_detail = {"path": str(config_path), "sha256": sha256(config_path), "actual": actual,
                         "expected": EXPECTED_ARCHITECTURE, "embedded_revision": embedded_revision,
                         "note": "Architecture alone does not authenticate weights; hashes are checked separately."}
    except (OSError, ValueError, AttributeError) as exc:
        config_ok = False
        config_detail = {"path": str(config_path), "error": type(exc).__name__ + ": " + str(exc)}
    add_check(report, "model_config", config_ok, config_detail)

    index_path = model / "model.safetensors.index.json"
    try:
        index = read_json(index_path)
        shard_names = set(index["weight_map"].values())
        index_ok = shard_names == set(WEIGHT_HASHES)
        index_detail = {"path": str(index_path), "sha256": sha256(index_path), "referenced_shards": sorted(shard_names),
                        "expected_shards": sorted(WEIGHT_HASHES)}
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        index_ok = False
        index_detail = {"path": str(index_path), "error": type(exc).__name__ + ": " + str(exc)}
    add_check(report, "model_weight_index", index_ok, index_detail)

    shards = []
    for name, expected in WEIGHT_HASHES.items():
        path = model / name
        item = {"name": name, "expected_sha256": expected, "present": path.is_file(), "verified": False}
        if item["present"]:
            try:
                item.update(size_bytes=path.stat().st_size, observed_sha256=sha256(path))
                item["verified"] = item["size_bytes"] > 0 and item["observed_sha256"] == expected
            except OSError as exc:
                item["error"] = str(exc)
        shards.append(item)
    add_check(report, "model_weight_hashes", all(x["verified"] for x in shards), shards)
    tokenizers = {name: (model / name).is_file() for name in ("tokenizer.json", "tokenizer_config.json")}
    add_check(report, "tokenizer_files", all(tokenizers.values()), tokenizers)
    model_details["weights_bound_to_frozen_revision"] = verified_evidence and all(x["verified"] for x in shards)
    report["model"] = model_details


def audit_data(report, repo: Path, old_input: Path, new_input: Path):
    data = repo / "experiments/care_module1_v6/data"
    manifest_path = data / "new_set_manifest.json"
    try:
        manifest = read_json(manifest_path)
        manifest_ok = manifest.get("raw_inputs_server_only") == RAW_INPUT_HASHES
        manifest_detail = {"path": str(manifest_path), "sha256": sha256(manifest_path),
                           "recorded_hashes": manifest.get("raw_inputs_server_only"), "expected_hashes": RAW_INPUT_HASHES}
    except (OSError, ValueError, AttributeError) as exc:
        manifest_ok = False
        manifest_detail = {"path": str(manifest_path), "error": type(exc).__name__ + ": " + str(exc)}
    add_check(report, "raw_input_provenance", manifest_ok, manifest_detail)
    raw_results = []
    for name, path, expected_count in (("inputs_192.jsonl", old_input, 192), ("new_inputs_100.jsonl", new_input, 100)):
        item = {"name": name, "path": str(path), "expected_sha256": RAW_INPUT_HASHES[name],
                "expected_rows": expected_count, "present": path.is_file(), "verified": False}
        if item["present"]:
            try:
                item.update(observed_sha256=sha256(path), rows=len(read_jsonl(path)))
                item["verified"] = item["observed_sha256"] == item["expected_sha256"] and item["rows"] == expected_count
            except (OSError, ValueError) as exc:
                item["error"] = type(exc).__name__ + ": " + str(exc)
        add_check(report, "raw_input_" + name.replace(".jsonl", ""), item["verified"], item)
        raw_results.append(item)

    predictions = []
    for name, expected_count in (("old_192_predictions_v6.jsonl", 192), ("new_100_predictions_v6.jsonl", 100)):
        path = data / name
        item = {"path": str(path), "expected_rows": expected_count, "verified": False}
        try:
            rows = read_jsonl(path)
            ids = [row.get("sample_id") for row in rows]
            item.update(rows=len(rows), unique_sample_ids=len(set(ids)), sha256=sha256(path))
            item["verified"] = len(rows) == expected_count and len(set(ids)) == expected_count and None not in ids
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            item["error"] = type(exc).__name__ + ": " + str(exc)
        add_check(report, "frozen_" + name.replace(".jsonl", ""), item["verified"], item)
        predictions.append(item)

    rule_scope = []
    for name in ("rules_v5.jsonl", "new_rules_100.jsonl"):
        path = data / name
        item = {"path": str(path)}
        try:
            rows = read_jsonl(path)
            missing = [row.get("rule_id") for row in rows if not row.get("action")]
            item.update(rows=len(rows), missing_or_empty_action_count=len(missing),
                        rules_without_action=missing, sha256=sha256(path))
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            item["error"] = type(exc).__name__ + ": " + str(exc)
        rule_scope.append(item)
    report["data"] = {"raw_inputs": raw_results, "frozen_predictions": predictions, "rule_action_audit": rule_scope}
    report["scope"] = {
        "complete_new_model_suite_implemented": False,
        "complete_new_model_suite_runnable": False,
        "full_care_efficacy_evaluable_from_current_frozen_data": False,
        "limitations": [
            "The frozen V6 dataset evaluates predicate applicability (T/F/U), not complete clinical treatment actions.",
            "Missing rule actions cannot be supplied by inventing clinical recommendations or reconstructing patients from labels.",
            "Frozen M1 predictions and gold evaluation labels must remain separate; no M1 recomputation is authorized by this audit.",
            "A controlled routing task needs an explicit answer/control-domain contract, rule-family splits, token roles, and held-out scoring.",
            "The source axis needs matched context/prior conflicts with verified closed-book knowledge; such pairs are not supplied by V6 labels.",
            "The determination axis needs known/unknown critical-attribute pairs and answer/clarification targets; safety-refusal labels are not a substitute.",
            "The historical 384-case signed-routing dataset can be recovered from GitHub; it does not supply new M1-driven source/determination or unknown-attribute evaluations.",
            "Passing runtime prerequisites permits limited runtime checks only; it does not imply a complete runnable P0-P6 suite or method efficacy.",
        ],
    }


def parse_gpus(value: str):
    try:
        ids = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated physical integer GPU IDs, for example 0 or 0,1.") from exc
    if not ids or len(set(ids)) != len(ids) or any(x < 0 for x in ids):
        raise argparse.ArgumentTypeError("GPU IDs must be unique non-negative integers.")
    return ids


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--gpu", type=parse_gpus, required=True, help="Physical GPU IDs, e.g. 0 or 0,1; allowed: 0,1,2,3,6,7.")
    parser.add_argument("--old-input", "--old-inputs", dest="old_input", type=Path)
    parser.add_argument("--new-input", "--new-inputs", dest="new_input", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="New JSON file; existing paths are never overwritten.")
    args = parser.parse_args(argv)
    args.repo_root = args.repo_root.expanduser().resolve()
    args.model_path = args.model_path.expanduser().resolve()
    # lexists also rejects a broken symlink. Do not resolve away the final output component.
    out = args.out.expanduser().absolute()
    if os.path.lexists(out):
        parser.error("Refusing to overwrite existing output path: " + str(out))
    default_data = args.repo_root / "experiments/care_module1_v6/data"
    old_input = (args.old_input or default_data / "inputs_192.jsonl").expanduser().resolve()
    new_input = (args.new_input or default_data / "new_inputs_100.jsonl").expanduser().resolve()
    report = {
        "schema": "care.module2.preflight.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "checking", "audit_mode": "offline_read_only_except_explicit_report",
        "repo_root": str(args.repo_root), "reference_snapshot_commit": SNAPSHOT_COMMIT,
        "reference_snapshot_is_not_a_live_git_head_check": True,
        "network_access_attempted": False, "packages_installed": False,
        "model_loaded": False, "inference_performed": False, "module1_modified_or_recomputed": False,
        "output_policy": {"path": str(out), "mode": "exclusive_create", "overwrite_allowed": False,
                          "new_experiment_run_directory_required": True},
        "checks": {}, "runtime_blockers": [],
    }
    add_check(report, "repo_root_present", args.repo_root.is_dir(), {"path": str(args.repo_root)})
    audit_runtime(report, args.gpu)
    audit_model(report, args.repo_root, args.model_path)
    audit_data(report, args.repo_root, old_input, new_input)
    report["runtime_prerequisites_ready"] = not report["runtime_blockers"]
    report["status"] = "ready_for_limited_runtime_checks" if report["runtime_prerequisites_ready"] else "blocked"
    report["efficacy_claim"] = "none"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with out.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except FileExistsError:
        parser.error("Refusing to overwrite output created concurrently: " + str(out))
    print(json.dumps({"status": report["status"], "runtime_blockers": report["runtime_blockers"],
                      "complete_new_model_suite_runnable": False, "out": str(out)}, ensure_ascii=False))
    return 0 if report["runtime_prerequisites_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
