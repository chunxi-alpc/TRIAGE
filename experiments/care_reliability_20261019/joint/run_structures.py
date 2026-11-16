#!/usr/bin/env python3
"""Generate frozen V6/V7 structures in the original Phi-4 runtime.

Exact TrialMatchAI requests are generated once and replayed into both versions.
The cache never reads reference labels; missing requests fail closed.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

EXPERIMENT = Path(__file__).resolve().parents[1]
REPOSITORY = EXPERIMENT.parents[1]
sys.path[:0] = [str(REPOSITORY), str(EXPERIMENT)]
from joint.context import prediction_input, prompt_sha256
from joint.evaluate import read_jsonl
from joint.structures import normalize_structure


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def file_digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def implementation_hashes():
    paths = sorted((REPOSITORY / "src/module1").glob("*.py"))
    paths += [Path(__file__), EXPERIMENT / "joint/context.py", EXPERIMENT / "joint/structures.py"]
    return {str(path.relative_to(REPOSITORY)): file_digest(path) for path in paths}


def request_key(patient, condition, criterion_kind="inclusion"):
    # These are precisely the three fields read by TrialMatchAIAdapter._messages.
    kind = "exclusion" if criterion_kind == "exclusion" else "inclusion"
    return digest({"patient": patient, "criterion_text": condition["text"], "criterion_kind": kind})


def collect_requests(cases):
    from src.module1.care_v6 import _patient_text, _rule_list
    requests = {}
    logical_calls = 0
    for case in cases:
        inputs = prediction_input(case)
        patient = _patient_text(inputs["patient"])
        for rule in _rule_list(copy.deepcopy(inputs["guideline"])):
            for role, default_kind in (("conditions", "inclusion"), ("exceptions", "exclusion")):
                for atom in rule[role]:
                    kind = atom.get("criterion_kind", default_kind)
                    key = request_key(patient, atom, kind)
                    if key not in requests:
                        requests[key] = {"request_key": key, "patient": patient,
                                         "condition": {"text": atom["text"]}, "criterion_kind": kind,
                                         "uses": []}
                    requests[key]["uses"].append({"case_id": case["case_id"], "rule_id": rule["rule_id"],
                                                 "condition_id": atom["condition_id"], "role": role,
                                                 "versions": ["v6", "v7"]})
                    logical_calls += 2
    return list(requests.values()), logical_calls


class StrictReplay:
    def __init__(self, rows):
        self.rows = rows
        self.calls = 0

    def judge(self, patient, condition, criterion_kind="inclusion"):
        key = request_key(patient, condition, criterion_kind)
        if key not in self.rows:
            raise KeyError("Unprepared TrialMatchAI request: " + key)
        self.calls += 1
        return copy.deepcopy(self.rows[key]["output"])


def model_manifest(path):
    path = Path(path).resolve()
    if not path.is_dir():
        raise ValueError("Existing local model required: " + str(path))
    names = ("*.json", "*.safetensors", "pytorch_model.bin", "vocab.txt", "*.txt")
    paths = {p for pattern in names for p in path.glob(pattern) if p.is_file()}
    return {"path": str(path), "file_sha256": {p.name: file_digest(p) for p in sorted(paths)}}


def package_versions():
    result = {}
    for name in ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "tokenizers"):
        result[name] = importlib.metadata.version(name)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=EXPERIMENT / "data/mechanism_cases.jsonl")
    parser.add_argument("--split", choices=("train", "confirmation"), default="confirmation")
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-model", required=True)
    parser.add_argument("--evidence-model", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.case_limit is not None and (args.split != "train" or args.case_limit < 1):
        parser.error("A positive case-limit is allowed only for training throughput checks")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count or args.batch_size < 1:
        parser.error("Invalid shard or batch configuration")
    cases = [c for c in read_jsonl(args.cases) if c["split"] == args.split]
    if args.case_limit:
        cases = cases[:args.case_limit]
    cases = [c for i, c in enumerate(cases) if i % args.shard_count == args.shard_index]
    if not cases:
        parser.error("No cases selected")
    frozen = json.loads((args.cases.parent / "manifest.json").read_text())
    if frozen["files"][args.cases.name]["sha256"] != file_digest(args.cases):
        raise ValueError("Input data differ from frozen manifest")
    requests, logical_calls = collect_requests(cases)
    contract = {"data_sha256": file_digest(args.cases), "implementation_sha256": implementation_hashes(),
                "split": args.split, "case_ids": [c["case_id"] for c in cases],
                "shard_count": args.shard_count, "shard_index": args.shard_index,
                "generation": {"batch_size": args.batch_size, "max_new_tokens": 2048, "do_sample": False},
                "request_manifest_sha256": digest(requests),
                "model_paths": {k: str(Path(getattr(args, k)).resolve()) for k in
                                ("base_model", "adapter_model", "evidence_model")}}
    if args.resume:
        prior = json.loads((args.output / "run_manifest.json").read_text())
        if prior["contract"] != contract:
            raise ValueError("Resume rejected: input, source, model path, or generation contract changed")
        if prior["status"] == "completed":
            raise ValueError("Completed runs are immutable")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    status = {"status": "prepared" if args.dry_run else "initializing", "contract": contract,
              "cases": len(cases), "rules_per_version": sum(len(c["rules"]) for c in cases),
              "logical_trialmatch_calls_two_versions": logical_calls,
              "unique_trialmatch_requests": len(requests), "reference_labels_in_model_inputs": False,
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "device": args.device,
              "python_executable": sys.executable, "started_utc": datetime.now(timezone.utc).isoformat(),
              "request_reuse": "exact patient, criterion text, and criterion kind; both versions",
              "batching_note": "same existing judge_many implementation; batching can alter floating point rounding"}
    if args.dry_run:
        save(args.output / "run_manifest.json", status)
        save(args.output / "request_manifest.json", requests)
        print(json.dumps({k: status[k] for k in ("cases", "unique_trialmatch_requests", "logical_trialmatch_calls_two_versions")}))
        return 0
    started = time.perf_counter()
    # Validate immutable provenance before changing any run file on resume.
    provenance = {"contract": contract, "versions": package_versions(),
                  "models": {k: model_manifest(getattr(args, k)) for k in
                             ("base_model", "adapter_model", "evidence_model")}}
    provenance_path = args.output / "immutable_provenance.json"
    if args.resume:
        if json.loads(provenance_path.read_text()) != provenance:
            raise ValueError("Resume rejected: model bytes, original runtime, or immutable contract changed")
        if digest(json.loads((args.output / "request_manifest.json").read_text())) != contract["request_manifest_sha256"]:
            raise ValueError("Resume rejected: request manifest changed")
        for key in ("active_adapters", "lora_parameter_count"):
            if key in prior:
                status[key] = prior[key]
    else:
        save(provenance_path, provenance)
        save(args.output / "request_manifest.json", requests)
    status.update(versions=provenance["versions"], models=provenance["models"],
                  immutable_provenance_sha256=file_digest(provenance_path))
    save(args.output / "run_manifest.json", status)
    try:
        import torch
        from src.module1.care_v6 import TrialMatchAIAdapter, BioLinkBERTMedNLI, CAREModule1V6
        from src.module1.care_v7 import CAREModule1V7
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no CPU fallback")
        torch.cuda.set_device(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
        status["gpu_name"] = torch.cuda.get_device_name(args.device)
        cache = {}
        journal = args.output / "trialmatch_generations.jsonl"
        request_keys = {r["request_key"] for r in requests}
        batches = [(start, requests[start:start + args.batch_size])
                   for start in range(0, len(requests), args.batch_size)]
        missing_batches = []
        for batch_start, batch in batches:
            batch_path = args.output / "generation_batches" / f"batch_{batch_start:06d}.json"
            if not batch_path.exists():
                missing_batches.append((batch_start, batch))
                continue
            saved = json.loads(batch_path.read_text())
            expected_keys = [r["request_key"] for r in batch]
            if (saved["request_keys"] != expected_keys or len(saved["outputs"]) != len(batch)
                    or saved["immutable_provenance_sha256"] != status["immutable_provenance_sha256"]):
                raise ValueError("Committed generation batch provenance or coverage differs")
            for key, output in zip(expected_keys, saved["outputs"]):
                cache[key] = {"request_key": key, "output": output, "execution_batch_start": batch_start}
        status["reused_completed_requests"] = len(cache)
        # Atomic complete-batch records are authoritative. Rebuild the readable
        # journal after an interrupted write without changing batch membership.
        if args.resume and journal.exists():
            journal.rename(args.output / ("interrupted_journal_" + str(time.time_ns()) + ".jsonl"))
        with journal.open("w", buffering=1) as stream:
            for request in requests:
                if request["request_key"] in cache:
                    stream.write(json.dumps(cache[request["request_key"]], ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            if missing_batches:
                adapter = TrialMatchAIAdapter.from_pretrained(args.device, base_model=args.base_model,
                             adapter_model=args.adapter_model)
                status["active_adapters"] = list(adapter.model.active_adapters)
                status["lora_parameter_count"] = sum(p.numel() for n, p in adapter.model.named_parameters() if ".lora_" in n)
                status["status"] = "generating_trialmatch"
                save(args.output / "run_manifest.json", status)
                for batch_start, batch in missing_batches:
                    outputs = adapter.judge_many(batch, batch_size=args.batch_size)
                    if len(outputs) != len(batch):
                        raise RuntimeError("TrialMatchAI returned an incomplete generation batch")
                    batch_path = args.output / "generation_batches" / f"batch_{batch_start:06d}.json"
                    save(batch_path, {"request_keys": [r["request_key"] for r in batch], "outputs": outputs,
                                      "immutable_provenance_sha256": status["immutable_provenance_sha256"]})
                    for request, output in zip(batch, outputs):
                        row = {"request_key": request["request_key"], "output": output,
                               "execution_batch_start": batch_start}
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                        cache[row["request_key"]] = row
                    stream.flush()
                    os.fsync(stream.fileno())
                    status.update(completed_trialmatch_requests=len(cache),
                                  elapsed_seconds=time.perf_counter()-started)
                    save(args.output / "run_manifest.json", status)
                del adapter
                torch.cuda.empty_cache()
        if set(cache) != request_keys:
            raise ValueError("Generation cache coverage is incomplete")
        replay = StrictReplay(cache)
        nli = BioLinkBERTMedNLI.from_pretrained(args.device, model_name=args.evidence_model)
        engines = {"v6": CAREModule1V6(replay, nli), "v7": CAREModule1V7(replay, nli)}
        status["status"] = "evaluating_structures"
        save(args.output / "run_manifest.json", status)
        structure_index = args.output / "structure_records.json"
        structure_hashes = json.loads(structure_index.read_text()) if structure_index.exists() else {}
        for source, engine in engines.items():
            for case in cases:
                inputs = prediction_input(case)
                raw = engine.run(**inputs)
                if any(a.get("technical_error") for r in raw["rules"] for a in r["conditions"] + r["exceptions"]):
                    raise RuntimeError("A structure contains a technical error; refusing silent cache failure")
                value = {"case_id": case["case_id"],
                         "input_sha256": prompt_sha256(json.dumps(inputs, sort_keys=True)),
                         "normalized": normalize_structure(raw, case, source), "raw": raw}
                relative = Path("structures") / source / (case["case_id"] + ".json")
                target = args.output / relative
                # Resume replays a deterministic model cache; preserve any complete structure.
                if target.exists():
                    if structure_hashes.get(str(relative)) != file_digest(target):
                        raise ValueError("Existing structure differs from its committed digest")
                    previous = json.loads(target.read_text())
                    if previous["input_sha256"] != value["input_sha256"]:
                        raise ValueError("Existing structure input digest differs")
                    value = previous
                else:
                    save(target, value)
                structure_hashes[str(relative)] = file_digest(target)
                save(structure_index, structure_hashes)
        if replay.calls != logical_calls:
            raise ValueError("Replay calls differ from planned logical calls")
        status.update(status="completed", completed_trialmatch_requests=len(cache),
                      structure_sha256=structure_hashes, actual_replayed_trialmatch_calls=replay.calls,
                      generation_journal_sha256=file_digest(journal),
                      elapsed_seconds=time.perf_counter()-started,
                      peak_allocated_gib=torch.cuda.max_memory_allocated(args.device) / 1024**3,
                      peak_reserved_gib=torch.cuda.max_memory_reserved(args.device) / 1024**3,
                      ended_utc=datetime.now(timezone.utc).isoformat())
        save(args.output / "run_manifest.json", status)
        save(args.output / "COMPLETE.json", {"manifest_sha256": file_digest(args.output / "run_manifest.json")})
        print(json.dumps({k: status[k] for k in ("status", "cases", "unique_trialmatch_requests", "elapsed_seconds", "peak_reserved_gib")}))
    except Exception as error:
        status.update(status="failed", error=type(error).__name__ + ": " + str(error),
                      elapsed_seconds=time.perf_counter()-started)
        save(args.output / "run_manifest.json", status)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

