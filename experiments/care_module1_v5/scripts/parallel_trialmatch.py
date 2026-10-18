"""Private deterministic batch-resume for frozen TrialMatchAI inference; no labels.

Commands: prepare builds the original request order and reuses a complete prefix;
worker generates disjoint original batches; merge validates one output per request.
All artifacts contain private patient/model text and belong in a private directory.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import pathlib
import sys
import time
import traceback
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
BASE = "/home/gcx/models/care_module1_v5/phi-4"
ADAPTER = "/home/gcx/models/care_module1_v5/trialmatchai-phi4-reasoning-lora"
TIMING = "shared_batch_wall_time_divided_by_batch_size"


def now():
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path):
    return [json.loads(line) for line in pathlib.Path(path).read_text().splitlines() if line.strip()]


def atomic_json(path, data):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".tmp")
    pending.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    os.chmod(pending, 0o600)
    pending.replace(path)


def write_jsonl(path, rows):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o600)


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def identity(row):
    return tuple(row[key] for key in ("sample_id", "rule_id", "condition_id", "criterion_kind"))


def prepare(args):
    # Avoid importing the package's other modules before a worker needs them.
    from src.module1.care_v5 import _patient_text
    work = pathlib.Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    os.chmod(work, 0o700)
    rules = read_jsonl(args.rules)
    by_rule = {row["rule_id"]: row for row in rules}
    if len(by_rule) != len(rules):
        raise ValueError("duplicate rule IDs")
    requests = []
    for sample in read_jsonl(args.inputs):
        rule = by_rule[sample["rule_id"]]
        patient = _patient_text(sample["sentences"])
        for kind, group in (("inclusion", rule["conditions"]), ("exclusion", rule.get("exceptions", []))):
            for condition in group:
                index = len(requests)
                requests.append({"request_id": index, "original_batch_id": index // args.batch_size,
                    "sample_id": sample["sample_id"], "rule_id": sample["rule_id"],
                    "condition_id": condition["condition_id"], "criterion_kind": kind,
                    "patient": patient, "condition": condition})
    existing = read_jsonl(args.existing)
    if len(existing) > len(requests):
        raise ValueError("existing output longer than request sequence")
    if len(existing) % args.batch_size and len(existing) != len(requests):
        raise ValueError("existing prefix ends within a batch; stop/archive only at full batch boundary")
    reused = []
    for index, row in enumerate(existing):
        if identity(row) != identity(requests[index]):
            raise ValueError(f"existing metadata mismatch at request {index}")
        output = row.get("trialmatch")
        if not isinstance(output, dict) or not isinstance(output.get("raw_output"), str):
            raise ValueError(f"existing raw completion missing at request {index}")
        if output.get("max_new_tokens") != 2048 or output.get("prompt_style") != "official_cot_single_criterion":
            raise ValueError(f"existing prompt/budget mismatch at request {index}")
        if output.get("criterion_kind") != row["criterion_kind"]:
            raise ValueError(f"existing condition kind mismatch at request {index}")
        reused.append({**row, "request_id": index,
            "original_batch_id": requests[index]["original_batch_id"], "source": "reused_original_run"})
    write_jsonl(work / "requests_private.jsonl", requests)
    write_jsonl(work / "reused_private.jsonl", reused)
    assignments = [[] for _ in range(args.world_size)]
    for row in requests[len(existing):]:
        assignments[row["original_batch_id"] % args.world_size].append(row)
    for rank, rows in enumerate(assignments):
        write_jsonl(work / f"assignment_rank{rank}_private.jsonl", rows)
    manifest = {"schema_version": 1, "created_utc": now(), "batch_size": args.batch_size,
        "world_size": args.world_size, "request_count": len(requests), "reused_count": len(reused),
        "remaining_count": len(requests)-len(reused), "resume_batch_id": len(reused)//args.batch_size,
        "inputs_sha256": sha(args.inputs), "rules_sha256": sha(args.rules),
        "existing_sha256": sha(args.existing), "max_new_tokens": 2048,
        "assignment_counts": [len(rows) for rows in assignments],
        "labels_read": False, "assignment": "original_zero_based_batch_id_mod_world_size"}
    atomic_json(work / "manifest_private.json", manifest)
    print(json.dumps({key: manifest[key] for key in ("request_count", "reused_count", "remaining_count", "assignment_counts")}))


def worker(args):
    work = pathlib.Path(args.work_dir)
    manifest = json.loads((work / "manifest_private.json").read_text())
    if not 0 <= args.rank < manifest["world_size"]:
        raise ValueError("rank outside prepared world size")
    assignment = read_jsonl(work / f"assignment_rank{args.rank}_private.jsonl")
    status_path = work / f"worker_rank{args.rank}_status_private.json"
    output_path = work / f"worker_rank{args.rank}_outputs_private.jsonl"
    if output_path.exists() and output_path.stat().st_size:
        raise ValueError("worker output already exists; refusing duplicate inference")
    started = time.perf_counter()
    status = {"status": "loading", "start_utc": now(), "rank": args.rank, "device": args.device,
        "assigned_conditions": len(assignment), "completed_conditions": 0,
        "max_new_tokens": 2048, "batch_size": manifest["batch_size"], "labels_read": False}
    atomic_json(status_path, status)
    try:
        import torch
        from src.module1.care_v5 import TrialMatchAIAdapter
        torch.cuda.set_device(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
        load_start = time.perf_counter()
        model = TrialMatchAIAdapter.from_pretrained(args.device, base_model=args.base_model, adapter_model=args.adapter_model)
        active = getattr(model.model, "active_adapters", None)
        if callable(active):
            active = active()
        if not active:
            raise RuntimeError("required LoRA has no active adapter")
        status.update(load_seconds=time.perf_counter()-load_start, active_adapters=list(active) if not isinstance(active,str) else [active],
            parameter_count=sum(parameter.numel() for parameter in model.model.parameters()),
            trainable_parameter_count=sum(parameter.numel() for parameter in model.model.parameters() if parameter.requires_grad),
            gpu_name=torch.cuda.get_device_name(args.device), status="running")
        atomic_json(status_path, status)
        groups = {}
        for row in assignment:
            bid = row["original_batch_id"]
            if bid % manifest["world_size"] != args.rank:
                raise ValueError("batch assigned to incorrect rank")
            groups.setdefault(bid, []).append(row)
        with output_path.open("w") as handle:
            os.chmod(output_path, 0o600)
            for bid, rows in sorted(groups.items()):
                expected = min(manifest["batch_size"], manifest["request_count"] - bid * manifest["batch_size"])
                if len(rows) != expected or [row["request_id"] for row in rows] != list(range(bid*manifest["batch_size"],bid*manifest["batch_size"]+expected)):
                    raise ValueError("original full batch grouping was altered")
                outputs = model.judge_many(rows, batch_size=manifest["batch_size"])
                for row, output in zip(rows, outputs):
                    output["original_batch_id"] = bid
                    output["worker_rank"] = args.rank
                    record = {key: row[key] for key in ("request_id", "original_batch_id", "sample_id", "rule_id", "condition_id", "criterion_kind")}
                    record.update(trialmatch=output, source="parallel_remaining_run", worker_rank=args.rank)
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                status.update(completed_conditions=status["completed_conditions"]+len(rows), last_original_batch_id=bid,
                    elapsed_seconds=time.perf_counter()-started)
                atomic_json(status_path, status)
        status["status"] = "completed"
    except Exception as exc:
        status.update(status="failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    finally:
        if "torch" in locals() and torch.cuda.is_available():
            status.update(peak_allocated_gib=torch.cuda.max_memory_allocated(args.device)/1024**3,
                peak_reserved_gib=torch.cuda.max_memory_reserved(args.device)/1024**3)
        status.update(end_utc=now(), elapsed_seconds=time.perf_counter()-started)
        atomic_json(status_path, status)
    if status["status"] != "completed":
        raise SystemExit(2)


def merge(args):
    work = pathlib.Path(args.work_dir)
    manifest = json.loads((work / "manifest_private.json").read_text())
    requests = read_jsonl(work / "requests_private.jsonl")
    all_rows = read_jsonl(work / "reused_private.jsonl")
    worker_status = []
    for rank in range(manifest["world_size"]):
        status = json.loads((work / f"worker_rank{rank}_status_private.json").read_text())
        if status["status"] != "completed":
            raise ValueError(f"worker {rank} has not completed")
        worker_status.append(status)
        all_rows.extend(read_jsonl(work / f"worker_rank{rank}_outputs_private.jsonl"))
    by_id = {}
    for row in all_rows:
        index = row["request_id"]
        if not isinstance(index,int) or not 0 <= index < len(requests) or index in by_id:
            raise ValueError("duplicate or invalid request_id")
        if identity(row) != identity(requests[index]) or row["original_batch_id"] != requests[index]["original_batch_id"]:
            raise ValueError(f"merged identity mismatch at request {index}")
        if row["trialmatch"].get("max_new_tokens") != 2048:
            raise ValueError("merged output budget mismatch")
        by_id[index] = row
    if set(by_id) != set(range(len(requests))):
        raise ValueError("merged outputs have missing requests")
    ordered = [by_id[index] for index in range(len(requests))]
    write_jsonl(args.output, ordered)
    batch_times = {}
    for row in ordered:
        bid = row["original_batch_id"]
        elapsed = row["trialmatch"].get("batch_elapsed_seconds",0.0)
        if bid in batch_times and abs(batch_times[bid]-elapsed) > 1e-6:
            raise ValueError(f"inconsistent time within original batch {bid}")
        batch_times[bid] = elapsed
    summary = {"status": "completed", "merged_utc": now(), "request_count": len(ordered),
        "reused_count": manifest["reused_count"], "new_conditions": len(ordered)-manifest["reused_count"],
        "original_batch_count": len(batch_times), "sum_batch_active_seconds": sum(batch_times.values()),
        "allocated_condition_seconds": sum(row["trialmatch"].get("inference_seconds",0) for row in ordered),
        "timing_method": TIMING, "parallel_workers": worker_status, "output_sha256": sha(args.output),
        "labels_read": False}
    atomic_json(work / "timing_summary_private.json", summary)
    print(json.dumps({key: summary[key] for key in ("status","request_count","reused_count","new_conditions")}))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--inputs", required=True); p.add_argument("--rules", required=True)
    p.add_argument("--existing", required=True); p.add_argument("--work-dir", required=True)
    p.add_argument("--batch-size", type=int, default=8); p.add_argument("--world-size", type=int, default=4)
    p.set_defaults(function=prepare)
    p = sub.add_parser("worker")
    p.add_argument("--work-dir", required=True); p.add_argument("--rank",type=int,required=True)
    p.add_argument("--device",required=True); p.add_argument("--base-model",default=BASE)
    p.add_argument("--adapter-model",default=ADAPTER); p.set_defaults(function=worker)
    p = sub.add_parser("merge")
    p.add_argument("--work-dir",required=True); p.add_argument("--output",required=True)
    p.set_defaults(function=merge)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
