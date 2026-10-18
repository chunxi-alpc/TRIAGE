"""Run the fixed 192-row CARE V5 exploratory regression on the GPU server."""
from __future__ import annotations
import argparse, collections, hashlib, json, os, pathlib, sys, time, traceback
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.module1.care_v5 import CAREModule1V5, TrialMatchAIAdapter, BioLinkBERTMedNLI, _patient_text


def load_jsonl(path):
    return [json.loads(line) for line in pathlib.Path(path).read_text().splitlines() if line.strip()]


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def public_span(row):
    return {k: row[k] for k in ("sentence_id", "start", "end", "nli_relation", "class_id", "scores", "evidence_source") if k in row}



class PrecomputedTrialMatch:
    """Reuse each frozen batch output exactly once in the engine's original order."""
    def __init__(self, requests, outputs):
        if len(requests) != len(outputs):
            raise ValueError("precomputed request/output count mismatch")
        self.cache = collections.defaultdict(collections.deque)
        self.remaining = len(outputs)
        for request, output in zip(requests, outputs):
            self.cache[self._key(request["patient"], request["condition"], request["criterion_kind"])].append(output)

    @staticmethod
    def _key(patient, condition, criterion_kind):
        # Only these fields enter TrialMatchAI's prompt; no test labels enter this cache.
        return patient, condition["text"], criterion_kind

    def judge(self, patient, condition, criterion_kind="inclusion"):
        key = self._key(patient, condition, criterion_kind)
        if not self.cache.get(key):
            raise RuntimeError("precomputed TrialMatchAI request missing or already consumed; no fresh inference fallback")
        self.remaining -= 1
        return dict(self.cache[key].popleft())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", default="/home/gcx/care_module1_20261018_v4/data/private/inputs.jsonl")
    parser.add_argument("--rules", default=str(ROOT / "experiments/care_module1_v5/data/rules_v5.jsonl"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", default="/home/gcx/.cache/huggingface")
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--base-model", default="/home/gcx/models/care_module1_v5/phi-4")
    parser.add_argument("--adapter-model", default="/home/gcx/models/care_module1_v5/trialmatchai-phi4-reasoning-lora")
    parser.add_argument("--evidence-model", default="/home/gcx/models/care_module1_v5/biolinkbert-mednli")
    parser.add_argument("--nli-batch-size", type=int, default=32)
    parser.add_argument("--trialmatch-batch-size", type=int, default=8)
    parser.add_argument("--expected-count", type=int, default=192)
    args = parser.parse_args()
    out = pathlib.Path(args.output)
    private_dir = out / "private"
    private_dir.mkdir(parents=True, exist_ok=False)
    out.mkdir(parents=True, exist_ok=True)
    experiment_start = time.perf_counter()
    status = {"status": "starting", "start_utc": datetime.now(timezone.utc).isoformat(),
              "device": args.device, "inputs": str(args.inputs), "sample_count": args.expected_count,
              "models": ["microsoft/phi-4 + majdabd33/trialmatchai-phi4-reasoning-lora", "cnut1648/biolinkbert-mednli"]}
    (out / "run_status.json").write_text(json.dumps(status, indent=2) + "\n")
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; refusing CPU fallback")
        torch.manual_seed(20261018)
        torch.cuda.manual_seed_all(20261018)
        gpu_index = int(args.device.split(":", 1)[1])
        torch.cuda.set_device(gpu_index)
        torch.cuda.reset_peak_memory_stats(gpu_index)
        status["gpu_name"] = torch.cuda.get_device_name(gpu_index)
        status["gpu_total_memory_gib"] = round(torch.cuda.get_device_properties(gpu_index).total_memory / 1024**3, 3)
        status["versions"] = {"python": sys.version.split()[0], "torch": torch.__version__}
        t0 = time.perf_counter()
        trialmatch = TrialMatchAIAdapter.from_pretrained(args.device, cache_dir=args.cache,
            base_model=args.base_model, adapter_model=args.adapter_model)
        status["trialmatch_load_seconds"] = round(time.perf_counter() - t0, 3)
        status["trialmatch_active_adapters"] = list(trialmatch.model.active_adapters)
        status["trialmatch_lora_parameter_count"] = sum(p.numel() for name,p in trialmatch.model.named_parameters() if ".lora_" in name)
        status["versions"]["transformers"] = __import__("transformers").__version__
        status["versions"]["peft"] = __import__("peft").__version__
        status["versions"]["bitsandbytes"] = __import__("bitsandbytes").__version__
        t0 = time.perf_counter()
        mednli = BioLinkBERTMedNLI.from_pretrained(args.device, cache_dir=args.cache,
            model_name=args.evidence_model)
        status["mednli_load_seconds"] = round(time.perf_counter() - t0, 3)
        status["mednli_id2label"] = {str(k): v for k, v in mednli.id2label.items()}
        inputs, rules = load_jsonl(args.inputs), load_jsonl(args.rules)
        rule_by_id = {row["rule_id"]: row for row in rules}
        if len(inputs) != args.expected_count:
            raise ValueError(f"Expected frozen sample count {args.expected_count}, found {len(inputs)}")
        requests, request_metadata = [], []
        for sample in inputs:
            rule = rule_by_id[sample["rule_id"]]
            patient = _patient_text(sample["sentences"])
            for kind, group in (("inclusion", rule["conditions"]), ("exclusion", rule.get("exceptions", []))):
                for condition in group:
                    requests.append({"patient": patient, "condition": condition, "criterion_kind": kind})
                    request_metadata.append({"sample_id": sample["sample_id"], "rule_id": sample["rule_id"],
                        "condition_id": condition["condition_id"], "criterion_kind": kind})
        status.update(status="precomputing_trialmatch", trialmatch_condition_count=len(requests),
            trialmatch_batch_size=args.trialmatch_batch_size, trialmatch_max_new_tokens=trialmatch.MAX_NEW_TOKENS,
            trialmatch_timing_method="shared_batch_wall_time_divided_by_batch_size")
        (out / "run_status.json").write_text(json.dumps(status, indent=2) + "\n")
        t0 = time.perf_counter()
        outputs = []
        # Persist every completed batch so an interrupted later batch cannot erase
        # already generated raw completions. No labels are read by this process.
        with (private_dir / "trialmatch_precomputed.jsonl").open("w") as handle:
            for start in range(0, len(requests), args.trialmatch_batch_size):
                batch = trialmatch.judge_many(requests[start:start + args.trialmatch_batch_size],
                    batch_size=args.trialmatch_batch_size)
                for metadata, output in zip(request_metadata[start:start + len(batch)], batch):
                    output["generation_batch_id"] = start // args.trialmatch_batch_size + 1
                    handle.write(json.dumps({**metadata, "trialmatch": output}, ensure_ascii=False) + "\n")
                handle.flush(); os.fsync(handle.fileno())
                outputs.extend(batch)
                status["trialmatch_conditions_completed"] = len(outputs)
                status["trialmatch_elapsed_seconds"] = round(time.perf_counter() - t0, 3)
                (out / "run_status.json").write_text(json.dumps(status, indent=2) + "\n")
                print(json.dumps({"conditions_completed": len(outputs), "total": len(requests)}), flush=True)
        status["trialmatch_precompute_seconds"] = time.perf_counter() - t0
        status["trialmatch_shared_generation_seconds"] = sum(row["inference_seconds"] for row in outputs)
        status["trialmatch_parse_errors"] = dict(collections.Counter(row["parse_error"] for row in outputs if row.get("parse_error")))
        status["trialmatch_truncated_conditions"] = sum(bool(row.get("truncated")) for row in outputs)
        status["trialmatch_output_tokens"] = sum(row["output_tokens"] for row in outputs)
        cached_trialmatch = PrecomputedTrialMatch(requests, outputs)
        engine = CAREModule1V5(cached_trialmatch, mednli, nli_batch_size=args.nli_batch_size)
        status["status"] = "evaluating_evidence"
        (out / "run_status.json").write_text(json.dumps(status, indent=2) + "\n")
        private_path, public_path = private_dir / "predictions_private.jsonl", out / "predictions_public.jsonl"
        with private_path.open("w") as private_file, public_path.open("w") as public_file:
            for number, sample in enumerate(inputs, 1):
                started = time.perf_counter()
                rule = rule_by_id[sample["rule_id"]]
                graph = engine.run([rule], sample["sentences"], question=rule["text"])
                prediction = graph["rules"][0]
                condition_private = []
                condition_public = []
                for group in (prediction["conditions"], prediction["exceptions"]):
                    for condition in group:
                        completion = condition["trialmatch"]["raw_output"]
                        condition["trialmatch_cache_lookup_seconds"] = condition["trialmatch_seconds"]
                        condition["trialmatch_seconds"] = condition["trialmatch"]["inference_seconds"]
                        condition["trialmatch_timing_method"] = condition["trialmatch"]["timing_method"]
                        condition_private.append({"condition_id": condition["condition_id"],
                            "raw_trialmatch_output": completion,
                            "nli_candidates": condition["nli_candidates"]})
                        spans = condition["support_evidence"] + condition["refute_evidence"] + condition["unknown_evidence"]
                        condition_public.append({"condition_id": condition["condition_id"],
                            "trialmatch_raw_label": condition["trialmatch"].get("raw_label"),
                            "trialmatch_state": condition["trialmatch"]["state"],
                            "trialmatch_parse_error": condition["trialmatch"].get("parse_error"),
                            "trialmatch_truncated": condition["trialmatch"]["truncated"],
                            "trialmatch_input_tokens": condition["trialmatch"]["input_tokens"],
                            "trialmatch_output_tokens": condition["trialmatch"]["output_tokens"],
                            "trialmatch_generation_batch_id": condition["trialmatch"].get("generation_batch_id"),
                            "trialmatch_generation_batch_size": condition["trialmatch"]["generation_batch_size"],
                            "trialmatch_timing_method": condition["trialmatch_timing_method"],
                            "trialmatch_cache_lookup_seconds": condition["trialmatch_cache_lookup_seconds"],
                            "trialmatch_seconds": condition["trialmatch_seconds"],
                            "numeric_seconds": condition["numeric_seconds"],
                            "evidence_selection_seconds": condition["evidence_selection_seconds"],
                            "mednli_seconds": condition["mednli_seconds"],
                            "trialmatch_output_sha256": sha(completion),
                            "state": condition["status"], "nli_evidence_state": condition["evidence_state"],
                            "resolution": condition["resolution"], "conflict": condition["conflict"],
                            "support_spans": [public_span(row) for row in condition["support_evidence"]],
                            "refute_spans": [public_span(row) for row in condition["refute_evidence"]],
                            "unknown_spans": [public_span(row) for row in condition["unknown_evidence"]],
                            "nli_candidate_scores": [{"sentence_id": row["sentence_id"], "start": row["start"],
                                "end": row["end"], "relation": row["nli_relation"], "class_id": row["class_id"],
                                "scores": row["scores"], "skipped_reason": row.get("skipped_reason")}
                                for row in condition["nli_candidates"]]})
                        numeric = condition.get("numeric") or {}
                        condition_public[-1]["numeric_status"] = numeric.get("status", "U")
                        condition_public[-1]["numeric_evidence_spans"] = [
                            {"sentence_id": span["sentence_id"], "start": span["start"], "end": span["end"]}
                            for span in numeric.get("evidence", [])]
                postprocess_seconds = time.perf_counter() - started
                shared_trialmatch_seconds = sum(c["trialmatch_seconds"] for c in condition_public)
                graph["metadata"].update(
                    trialmatch_timing_method="shared_batch_wall_time_divided_by_batch_size",
                    postprocess_seconds=postprocess_seconds, trialmatch_seconds=shared_trialmatch_seconds,
                    elapsed_seconds=postprocess_seconds + shared_trialmatch_seconds)
                private_row = {"sample_id": sample["sample_id"], "rule_id": sample["rule_id"],
                    "patient_sentences": sample["sentences"], "trialmatch_and_evidence": condition_private,
                    "module1_output": graph}
                public_row = {"sample_id": sample["sample_id"], "rule_id": sample["rule_id"],
                    "trialmatch_applicability": prediction["trialmatch_applicability"],
                    "trialmatch_model_only_applicability": prediction["trialmatch_model_only_applicability"],
                    "applicability": prediction["applicability"], "uncertain": prediction["uncertain"],
                    "conditions": condition_public,
                    "elapsed_seconds": round(postprocess_seconds + shared_trialmatch_seconds, 6),
                    "postprocess_seconds": postprocess_seconds,
                    "timing_method": "postprocess_wall_time_plus_allocated_shared_trialmatch_batch_time",
                    "trialmatch_seconds": shared_trialmatch_seconds,
                    "numeric_seconds": sum(c["numeric_seconds"] for c in condition_public),
                    "evidence_selection_seconds": sum(c["evidence_selection_seconds"] for c in condition_public),
                    "mednli_seconds": sum(c["mednli_seconds"] for c in condition_public)}
                private_file.write(json.dumps(private_row, ensure_ascii=False) + "\n")
                public_file.write(json.dumps(public_row, ensure_ascii=False) + "\n")
                if number % 8 == 0:
                    private_file.flush(); public_file.flush(); os.fsync(private_file.fileno()); os.fsync(public_file.fileno())
                    print(json.dumps({"completed": number, "total": len(inputs)}, flush=True))
        if cached_trialmatch.remaining:
            raise RuntimeError(f"{cached_trialmatch.remaining} precomputed TrialMatchAI outputs were not consumed")
        status["status"] = "completed"
        status["sample_count_completed"] = len(inputs)
        status["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated(gpu_index) / 1024**3, 3)
        status["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved(gpu_index) / 1024**3, 3)
    except Exception as exc:
        status["status"] = "failed"
        status["error_type"] = type(exc).__name__
        status["error"] = str(exc)[:2000]
        status["traceback"] = traceback.format_exc(limit=12)
    finally:
        status["end_utc"] = datetime.now(timezone.utc).isoformat()
        status["elapsed_seconds"] = round(time.perf_counter() - experiment_start, 3)
        (out / "run_status.json").write_text(json.dumps(status, indent=2) + "\n")
        print(json.dumps({k: status.get(k) for k in ("status", "gpu_name", "error_type", "error", "sample_count_completed")}, ensure_ascii=False))
    if status["status"] != "completed":
        raise SystemExit(2)

if __name__ == "__main__":
    main()
