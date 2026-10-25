#!/usr/bin/env python3
"""Executable no-tool MedCalc native baseline; no clinical attention claim."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

from medcalc_native import OfficialMedCalc, digest, method_input, predict
from medcalc_data import read_rows, sha_file, synthetic_rows


def hf_generator(model_path, config, seed):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if torch.cuda.device_count() != 1:
        raise ValueError("Expose one coordinator-assigned GPU")
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, torch_dtype="auto",
                                                attn_implementation="sdpa").to("cuda").eval()
    def generate(messages):
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False, truncation=False).to(model.device)
        length = int(encoded["input_ids"].shape[1])
        budget = config["max_new_tokens"]
        if length + budget > min(config["context_limit"], model.config.max_position_embeddings):
            raise ValueError("Complete native input exceeds fixed budget; no truncation")
        calls = [0]
        def counter(*unused):
            calls[0] += 1
        hook = model.register_forward_hook(counter)
        torch.cuda.reset_peak_memory_stats(model.device)
        torch.cuda.synchronize(model.device)
        start = time.perf_counter()
        result = {"status": "error", "text": "", "input_tokens": length}
        try:
            with torch.inference_mode():
                output = model.generate(**encoded, do_sample=False, max_new_tokens=budget,
                    pad_token_id=tokenizer.eos_token_id, temperature=None, top_k=None, top_p=None)
            tokens = output[0, length:].tolist()
            result.update(status="ok", text=tokenizer.decode(tokens, skip_special_tokens=True), output_tokens=len(tokens),
                          token_ids_sha256=digest(tokens))
        except Exception as error:
            result.update(error_type=type(error).__name__, error=str(error))
        finally:
            hook.remove()
        torch.cuda.synchronize(model.device)
        result.update(actual_forward_calls=calls[0], elapsed_seconds=time.perf_counter() - start,
                      peak_allocated_gib=torch.cuda.max_memory_allocated(model.device) / 2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved(model.device) / 2**30)
        return result
    return generate, {"model_config": model.config.to_dict(), "chat_template_sha256": digest(tokenizer.chat_template),
                       "dtype": str(next(model.parameters()).dtype), "greedy": True, "thinking": False,
                       "generation_seed": seed, "greedy_duplicate_seeds_not_independent": True}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--config-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["synthetic", "train", "test"], required=True)
    parser.add_argument("--engine", choices=["fake", "hf"], default="fake")
    parser.add_argument("--model")
    parser.add_argument("--model-fingerprint")
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], default=0)
    args = parser.parse_args()
    lock = json.loads(args.config_lock.read_text())
    official = OfficialMedCalc(args.official_root)
    style = lock["prompt_style"]
    if style not in {"direct_answer", "zero_shot"}:
        raise ValueError("Unknown native prompt style")
    if args.engine == "fake" and args.split != "synthetic":
        raise ValueError("Fake outputs are only allowed for the built-in synthetic cases")
    if args.split == "test":
        sources = {Path(__file__).name: sha_file(__file__), "medcalc_native.py": sha_file(Path(__file__).with_name("medcalc_native.py")),
                   "medcalc_data.py": sha_file(Path(__file__).with_name("medcalc_data.py"))}
        if (lock.get("source_hashes") != sources or not lock.get("model_fingerprint") or
                lock["model_fingerprint"] != args.model_fingerprint):
            raise ValueError("Freeze the actual method source and model before test generation")
    if args.split == "synthetic":
        rows, data_sha = synthetic_rows(), None
    else:
        rows, data_sha = read_rows(args.official_root, args.split, lock)
    os.umask(0o077)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    run = {"status": "running", "split": args.split, "seed": args.seed, "expected_rows": len(rows),
           "config_lock_sha256": sha_file(args.config_lock), "data_sha256": data_sha,
           "official_function_sources": official.source_hashes,
           "native_Qwen_adaptation": True, "arithmetic_tools_provided_to_model": False}
    manifest = args.output_dir / "run.json"
    manifest.write_text(json.dumps(run, indent=2))
    if args.engine == "fake":
        def generate(messages):
            return {"text": '{"answer": "5"}', "status": "ok", "actual_forward_calls": 0,
                    "input_tokens": 0, "output_tokens": 0, "elapsed_seconds": 0., "peak_allocated_gib": 0.,
                    "peak_reserved_gib": 0., "backend": "synthetic_no_model"}
        runtime = {"model_inference": False}
    else:
        if not args.model:
            raise ValueError("Supply the coordinator's existing local Qwen checkpoint")
        try:
            generate, runtime = hf_generator(args.model, lock["generation"], args.seed)
        except Exception as error:
            run.update(status="failed", error_type=type(error).__name__, error_sha256=digest(str(error)))
            manifest.write_text(json.dumps(run, indent=2))
            raise
        runtime.update(model_path=args.model, model_fingerprint=args.model_fingerprint)
    run["runtime"] = runtime
    manifest.write_text(json.dumps(run, indent=2))
    statuses = Counter()
    with (args.output_dir / "predictions.jsonl").open("x") as public, (args.output_dir / "predictions.private.jsonl").open("x") as private:
        for row in rows:
            try:
                record, trace = predict(method_input(row), generate, official, sample_id=row["sample_id"], style=style)
            except Exception as error:
                record = {"sample_id": row["sample_id"], "status": "error", "error_type": type(error).__name__,
                          "error_sha256": digest(str(error)), "costs": None}
                trace = {"sample_id": row["sample_id"], "generation": {"status": "error", "text": "",
                          "error_type": type(error).__name__, "error": str(error)}}
            record.update(seed=args.seed, method="native_" + style, source_cluster_sha256=row.get("source_cluster_sha256"),
                          note_text_sha256=row.get("note_text_sha256"))
            trace.update(seed=args.seed, method="native_" + style)
            public.write(json.dumps(record, ensure_ascii=False) + "\n")
            private.write(json.dumps(trace, ensure_ascii=False) + "\n")
            public.flush(); private.flush()
            statuses[record["status"]] += 1
    run.update(status="complete_verified" if not statuses["error"] else "failed", output_rows=sum(statuses.values()),
               prediction_statuses=dict(statuses), accuracy="not_evaluated")
    manifest.write_text(json.dumps(run, indent=2))
    print(json.dumps({"status": run["status"], "split": args.split, "rows": run["output_rows"]}))
    return run


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result["status"] == "complete_verified" else 1)
