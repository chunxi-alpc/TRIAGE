"""CounterFact efficacy adapter: exact official preprocessing/metric, one token.

Run on the allocated server only. CAD is an implementation adaptation;
Spotlight is a paper reimplementation, not an official author release.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from baseline_operators import cad_first_token, spotlight_generate


PIN = "aab9cfed80dc012fb9e83f8bdefb8675c33f40e7"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompts_from_row(row: dict):
    """Match upstream add_unmediated_fact=True, chat=False, add_marker=False."""
    prefix = row["context"].rstrip(". ") + ". "
    if not row["prompt"].startswith(prefix):
        raise ValueError("Official contextual prompt does not start with its exact context prefix")
    prior = row["prompt"][len(prefix):]
    old_fact = row["context"].replace(row["target_mediated"], row["target_unmediated"]) + ". "
    full = f"Previously {old_fact}Currently {row['prompt']}"
    attribute = row["attribute"]
    if not attribute or not prior:
        raise ValueError("Empty attribute or prior question")
    # str.replace in the upstream marked path edits every nonoverlapping occurrence.
    spans, start = [], 0
    while (index := full.find(attribute, start)) >= 0:
        spans.append((index, index + len(attribute)))
        start = index + len(attribute)
    if not spans:
        raise ValueError("Official attribute was not found in the complete prompt")
    return full, prior, spans


def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--seka-root", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-fingerprint", required=True,
                        help="Coordinator's model manifest digest; not an inferred checkpoint identity")
    parser.add_argument("--data-path", required=True, type=Path, help="Actual official counterfact.jsonl file")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--subset", default="0:16")
    parser.add_argument("--method", choices=["cad", "spotlight"], required=True)
    parser.add_argument("--controls", nargs="+", default=None)
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    source_revision = subprocess.check_output(
        ["git", "-C", str(args.seka_root), "rev-parse", "HEAD"], text=True).strip()
    if source_revision != PIN:
        raise ValueError("Official SEKA checkout differs from the pinned revision")
    sys.path.insert(0, str(args.seka_root.resolve()))
    from benchmarks.counterfact.preprocess import load_dataset
    from benchmarks.counterfact.evaluate import efficacy

    controls = args.controls or ["no_hook", "matched_zero", "normal", "reverse"]
    if args.controls is None and args.method == "spotlight":
        controls.append("wrong_span")
    allowed = {"no_hook", "matched_zero", "normal", "reverse"}
    if args.method == "spotlight":
        allowed.add("wrong_span")
    if len(set(controls)) != len(controls) or not set(controls) <= allowed:
        raise ValueError("Unknown or duplicate controls")
    if "matched_zero" in controls and "no_hook" not in controls:
        raise ValueError("Include no_hook when measuring matched_zero numerical drift")
    # Fix ordering for reference comparisons; no output-dependent selection.
    controls = [c for c in ["no_hook", "matched_zero", "normal", "reverse", "wrong_span"] if c in controls]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "task_state": "running", "execution_status": "started", "method": args.method,
        "official_metric": "SEKA CounterFact efficacy, first-token max over lower/capitalize/upper targets",
        "method_status": "implementation_adaptation" if args.method == "cad" else "paper_reimplementation",
        "source_revision": source_revision, "checkpoint": args.model,
        "checkpoint_status": "Qwen3-8B adapted checkpoint, not paper Qwen3-8B-Base",
        "model_fingerprint": args.model_fingerprint, "data_sha256": file_sha256(args.data_path),
        "adapter_sha256": file_sha256(Path(__file__).with_name("baseline_operators.py")),
        "runner_sha256": file_sha256(Path(__file__)), "subset": args.subset,
        "controls": controls, "seed": args.seed, "alpha": 1.0, "spotlight_target": .1,
        "max_new_tokens": 1, "greedy": True, "batch_size": 1,
        "add_unmediated_fact": True, "chat": False, "candidate_filtering": False,
        "versions": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "tokenizers", "datasets"]},
    }
    manifest_path = args.output_dir / "manifest.json"

    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    save_manifest()
    try:
        torch.manual_seed(args.seed)
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype="auto", attn_implementation="eager").to("cuda").eval()
        manifest["model_dtype"] = str(next(model.parameters()).dtype)
        manifest["generation_config"] = model.generation_config.to_dict()
        dataset = load_dataset(str(args.data_path), tokenizer, example_subset=args.subset)
        manifest["expected_examples"] = len(dataset)
        save_manifest()
        grouped = {c: defaultdict(list) for c in controls}
        numeric = []
        elapsed_start = time.perf_counter()
        completed = 0
        with (args.output_dir / "predictions.jsonl").open("x") as output:
            for row in dataset:
                full, prior, spans = prompts_from_row(row)
                reference = None
                for control in controls:
                    actual_forward_calls = [0]

                    def count_forward(*unused):
                        actual_forward_calls[0] += 1

                    handle = model.register_forward_hook(count_forward)
                    torch.cuda.reset_peak_memory_stats(model.device)
                    try:
                        if args.method == "cad":
                            result = cad_first_token(model, tokenizer, full, prior,
                                                     alpha=1., control=control, seed=args.seed)
                        else:
                            result = spotlight_generate(model, tokenizer, full, spans,
                                                        target=.1, control=control,
                                                        max_new_tokens=1, seed=args.seed)
                    finally:
                        handle.remove()
                    if actual_forward_calls[0] != result.metadata["forward_calls"]:
                        raise ValueError("Actual model forward count differs from recorded generation calls")
                    scores = result.first_token_scores
                    log_probs = torch.log_softmax(scores, dim=-1)
                    target_ids = torch.as_tensor(row["target_mediated.token_id"], dtype=torch.long)
                    comparator_ids = torch.as_tensor(row["target_unmediated.token_id"], dtype=torch.long)
                    target_logp = float(log_probs[target_ids].max())
                    comparator_logp = float(log_probs[comparator_ids].max())
                    record = {
                        "id": str(row["id"]), "control": control,
                        "prediction": result.text, "output_token_ids": result.output_token_ids,
                        "target_token_ids": target_ids.tolist(), "comparator_token_ids": comparator_ids.tolist(),
                        "target_logp": target_logp, "comparator_logp": comparator_logp,
                        "efficacy_correct": target_logp > comparator_logp,
                        "prompt_sha256": hashlib.sha256(full.encode()).hexdigest(),
                        "prior_sha256": hashlib.sha256(prior.encode()).hexdigest(),
                        "attribute_character_spans": spans, "metadata": result.metadata,
                        "peak_allocated_gib": torch.cuda.max_memory_allocated(model.device) / 2**30,
                        "peak_reserved_gib": torch.cuda.max_memory_reserved(model.device) / 2**30,
                        "actual_forward_calls": actual_forward_calls[0],
                    }
                    if control == "no_hook":
                        reference = scores
                    if control == "matched_zero":
                        drift = {
                            "id": str(row["id"]),
                            "max_abs_first_token_score_difference": float((scores - reference).abs().max()),
                            "argmax_changed": bool(scores.argmax() != reference.argmax()),
                        }
                        record["numerical_control"] = drift
                        numeric.append(drift)
                    grouped[control][str(row["id"])].append((target_logp, comparator_logp))
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output.flush()
                    completed += 1
                manifest["completed_prediction_records"] = completed
                save_manifest()
        metrics = {}
        for control, by_id in grouped.items():
            metrics[control] = efficacy(
                [[p[0] for p in pairs] for pairs in by_id.values()],
                [[p[1] for p in pairs] for pairs in by_id.values()], store_values=True).to_dict()
        (args.output_dir / "efficacy_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        (args.output_dir / "numerical_controls.json").write_text(json.dumps(numeric, indent=2) + "\n")
        manifest.update(execution_status="finished", verification_pending=True,
                        completed_prediction_records=completed, elapsed_seconds=time.perf_counter() - elapsed_start)
        # Independent audit, not this runner, changes task_state to complete_verified.
        save_manifest()
    except Exception as error:
        manifest.update(task_state="blocked", execution_status="failed",
                        error_type=type(error).__name__, error_message=str(error), partial_results_retained=True)
        save_manifest()
        raise


if __name__ == "__main__":
    main()
