#!/usr/bin/env python3
"""Correct only the secondary answer-token boundary of a completed M2 run.

The saved operator, head selection, patients, conditions and forward computation
are unchanged. Every replayed condition is checked against the original saved
legacy logits and actual attention before the corrected readout is accepted.
Original records and fingerprints are never overwritten.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.run_module2_mechanism import context_key, jsonable, sha256, source_fingerprint, write_json
from scripts.run_module2_mechanism_causal import CONDITIONS, plan
from src.module2.mechanism_extraction import tokenize_online


READOUT_SPEC = {
    "schema_version": "module2.boundary_readout.v1",
    "purpose": "technical correction of the secondary constrained A/B answer readout",
    "candidate_labels": ["A", "B"],
    "candidate_continuations": [" A", " B"],
    "candidate_token_ids": [362, 425],
    "legacy_candidate_token_ids": [32, 33],
    "boundary_rule": "tokenize(prompt + continuation) == tokenize(prompt) + one candidate token",
    "prompt_changed": False,
    "new_fit_or_selection": False,
}


def record_key(record):
    return (str(record["case_id"]), str(record["variant"]), str(record["method"]),
            str(record["mode"]), float(record["gain"]))


def unique_records(records, expected_keys):
    """Fail closed on duplicates, missing cells or extra conditions."""
    mapping = {}
    for row in records:
        key = record_key(row)
        if key in mapping:
            raise ValueError("Duplicate patient/method/mode/gain record")
        mapping[key] = row
    if set(mapping) != set(expected_keys):
        raise ValueError("Record coverage differs from the complete frozen causal grid")
    return mapping


def verify_boundary(tokenizer, prompt, expected_prefix_ids):
    """Check the actual full prompt boundary, never isolated label tokenization."""
    base = list(tokenizer.encode(prompt, add_special_tokens=False))
    if base != list(expected_prefix_ids):
        raise ValueError("Boundary audit and model input tokenization differ")
    audits = []
    for continuation, expected in zip(READOUT_SPEC["candidate_continuations"],
                                      READOUT_SPEC["candidate_token_ids"]):
        full = list(tokenizer.encode(prompt + continuation, add_special_tokens=False))
        if len(full) != len(base) + 1 or full[:-1] != base or full[-1] != expected:
            raise ValueError("Candidate continuation does not append its frozen single token")
        audits.append({"continuation": continuation, "token_id": expected,
                       "prefix_tokens_unchanged": True, "appended_tokens": 1,
                       "prompt_token_count": len(base)})
    return audits


def compare_saved_forward(original, replay, legacy_logits, *, logit_atol=1e-5, attention_atol=1e-6):
    """Compare the originally saved readout and all head/query/token masses."""
    old = np.asarray([original["logits_A"], original["logits_B"]], dtype=float)
    new = np.asarray(legacy_logits, dtype=float)
    if new.shape != (2,) or not np.isfinite(old).all() or not np.isfinite(new).all():
        raise ValueError("Legacy readouts must be two finite values")
    logit_error = float(np.max(np.abs(old - new)))
    before, after = original["attention_mass"], replay["attention_mass"]
    if len(before) != len(after):
        raise ValueError("Original/replay attention coverage differs")
    attention_error = 0.
    for a, b in zip(before, after):
        if any(a[k] != b[k] for k in ("layer", "kv_head", "rule_id", "positions")):
            raise ValueError("Original/replay attention targets differ")
        av, bv = np.asarray(a["per_query_token_mass"]), np.asarray(b["per_query_token_mass"])
        if av.shape != bv.shape or not np.isfinite(av).all() or not np.isfinite(bv).all():
            raise ValueError("Original/replay attention arrays must match and be finite")
        attention_error = max(attention_error, float(np.max(np.abs(av - bv))))
    if logit_error > logit_atol or attention_error > attention_atol:
        raise RuntimeError(f"Forward replay differs: legacy logits={logit_error}, attention={attention_error}")
    old_prediction = "A" if old[0] >= old[1] else "B"
    new_prediction = "A" if new[0] >= new[1] else "B"
    if old_prediction != new_prediction:
        raise RuntimeError("Replay changed the original legacy A/B ranking")
    return {"legacy_logit_max_abs_error": logit_error,
            "attention_max_abs_error": attention_error,
            "legacy_ranking_identical": True,
            "bitwise_numeric_identity": logit_error == 0. and attention_error == 0.,
            "logit_atol": logit_atol, "attention_atol": attention_atol}


def run(args):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.module3.guideline_prefix import (prepare_guideline_prefix, prepare_guideline_tail,
                                             run_guideline_suffix, run_guideline_tail)
    started = time.monotonic()
    source = Path(args.source)
    original_dir = source / "causal"
    output = source / "causal_boundary_readout"
    if json.loads((original_dir / "status.json").read_text())["status"] != "complete":
        raise ValueError("The original causal experiment must finish before readout correction")
    if json.loads((source / "status.json").read_text())["status"] != "complete":
        raise ValueError("The original representation experiment is incomplete")
    representation = json.loads((source / "fingerprint.json").read_text())
    original_fingerprint = json.loads((original_dir / "fingerprint.json").read_text())
    frozen_conditions = [dict(method=m, mode=mode, gain=g) for m, mode, g in CONDITIONS]
    if original_fingerprint["conditions"] != frozen_conditions:
        raise ValueError("Replay conditions differ from the original frozen grid")
    if representation["candidate_token_ids"] != READOUT_SPEC["legacy_candidate_token_ids"]:
        raise ValueError("This correction is defined only for the recorded 32/33 legacy readout")
    if sha256(args.triplets) != representation["data_sha256"]:
        raise ValueError("Replay patients differ from the original input")
    for name, metadata in representation["model_files"].items():
        if sha256(Path(args.model) / name) != metadata["sha256"]:
            raise ValueError("The frozen model/tokenizer files changed")
    for filename, field in (("operators.npz", "operators_sha256"),
                             ("selection.json", "selection_sha256"),
                             ("fingerprint.json", "representation_fingerprint_sha256")):
        if sha256(source / filename) != original_fingerprint[field]:
            raise ValueError("Original causal dependencies changed")
    selection = json.loads((source / "selection.json").read_text())
    heads = [tuple(map(int, h)) for h in selection["selected_heads"]]
    beta = np.asarray(selection["bias_unit_rms"], float)
    saved = np.load(source / "operators.npz")
    directions = {"projected": saved["delta"], "raw_mean": saved["raw_unit"],
                  "random": saved["random_direction"], "reversed": -saved["delta"]}
    rows = [json.loads(line) for line in Path(args.triplets).read_text().splitlines()]
    canonical = [r for r in rows if r["split"] == "confirm" and r["target_rule_id"] == "A"]
    if len(canonical) != 12 or len({context_key(r) for r in canonical}) != 12:
        raise ValueError("Expected the original 12 distinct confirmation configurations")
    expected = {(context_key(row), variant, method, mode, gain)
                for row in canonical for variant in ("positive", "negative")
                for method, mode, gain in CONDITIONS}
    originals = [json.loads(line) for line in (original_dir / "records.jsonl").read_text().splitlines()]
    original_index = unique_records(originals, expected)
    signature = {
        "schema_version": "module2.mechanism.boundary_replay.v1",
        "readout_spec": READOUT_SPEC,
        "original_causal_records_sha256": sha256(original_dir / "records.jsonl"),
        "original_causal_fingerprint_sha256": sha256(original_dir / "fingerprint.json"),
        "original_causal_fingerprint": original_fingerprint,
        "representation_fingerprint_sha256": sha256(source / "fingerprint.json"),
        "operators_sha256": sha256(source / "operators.npz"),
        "selection_sha256": sha256(source / "selection.json"),
        "triplets_sha256": sha256(args.triplets),
        "source": source_fingerprint([
            "scripts/replay_module2_boundary_readout.py", "scripts/run_module2_mechanism_causal.py",
            "scripts/run_module2_mechanism.py", "src/module3/guideline_prefix.py",
            "src/module2/mechanism_extraction.py", "src/module2/extraction.py"]),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "numpy": np.__version__, "dtype": "float32", "threads": args.threads,
        "conditions": frozen_conditions, "heads": heads,
        "unique_prompts": 24, "candidate_boundary_checks": 48,
        "expected_records": len(expected), "new_model_fit": False,
        "new_head_selection": False, "new_gain_selection": False,
        "new_case_selection": False,
        "validation": "all conditions against original legacy logits and per-query-token attention",
        "logit_atol": 1e-5, "attention_atol": 1e-6,
    }
    output.mkdir(exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    fingerprint_path = output / "fingerprint.json"
    if fingerprint_path.exists():
        if json.loads(fingerprint_path.read_text()) != json.loads(json.dumps(signature, default=jsonable)):
            raise ValueError("Refuse to resume correction under a different fingerprint")
    else:
        write_json(fingerprint_path, signature)
    torch.set_num_threads(args.threads)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    encoded, audits = {}, []
    for ci, row in enumerate(canonical):
        for variant in ("positive", "negative"):
            tokenized = tokenize_online(tokenizer, row, variant)
            checks = verify_boundary(tokenizer, tokenized["prompt"],
                                     tokenized["encoded"]["input_ids"][0].tolist())
            encoded[ci, variant] = tokenized
            audits.extend(dict(case_id=context_key(row), variant=variant, **check) for check in checks)
    if len(audits) != 48:
        raise ValueError("Boundary coverage is incomplete")
    write_json(output / "token_boundary_audit.json", audits)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
                    dtype=torch.float32, attn_implementation="eager").eval()
    targets = [{"layer": l, "kv_head": h, "rule_id": r} for l, h in heads for r in ("A", "B")]
    responses = [dict(t, delta=directions["projected"][t["layer"], t["kv_head"]]) for t in targets]
    records, validations = [], []
    completed = 0
    for ci, row in enumerate(canonical):
        case, prefix, original_cache = context_key(row), None, None
        for vi, variant in enumerate(("positive", "negative")):
            checkpoint = output / "checkpoints" / f"prompt_{ci:02d}_{vi}.json"
            if checkpoint.exists():
                previous = json.loads(checkpoint.read_text())
                records.extend(previous["records"])
                validations.extend(previous["validations"])
                completed += 1
                continue
            tokenized = encoded[ci, variant]
            ids = tokenized["encoded"]["input_ids"]
            if prefix is None:
                prefix = prepare_guideline_prefix(model, ids, tokenized["prefix_length"], tokenized["rule_positions"])
                original_cache = [(k.clone(), v.clone()) for k, v in prefix.cache.to_legacy_cache()]
            elif prefix.rule_positions != tokenized["rule_positions"]:
                raise ValueError("Opposite patients have different guide target mappings")
            if heads:
                tail = prepare_guideline_tail(model, prefix, ids, min(l for l, _ in heads),
                                               diagnostic_targets=targets, response_targets=responses)
                baseline = tail.baseline
                def execute(shifts, biases):
                    return run_guideline_tail(model, tail, shifts, score_bias=biases,
                                               diagnostic_targets=targets, response_targets=responses)
            else:
                baseline = run_guideline_suffix(model, prefix, ids)
                def execute(shifts, biases):
                    return run_guideline_suffix(model, prefix, ids, shifts, score_bias=biases)
            local, checks = [], []
            gold = row["oracle"][variant]["selected_rule_id"]
            for method, mode, gain in CONDITIONS:
                key = (case, variant, method, mode, gain)
                original = original_index[key]
                if original["gold_rule"] != gold:
                    raise ValueError("Replay changed the oracle patient-to-rule mapping")
                shifts, biases = plan(method, mode, gain, heads, directions, beta, gold)
                actual = baseline if method == "baseline" else execute(shifts, biases)
                legacy_logits = actual["logits"][READOUT_SPEC["legacy_candidate_token_ids"]].tolist()
                check = compare_saved_forward(original, actual, legacy_logits)
                checks.append(dict(case_id=case, variant=variant, method=method, mode=mode, gain=gain, **check))
                new_logits = actual["logits"][READOUT_SPEC["candidate_token_ids"]].tolist()
                if not np.isfinite(new_logits).all():
                    raise ValueError("Corrected candidate logits are nonfinite")
                prediction = "A" if new_logits[0] >= new_logits[1] else "B"
                legacy_prediction = "A" if legacy_logits[0] >= legacy_logits[1] else "B"
                record = deepcopy(original)
                record.update(logits_A=float(new_logits[0]), logits_B=float(new_logits[1]),
                    prediction=prediction, correct=prediction == gold,
                    legacy_logits_A=float(legacy_logits[0]), legacy_logits_B=float(legacy_logits[1]),
                    legacy_prediction=legacy_prediction, legacy_correct=legacy_prediction == gold,
                    attention_mass=actual["attention_mass"], query_response=actual["query_response"],
                    readout_spec=READOUT_SPEC,
                    correction_reason="canonical single-token continuation at the actual Answer: boundary")
                for rule in ("A", "B"):
                    values = [x["target_attention_mass"] for x in actual["attention_mass"] if x["rule_id"] == rule]
                    record["mass_" + rule] = float(np.mean(values)) if values else None
                local.append(record)
            for (oldk, oldv), (key, value) in zip(original_cache, prefix.cache.to_legacy_cache()):
                if not torch.equal(oldk, key) or not torch.equal(oldv, value):
                    raise RuntimeError("Replay mutated the shared original guideline cache")
            write_json(checkpoint, {"records": local, "validations": checks})
            records.extend(local)
            validations.extend(checks)
            completed += 1
            write_json(output / "status.json", {"status": "running", "prompts_completed": completed,
                "expected_prompts": 24, "records": len(records), "expected_records": len(expected),
                "elapsed_seconds": time.monotonic() - started})
            print(f"boundary replay {completed}/24 | records={len(records)}/{len(expected)} | elapsed={time.monotonic()-started:.1f}s", flush=True)
    unique_records(records, expected)
    unique_records(validations, expected)
    with (output / "records.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False, default=jsonable) + "\n")
    write_json(output / "validation.json", validations)
    write_json(output / "status.json", {"status": "complete", "prompts_completed": completed,
        "records": len(records), "expected_records": len(expected), "missing_records": 0,
        "boundary_checks": len(audits), "forward_replay_checks": len(validations),
        "max_legacy_logit_error": max(x["legacy_logit_max_abs_error"] for x in validations),
        "max_attention_error": max(x["attention_max_abs_error"] for x in validations),
        "all_legacy_rankings_identical": all(x["legacy_ranking_identical"] for x in validations),
        "all_saved_values_exactly_reproduced": all(x["bitwise_numeric_identity"] for x in validations),
        "elapsed_seconds": time.monotonic() - started})
    print("boundary replay complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", default="experiments/module2_mechanism_20261028/results")
    parser.add_argument("--triplets", default="experiments/module2_mechanism_20261028/inputs/triplets.jsonl")
    parser.add_argument("--threads", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
