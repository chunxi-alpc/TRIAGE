#!/usr/bin/env python3
"""Fixed follow-up stress diagnostic of frozen rule-reading routes and A/B output."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from scripts.run_module2_mechanism import context_key, jsonable, sha256, source_fingerprint, write_json
from scripts.run_module2_mechanism_causal import compare
from scripts.replay_module2_boundary_readout import READOUT_SPEC, verify_boundary
from src.module2.extraction import span_token_positions
from src.module2.mechanism_extraction import tokenize_online


CONDITIONS = (
    ("baseline", 0., 0., None),
    ("projected4", 4., 0., None),
    ("projected32", 32., 0., None),
    ("projected_minus32", -32., 0., None),
    ("score_plus8", 0., 8., None),
    ("score_minus8", 0., -8., None),
    ("note_gold", 0., 0., "gold"),
    ("note_sibling", 0., 0., "sibling"),
)
NOTE_TEMPLATE = (
    "\nRouting result (provided by the rule parser):\n"
    "Applicable rule: {applicable}.\n"
)
STRONG = {"projected32", "projected_minus32", "score_plus8", "score_minus8"}
CANDIDATES = READOUT_SPEC["candidate_token_ids"]


def load_records(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def insert_note(prompt, rule):
    if rule not in {"A", "B"} or not prompt.endswith("\nAnswer:"):
        raise ValueError("Expected an A/B rule and the original exact answer cue")
    index = prompt.rindex("\nAnswer:")
    return prompt[:index] + NOTE_TEMPLATE.format(applicable=rule) + prompt[index:]


def intervention(heads, delta, gold, key_gain=0., score_bias=0.):
    if gold not in {"A", "B"} or (key_gain and score_bias):
        raise ValueError("Choose exactly one intervention type and a legal gold rule")
    shifts, biases = [], []
    for layer, head in heads:
        for rule in ("A", "B"):
            sign = 1. if rule == gold else -1.
            common = dict(layer=layer, kv_head=head, rule_id=rule)
            if key_gain:
                shifts.append(dict(common, delta=delta[layer, head],
                                   coefficient=sign * key_gain))
            if score_bias:
                biases.append(dict(common, bias=sign * score_bias))
    return shifts, biases


def readout(logits, tokenizer, gold):
    import torch
    if gold not in {"A", "B"}:
        raise ValueError("Expected a legal gold rule")
    logits = logits.detach().float().cpu()
    if logits.ndim != 1 or not bool(torch.isfinite(logits).all()):
        raise ValueError("Expected finite full-vocabulary logits")
    probabilities = torch.softmax(logits, dim=-1)
    values = logits[CANDIDATES]
    candidate_p = probabilities[CANDIDATES]
    conditional = torch.softmax(values, dim=-1)
    prediction = "A" if values[0] >= values[1] else "B"
    top_p, top_ids = torch.topk(probabilities, 5)
    top = [
        dict(token_id=int(i), text=tokenizer.decode([int(i)]),
             logit=float(logits[i]), probability=float(p))
        for p, i in zip(top_p, top_ids)
    ]
    sign = 1 if gold == "A" else -1
    return dict(
        logits_A=float(values[0]), logits_B=float(values[1]),
        prediction=prediction, correct=prediction == gold,
        gold_margin=float(sign * (values[0] - values[1])),
        candidate_probability_A=float(candidate_p[0]),
        candidate_probability_B=float(candidate_p[1]),
        candidate_probability_mass=float(candidate_p.sum()),
        conditional_probability_A=float(conditional[0]),
        conditional_probability_B=float(conditional[1]), top_tokens=top,
    )


def verify_anchor(old, actual):
    new = actual["logits"][CANDIDATES].tolist()
    old_logits = np.asarray([old["logits_A"], old["logits_B"]], dtype=float)
    if not np.isfinite(old_logits).all() or not np.isfinite(new).all():
        raise ValueError("Nonfinite anchor candidate logits")
    error = float(np.max(np.abs(np.asarray(new) - old_logits)))
    attention_error = 0.
    if len(old["attention_mass"]) != len(actual["attention_mass"]):
        raise ValueError("Anchor target count differs")
    for before, after in zip(old["attention_mass"], actual["attention_mass"]):
        if any(before[k] != after[k] for k in ("layer", "kv_head", "rule_id", "positions")):
            raise ValueError("Anchor target identities differ")
        a, b = np.asarray(before["per_query_token_mass"]), np.asarray(after["per_query_token_mass"])
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Invalid anchor attention arrays")
        attention_error = max(attention_error, float(np.max(np.abs(a-b))))
    if error > 1e-5 or attention_error > 1e-6:
        raise RuntimeError(f"Frozen anchor not reproduced: {error}, {attention_error}")
    if any(old[k] != actual[k] for k in ("prefix_length", "suffix_length")):
        raise ValueError("Anchor token lengths changed")
    return dict(candidate_logit_max_error=error, attention_max_error=attention_error,
                logit_atol=1e-5, attention_atol=1e-6)


def mass_summary(actual, heads, role_positions):
    mass = {(x["layer"], x["kv_head"], x["rule_id"]): x for x in actual["attention_mass"]}
    response = {(x["layer"], x["kv_head"], x["rule_id"]): x for x in actual["query_response"]}
    per_head = []
    for layer, head in heads:
        row = dict(layer=layer, kv_head=head)
        for rule in ("A", "B"):
            entry, q = mass[layer, head, rule], response[layer, head, rule]
            row[f"mass_{rule}"] = entry["target_attention_mass"]
            for role in ("id", "action"):
                use = set(role_positions[rule][role])
                row[f"mass_{rule}_{role}"] = float(sum(
                    m for p, m in zip(entry["positions"], entry["per_token_mass"]) if p in use))
            scores = np.asarray(q["scores"])
            row[f"response_{rule}"] = dict(mean=q["mean"], rms=q["rms"],
                positive_fraction=q["positive_fraction"], minimum=float(scores.min()),
                maximum=float(scores.max()))
        row["mass_other"] = 1. - row["mass_A"] - row["mass_B"]
        per_head.append(row)
    return dict(mass_A=float(np.mean([x["mass_A"] for x in per_head])),
                mass_B=float(np.mean([x["mass_B"] for x in per_head])),
                per_head_mass=per_head)


def run(args):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.module3.guideline_prefix import (
        prepare_guideline_prefix, prepare_guideline_tail, run_guideline_tail, run_guideline_suffix)

    start = time.monotonic()
    source, out = Path(args.source), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)
    prior = source / "causal_boundary_readout"
    if any(json.loads((p / "status.json").read_text())["status"] != "complete"
           for p in (source, prior)):
        raise ValueError("The prior frozen experiment must be complete")
    old_path = prior / "records.jsonl.gz"
    old_rows = load_records(old_path)
    old = {(r["case_id"], r["variant"], r["method"], r["mode"], float(r["gain"])): r
           for r in old_rows}
    if len(old_rows) != 216 or len(old) != 216:
        raise ValueError("Expected all 216 canonical prior records")
    previous_fingerprint = json.loads((source / "fingerprint.json").read_text())
    if sha256(args.triplets) != previous_fingerprint["data_sha256"]:
        raise ValueError("Frozen patients changed")
    for path, expected in previous_fingerprint["source"].items():
        if sha256(ROOT / path) != expected:
            raise ValueError(f"Previously frozen scientific source changed: {path}")
    for name, meta in previous_fingerprint["model_files"].items():
        if sha256(Path(args.model) / name) != meta["sha256"]:
            raise ValueError(f"Frozen model/tokenizer changed: {name}")
    causal_fingerprint = json.loads((source / "causal/fingerprint.json").read_text())
    for filename, field in (("operators.npz", "operators_sha256"),
                            ("selection.json", "selection_sha256")):
        if sha256(source / filename) != causal_fingerprint[field]:
            raise ValueError(f"Prior frozen causal asset changed: {filename}")
    heads = [tuple(x) for x in json.loads((source / "selection.json").read_text())["selected_heads"]]
    if heads != [(13, 4), (23, 6), (6, 2), (22, 7)]:
        raise ValueError("Expected the same four frozen heads")
    delta = np.load(source / "operators.npz")["delta"]
    rows = sorted((r for r in load_records(args.triplets)
                   if r["split"] == "confirm" and r["target_rule_id"] == "A"),
                  key=context_key)
    if len(rows) != 12 or len({r["group_id"] for r in rows}) != 3:
        raise ValueError("Expected all 12 prior confirmation configurations")
    first_case = {}
    for row in rows:
        first_case.setdefault(row["group_id"], context_key(row))
    protocol = ROOT / "experiments/rule_readout_diagnostic_20261028/PROTOCOL.zh.md"
    signature = dict(
        schema_version="rule_readout.diagnostic.v1", prior_commit="f3a17f721118407d6a9855d8faf48cbbb90e9770",
        protocol_sha256=sha256(protocol), input_sha256=sha256(args.triplets),
        previous_records_sha256=sha256(old_path),
        previous_fingerprint_sha256=sha256(source / "fingerprint.json"),
        operators_sha256=sha256(source / "operators.npz"),
        selection_sha256=sha256(source / "selection.json"),
        model_revision=previous_fingerprint["model_revision"],
        model_files=previous_fingerprint["model_files"],
        conditions=CONDITIONS, note_template=NOTE_TEMPLATE, heads=heads,
        torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__,
        threads=args.threads, dtype="float32", max_length=4096,
        readout=dict(READOUT_SPEC, purpose="reuse canonical A/B candidates with full-vocabulary diagnostics"),
        source=source_fingerprint([
            "scripts/run_rule_readout_diagnostic.py", "src/module3/guideline_prefix.py",
            "src/module2/mechanism_extraction.py", "src/module2/extraction.py",
            "scripts/run_module2_mechanism.py", "scripts/run_module2_mechanism_causal.py",
            "scripts/replay_module2_boundary_readout.py"]),
    )
    signature = json.loads(json.dumps(signature, default=jsonable))
    if (out / "fingerprint.json").exists():
        if json.loads((out / "fingerprint.json").read_text()) != signature:
            raise ValueError("Refuse to resume under a different diagnostic fingerprint")
    else:
        write_json(out / "fingerprint.json", signature)
    torch.set_num_threads(args.threads)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.float32, attn_implementation="eager").eval()
    records, checks, boundaries = [], [], []
    for ci, row in enumerate(rows):
        case = context_key(row)
        prefix = None
        for vi, variant in enumerate(("positive", "negative")):
            cp = out / "checkpoints" / f"prompt_{ci:02d}_{vi}.json"
            if cp.exists():
                saved = json.loads(cp.read_text())
                records.extend(saved["records"]); checks.extend(saved["checks"])
                boundaries.extend(saved["boundaries"])
                continue
            tokenized = tokenize_online(tokenizer, row, variant, max_length=4096)
            ids, prompt = tokenized["encoded"]["input_ids"], tokenized["prompt"]
            if prefix is None:
                prefix = prepare_guideline_prefix(
                    model, ids, tokenized["prefix_length"], tokenized["rule_positions"])
            original = [(k.clone(), v.clone()) for k, v in prefix.cache.to_legacy_cache()]
            role_positions = {
                rule: {role: span_token_positions(tokenized["offsets"],
                    tokenized["spans"]["rules"][rule][role], prompt)
                    for role in ("id", "action")} for rule in ("A", "B")}
            targets = [dict(layer=layer, kv_head=head, rule_id=rule)
                       for layer, head in heads for rule in ("A", "B")]
            responses = [dict(x, delta=delta[x["layer"], x["kv_head"]]) for x in targets]
            tail = prepare_guideline_tail(model, prefix, ids, min(x[0] for x in heads),
                diagnostic_targets=targets, response_targets=responses)
            gold = row["oracle"][variant]["selected_rule_id"]
            sibling = "B" if gold == "A" else "A"
            local, local_checks, local_boundaries = [], [], []
            for condition, key_gain, score_bias, note in CONDITIONS:
                note_rule = gold if note == "gold" else sibling if note else None
                actual_prompt = insert_note(prompt, note_rule) if note else prompt
                actual_ids = (tokenizer(actual_prompt, add_special_tokens=False,
                    return_tensors="pt", truncation=False)["input_ids"] if note else ids)
                if actual_ids.shape[1] > 4096 or not torch.equal(
                        actual_ids[:, :prefix.prefix_length], prefix.prefix_ids):
                    raise ValueError("Note changed the guideline prefix or exceeded the fixed limit")
                audited = verify_boundary(tokenizer, actual_prompt, actual_ids[0].tolist())
                local_boundaries.append(dict(case_id=case, variant=variant,
                    condition=condition, checks=audited))
                shifts, biases = intervention(heads, delta, gold, key_gain, score_bias)
                if condition == "baseline":
                    actual = tail.baseline
                elif note:
                    actual = run_guideline_suffix(model, prefix, actual_ids,
                        diagnostic_targets=targets, response_targets=responses)
                else:
                    actual = run_guideline_tail(model, tail, shifts, score_bias=biases,
                        diagnostic_targets=targets, response_targets=responses)
                if condition in {"baseline", "projected4"}:
                    prior_key = ((case, variant, "baseline", "none", 0.) if condition == "baseline"
                                 else (case, variant, "projected", "both", 4.))
                    local_checks.append(dict(case_id=case, variant=variant, condition=condition,
                        kind="prior_anchor", **verify_anchor(old[prior_key], actual)))
                if case == first_case[row["group_id"]] and condition in STRONG:
                    reference = run_guideline_suffix(model, prefix, ids, shifts, score_bias=biases,
                        diagnostic_targets=targets, response_targets=responses)
                    local_checks.append(dict(case_id=case, variant=variant, condition=condition,
                        kind="full_suffix_vs_tail",
                        **compare(reference, actual, logit_atol=1e-5, attention_atol=1e-6)))
                local.append(dict(
                    case_id=case, triplet_id=row["id"], group_id=row["group_id"], variant=variant,
                    patient_state=row["patient_facts"][variant][row["decisive_node_index"]],
                    gold_rule=gold, action_permutation=row["controls"]["action_permutation"],
                    rule_order_index=row["controls"]["rule_order_index"],
                    rule_order=row["controls"]["rule_order"], last_rule=row["controls"]["rule_order"][-1],
                    condition=condition, key_gain=key_gain, score_bias=score_bias, note_rule=note_rule,
                    prompt_sha256=hashlib.sha256(actual_prompt.encode()).hexdigest(),
                    token_count=int(actual_ids.shape[1]), prefix_length=actual["prefix_length"],
                    suffix_length=actual["suffix_length"],
                    **readout(actual["logits"], tokenizer, gold),
                    **mass_summary(actual, heads, role_positions)))
            for (oldk, oldv), (k, v) in zip(original, prefix.cache.to_legacy_cache()):
                if not torch.equal(oldk, k) or not torch.equal(oldv, v):
                    raise RuntimeError("A diagnostic changed the reusable guideline cache")
            write_json(cp, dict(records=local, checks=local_checks, boundaries=local_boundaries))
            records.extend(local); checks.extend(local_checks); boundaries.extend(local_boundaries)
            write_json(out / "status.json", dict(status="running", records=len(records),
                expected_records=192, prompts_completed=len(records)//8,
                elapsed_seconds=time.monotonic()-start))
            print(f"readout diagnostic {len(records)//8}/24 | records={len(records)}/192 | "
                  f"elapsed={time.monotonic()-start:.1f}s", flush=True)
    expected = {(context_key(r), v, c[0]) for r in rows
                for v in ("positive", "negative") for c in CONDITIONS}
    keys = [(r["case_id"], r["variant"], r["condition"]) for r in records]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise RuntimeError("Diagnostic coverage differs from the frozen 192 conditions")
    anchors = [x for x in checks if x["kind"] == "prior_anchor"]
    full = [x for x in checks if x["kind"] == "full_suffix_vs_tail"]
    if len(anchors) != 48 or len(full) != 24 or len(boundaries) != 192:
        raise RuntimeError("Incomplete anchor, path, or boundary validation")
    data = "".join(json.dumps(r, ensure_ascii=False, allow_nan=False)+"\n" for r in records).encode()
    (out / "records.jsonl").write_bytes(data)
    (out / "records.jsonl.gz").write_bytes(gzip.compress(data, mtime=0))
    write_json(out / "validation.json", checks)
    write_json(out / "token_boundary_audit.json", boundaries)
    write_json(out / "status.json", dict(
        status="complete", records=192, expected_records=192, missing_records=0,
        prompts_completed=24, n_paired_configs=12, n_guidelines=3, n_base_patient_states=6,
        anchor_checks=len(anchors), full_tail_checks=len(full), candidate_boundary_checks=384,
        max_anchor_logit_error=max(x["candidate_logit_max_error"] for x in anchors),
        max_anchor_attention_error=max(x["attention_max_error"] for x in anchors),
        max_full_tail_logit_error=max(x["logit_max_abs_error"] for x in full),
        max_full_tail_attention_error=max(x["attention_max_abs_error"] for x in full),
        max_full_tail_query_error=max(x["query_max_abs_error"] for x in full),
        planned_total_forward_invocations=dict(guideline_prefills=12, suffix_preparations=24,
            other_condition_forwards=168, full_suffix_checks=24, total=228),
        elapsed_seconds=time.monotonic()-start, records_sha256=hashlib.sha256(data).hexdigest()))
    print("readout diagnostic complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", default="experiments/module2_mechanism_20261028/results")
    parser.add_argument("--triplets", default="experiments/module2_mechanism_20261028/inputs/triplets.jsonl")
    parser.add_argument("--output", default="experiments/rule_readout_diagnostic_20261028/results")
    parser.add_argument("--threads", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
