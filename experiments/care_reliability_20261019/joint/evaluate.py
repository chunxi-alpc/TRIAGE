#!/usr/bin/env python3
"""Evaluate actual joint-model logs; CPU reference labels are never predictions."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def mean(values):
    return sum(values) / len(values) if values else None


def quantile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def cluster_interval(records, draws=10000, seed=20261019):
    """records contain (family, numerator, denominator), clustered by family."""
    groups = defaultdict(lambda: [0.0, 0])
    for family, numerator, denominator in records:
        groups[family][0] += numerator
        groups[family][1] += denominator
    groups = [value for value in groups.values() if value[1]]
    if not groups:
        return None
    rng = random.Random(seed)
    estimates = []
    for _ in range(draws):
        selection = [rng.choice(groups) for _ in groups]
        estimates.append(sum(value[0] for value in selection) / sum(value[1] for value in selection))
    return [quantile(estimates, .025), quantile(estimates, .975)]


def score_arm(cases, records, baseline=None):
    expected_ids = {c["case_id"] for c in cases}
    real = {r["case_id"]: r for r in records
            if r.get("case_id") in expected_ids and r.get("status") == "completed" and r.get("is_model_prediction") is True}
    invalid_controls = {r["case_id"] for r in records
                        if r.get("case_id") in expected_ids and r.get("status") == "zero_control_failed"}
    if invalid_controls:
        return {"status": "invalid_zero_control", "expected_cases": len(cases),
                "actual_model_outputs": len(real), "zero_control_failure_cases": len(invalid_controls),
                "case_accuracy": None, "app_accuracy": None, "non_accuracy": None, "H_Acc": None,
                "pair_accuracy": None, "misuse_rate": None, "baseline_correct_retention": None,
                "reason": "Mask-path zero differences cannot be counted as nonzero intervention effects"}
    if not real:
        return {"status": "blocked_or_not_run", "expected_cases": len(cases), "actual_model_outputs": 0,
                "case_accuracy": None, "H_Acc": None, "pair_accuracy": None, "misuse_rate": None}
    letters = set("ABCD")
    prediction = {cid: row.get("prediction") for cid, row in real.items() if row.get("prediction") in letters}
    case_correct = {c["case_id"]: prediction.get(c["case_id"]) == c["gold_answer"] for c in cases}
    app = [c for c in cases if c["reference_state"] == "T"]
    non = [c for c in cases if c["reference_state"] == "F"]
    unknown = [c for c in cases if c["reference_state"] == "U"]
    app_accuracy = mean([case_correct[c["case_id"]] for c in app])
    non_accuracy = mean([case_correct[c["case_id"]] for c in non])
    harmonic = (None if app_accuracy is None or non_accuracy is None else
                2 * app_accuracy * non_accuracy / (app_accuracy + non_accuracy) if app_accuracy + non_accuracy else 0.0)
    grouped = defaultdict(list)
    for case in cases:
        if case.get("pair_eligible"):
            grouped[case["pair_id"]].append(case)
    if any(len(pair) != 2 or {c["reference_state"] for c in pair} != {"T", "F"} for pair in grouped.values()):
        raise ValueError("Incomplete or invalid evaluation pair")
    pair_values = {pid: int(all(case_correct[c["case_id"]] for c in pair)) for pid, pair in grouped.items()}
    misuse_count = sum(prediction.get(c["case_id"]) == c["target_answer"] for c in non)
    missing_non = sum(c["case_id"] not in prediction for c in non)
    outputs = len(set(prediction) & {c["case_id"] for c in cases})
    result = {
        "status": "completed" if outputs == len(cases) else "partial_model_outputs",
        "expected_cases": len(cases), "actual_model_outputs": outputs,
        "missing_or_invalid_outputs": len(cases) - outputs,
        "case_accuracy": mean(list(case_correct.values())),
        "accuracy_missing_outputs_count_as_incorrect": True,
        "applicable_cases": len(app), "nonapplicable_cases": len(non), "unknown_cases": len(unknown),
        "app_accuracy": app_accuracy, "non_accuracy": non_accuracy, "H_Acc": harmonic,
        "unknown_accuracy": mean([case_correct[c["case_id"]] for c in unknown]),
        "pairs": len(pair_values), "pair_correct": sum(pair_values.values()), "pair_accuracy": mean(list(pair_values.values())),
        "pair_accuracy_ci95_family_bootstrap": cluster_interval(
            [(pair[0]["family_id"], pair_values[pid], 1) for pid, pair in grouped.items()]),
        "misuse_count": misuse_count, "misuse_denominator": len(non),
        "misuse_rate": misuse_count / len(non) if non and not missing_non else None,
        "misuse_identification_bounds_if_missing": ([misuse_count / len(non), (misuse_count + missing_non) / len(non)] if non else None),
        "structure_parse_failure_cases": sum(bool(real.get(c["case_id"], {}).get("structure", {}).get("parse_failure")) for c in cases),
    }
    if baseline is not None:
        base = {r["case_id"]: r for r in baseline if r.get("status") == "completed" and r.get("is_model_prediction") is True}
        retained = [c for c in cases if base.get(c["case_id"], {}).get("prediction") == c["gold_answer"]]
        result["baseline_correct_retention_denominator"] = len(retained)
        result["baseline_correct_retention"] = mean([case_correct[c["case_id"]] for c in retained])
        result["retention_definition"] = "accuracy conditional on a correct full-context unedited baseline; not closed-book clinical agreement"
    return result


def paired_comparison(cases, left, right):
    expected = {c["case_id"] for c in cases}
    if any(r.get("case_id") in expected and r.get("status") == "zero_control_failed" for r in left + right):
        return {"status": "invalid_zero_control", "paired_accuracy_difference": None}
    indexes = []
    for records in (left, right):
        indexes.append({r["case_id"]: r["prediction"] for r in records
                        if r.get("status") == "completed" and r.get("is_model_prediction") is True
                        and r.get("prediction") in set("ABCD")})
    if not all(expected <= set(index) for index in indexes):
        return {"status": "blocked_or_incomplete", "paired_accuracy_difference": None}
    pairs = defaultdict(list)
    for case in cases:
        pairs[case["pair_id"]].append(case)
    differences = []
    for pair in pairs.values():
        if len(pair) != 2:
            raise ValueError("Paired comparison contains incomplete pairs")
        correct = [int(all(index[c["case_id"]] == c["gold_answer"] for c in pair)) for index in indexes]
        differences.append((pair[0]["family_id"], correct[0] - correct[1], 1))
    return {"status": "completed_actual_prediction_comparison", "pairs": len(differences),
            "families": len({x[0] for x in differences}),
            "paired_accuracy_difference": mean([x[1] for x in differences]),
            "ci95_family_bootstrap_exploratory": cluster_interval(differences),
            "multiplicity_adjusted": False}


def gate_risk_coverage(cases, records):
    """Diagnostic reference comparisons, performed only after prediction logs exist."""
    case_map = {c["case_id"]: c for c in cases}
    record_map = {r["case_id"]: r for r in records if r.get("case_id") in case_map}
    evaluated = []
    failures = set()
    missing, condition_failures, available_cases = 0, 0, 0
    for case in cases:
        record = record_map.get(case["case_id"], {})
        completed = record.get("status") == "completed" and record.get("is_model_prediction") is True
        if not completed:
            missing += 1
        structure = record.get("structure", {})
        if structure.get("parse_failure") or structure.get("status") == "parse_failure":
            failures.add(case["case_id"])
        condition_failures += len(structure.get("condition_parse_failures", []))
        available = completed and structure.get("status") == "ok"
        available_cases += int(available)
        truth = {r["rule_id"]: r["state"] for r in case["reference_structure"]["rules"]}
        predicted = {r["rule_id"]: r for r in structure.get("rules", [])} if available else {}
        for rid, reference in truth.items():
            row = predicted.get(rid, {})
            evaluated.append({"allowed": bool(row.get("allow_attention_intervention")),
                              "gate_weight": row.get("gate_weight", row.get("confidence") or 0.0),
                              "correct": row.get("state") == reference,
                              "available": row.get("state") in {"T", "F", "U"}})
    if not evaluated:
        return {"status": "blocked_or_not_run"}
    curve = []
    for threshold in (0.0, .25, .5, .75, 1.0):
        accepted = [r for r in evaluated if r["allowed"] and r["gate_weight"] >= threshold]
        curve.append({"minimum_gate_weight": threshold, "accepted_rules": len(accepted),
                      "coverage": len(accepted) / len(evaluated),
                      "wrong_state_risk": mean([not r["correct"] for r in accepted])})
    return {"status": "completed_reference_diagnostic", "evaluated_rule_opportunities": len(evaluated),
            "expected_cases": len(cases), "available_structures_with_model_output": available_cases,
            "output_completeness": (len(cases) - missing) / len(cases) if cases else None,
            "missing_or_failed_model_outputs": missing,
            "parse_failure_cases": len(failures), "condition_parse_failure_occurrences": condition_failures,
            "curve": curve,
            "note": "Binary evidence gates are not claimed to be calibrated probabilities."}


def evaluate(cases, predictions, split="confirmation"):
    cases = [c for c in cases if c["split"] == split]
    allowed = {c["case_id"] for c in cases}
    records = [r for r in predictions if r.get("case_id") in allowed]
    keys = [(r["case_id"], r["source"], r["mode"]) for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate case/source/mode records; results may not be overwritten silently")
    arms = defaultdict(list)
    for record in records:
        arms[record["source"], record["mode"]].append(record)
    baseline = arms.get(("none", "baseline"), [])
    result = {"scope": "new invented mechanism test frozen before model forward; development-exposed schema and wording; not blinded or clinical confirmation",
              "split": split, "family_count": len({c["family_id"] for c in cases}),
              "all_variants": {}, "primary_canonical_pairs": {}, "gate_diagnostics": {}}
    for (source, mode), rows in sorted(arms.items()):
        name = source + "/" + mode
        result["all_variants"][name] = score_arm(cases, rows, baseline)
        canonical = [c for c in cases if c["variant"] == "canonical" and c.get("pair_eligible")]
        result["primary_canonical_pairs"][name] = score_arm(canonical, rows, baseline)
        if mode == "structured_prompt" and source != "none":
            result["gate_diagnostics"][source] = gate_risk_coverage(cases, rows)
    canonical = [c for c in cases if c["variant"] == "canonical" and c.get("pair_eligible")]
    comparisons = [(("v7", "structured_prompt"), ("v6", "structured_prompt")),
                   (("v7", "signed_bias"), ("v7", "structured_prompt")),
                   (("v7", "learned_subspace"), ("v7", "signed_bias")),
                   (("reference_positive", "signed_bias"), ("v7", "signed_bias")),
                   (("reference", "signed_bias"), ("reference_positive", "signed_bias"))]
    result["prespecified_canonical_pair_comparisons"] = [
        {"left": "/".join(left), "right": "/".join(right),
         **paired_comparison(canonical, arms.get(left, []), arms.get(right, []))}
        for left, right in comparisons]
    result["interpretation"] = (
        "H-Acc is the harmonic mean of app/non case accuracy; paired accuracy requires both cases correct. "
        "Surface variants share families and are not independent confirmations. Oracle/reference arms are diagnostic. "
        "A blocked arm has null metrics, never fabricated predictions."
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", choices=["train", "dev", "confirmation"], default="confirmation")
    args = parser.parse_args()
    result = evaluate(read_jsonl(args.cases), read_jsonl(args.predictions), args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "evaluated_saved_real_logs", "arms": list(result["all_variants"])}, indent=2))


if __name__ == "__main__":
    main()
