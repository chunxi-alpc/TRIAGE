"""Strict offline evaluation of redacted predictions against frozen labels."""
from __future__ import annotations
import argparse, collections, json, pathlib

LABELS = ("T", "F", "U")


def read_jsonl(path):
    return [json.loads(line) for line in pathlib.Path(path).read_text().splitlines() if line.strip()]


def unique_index(rows, name):
    ids = [row.get("sample_id") for row in rows]
    if any(not sample_id for sample_id in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"{name} has missing or duplicate sample IDs")
    return {row["sample_id"]: row for row in rows}


def f1_macro(gold, pred):
    if not gold:
        return None
    scores = []
    for label in LABELS:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return sum(scores) / len(scores)


def all_evidence_ids(row, method):
    if method == "qwen_v4":
        return set(row.get("evidence_sentence_ids", []))
    ids = set()
    for condition in row.get("conditions", []):
        if method == "trialmatch_model_only":
            continue
        if method == "trialmatch":
            ids.update(span["sentence_id"] for span in condition.get("numeric_evidence_spans", []))
        else:
            ids.update(span["sentence_id"] for span in condition.get("numeric_evidence_spans", []))
            for key in ("support_spans", "refute_spans"):
                ids.update(span["sentence_id"] for span in condition.get(key, []) if "sentence_id" in span)
    return ids


def evidence_correct_ids(predicted, label):
    predicted = set(predicted)
    if label["label"] in {"T", "F"}:
        proofs = label.get("acceptable_evidence_proofs", [])
        relevant = set(label.get("all_relevant_evidence_ids", []))
        return bool(proofs) and any(set(proof).issubset(predicted) for proof in proofs) and predicted.issubset(relevant)
    conflicts = set(label.get("conflict_evidence_ids", []))
    if conflicts:
        return conflicts == predicted
    return not predicted


def validate_pairs(labels, pairs):
    label_by_id = unique_index(labels, "labels")
    if not pairs or any(not row.get("pair_id") for row in pairs) or len({row.get("pair_id") for row in pairs}) != len(pairs):
        raise ValueError("expected nonempty unique explicit counterfactual pairs")
    covered = set()
    for pair in pairs:
        original, counter = pair.get("original_id"), pair.get("counterfactual_id")
        if original not in label_by_id or counter not in label_by_id:
            raise ValueError(f"pair {pair.get('pair_id')} references an unknown sample")
        if label_by_id[original].get("variant") != "original" or label_by_id[counter].get("variant") != "counterfactual":
            raise ValueError(f"pair {pair.get('pair_id')} has invalid variant assignments")
        if original in covered or counter in covered:
            raise ValueError("sample appears in multiple counterfactual pairs")
        covered.update((original, counter))
        if label_by_id[original].get("pair_id") != pair["pair_id"] or label_by_id[counter].get("pair_id") != pair["pair_id"]:
            raise ValueError(f"pair ID mismatch for {pair['pair_id']}")
    if len(covered) != 2 * len(pairs):
        raise ValueError("counterfactual pair coverage is incomplete")
    return label_by_id


def predicted_state(row, method):
    key = {"trialmatch_model_only": "trialmatch_model_only_applicability",
           "trialmatch": "trialmatch_applicability", "combined": "applicability",
           "qwen_v4": "label"}[method]
    return row.get(key, "U")


def method_metrics(labels, rows_by_id, method, pairs):
    label_by_id = unique_index(labels, "labels")
    extra = set(rows_by_id) - set(label_by_id)
    if extra:
        raise ValueError(f"{method} predictions contain unexpected sample IDs: {sorted(extra)[:5]}")
    missing = set(label_by_id) - set(rows_by_id)
    if missing:
        return {"status": "incomplete", "predictions": len(rows_by_id), "expected": len(labels),
                "missing_predictions": len(missing), "accuracy": None, "macro_f1": None,
                "known_evidence_accuracy": None, "joint_accuracy": None}
    gold = [label["label"] for label in labels]
    pred = [predicted_state(rows_by_id[label["sample_id"]], method) for label in labels]
    if any(value not in LABELS for value in pred):
        bad = sorted(set(pred) - set(LABELS))
        raise ValueError(f"{method} returned invalid labels: {bad}")
    correct = sum(g == p for g, p in zip(gold, pred))
    known = [label for label in labels if label["label"] in {"T", "F"}]
    evidence_available = method != "trialmatch_model_only"
    if evidence_available:
        evidence_flags = {label["sample_id"]: evidence_correct_ids(
            all_evidence_ids(rows_by_id[label["sample_id"]], method), label) for label in labels}
        evidence_known_correct = sum(evidence_flags[row["sample_id"]] for row in known)
        joint = sum(predicted_state(rows_by_id[row["sample_id"]], method) == row["label"] and evidence_flags[row["sample_id"]] for row in labels)
        all_evidence_correct = sum(evidence_flags.values())
    else:
        evidence_flags = {}
        evidence_known_correct = joint = all_evidence_correct = None
    pred_by_id = {label["sample_id"]: predicted_state(rows_by_id[label["sample_id"]], method) for label in labels}
    exact_pairs, flip_pairs = 0, 0
    for pair in pairs:
        left, right = pair["original_id"], pair["counterfactual_id"]
        exact_pairs += int(pred_by_id[left] == label_by_id[left]["label"] and pred_by_id[right] == label_by_id[right]["label"])
        flip_pairs += int(pred_by_id[left] != pred_by_id[right])
    counts = collections.Counter()
    for g, p in zip(gold, pred):
        if g == p: counts["correct"] += 1
        elif p == "U" and g != "U": counts["known_to_unknown"] += 1
        elif g == "U" and p != "U": counts["unknown_to_known"] += 1
        elif g == "T" and p == "F": counts["false_rejection_T_to_F"] += 1
        elif g == "F" and p == "T": counts["false_acceptance_F_to_T"] += 1
        else: counts["known_polarity_error"] += 1
    if evidence_available:
        counts["evidence_error_known"] = len(known) - evidence_known_correct
        counts["joint_correct"] = joint
    scenario = {}
    for variant in sorted({row.get("variant", "unknown") for row in labels}):
        selected = [row for row in labels if row.get("variant") == variant]
        scenario[variant] = {"n": len(selected), "correct": sum(pred_by_id[row["sample_id"]] == row["label"] for row in selected),
                             "unknown_predictions": sum(pred_by_id[row["sample_id"]] == "U" for row in selected)}
    if method == "trialmatch_model_only":
        latencies = [rows_by_id[row["sample_id"]].get("trialmatch_seconds") for row in labels]
    elif method == "trialmatch":
        latencies = [rows_by_id[row["sample_id"]].get("trialmatch_seconds", 0) + rows_by_id[row["sample_id"]].get("numeric_seconds", 0) for row in labels]
    elif method == "combined":
        latencies = [rows_by_id[row["sample_id"]].get("elapsed_seconds") for row in labels]
    else:
        direct = 79.19770555506693
        latencies = [direct / len(labels)] * len(labels)
    latencies = [value for value in latencies if isinstance(value, (int, float))]
    confusion = [[sum(g == gold_label and p == pred_label for g, p in zip(gold, pred)) for pred_label in LABELS] for gold_label in LABELS]
    result = {"status": "complete", "n": len(gold), "correct": correct, "accuracy": correct / len(gold),
        "macro_f1": f1_macro(gold, pred), "evidence_available": evidence_available,
        "known_evidence_accuracy": evidence_known_correct / len(known) if evidence_available and known else None,
        "known_evidence_correct": evidence_known_correct, "known_evidence_n": len(known) if evidence_available else None,
        "all_sample_evidence_correct": all_evidence_correct, "joint_correct": joint,
        "joint_accuracy": joint / len(labels) if evidence_available else None,
        "counterfactual_pair_accuracy": exact_pairs / len(pairs), "counterfactual_pairs_correct": exact_pairs,
        "counterfactual_pair_n": len(pairs), "raw_prediction_flip_rate": flip_pairs / len(pairs),
        "mean_elapsed_seconds": sum(latencies) / len(latencies) if latencies else None,
        "total_inference_seconds": sum(latencies) if latencies else None,
        "unknown_predictions": sum(p == "U" for p in pred), "gold_unknown": sum(g == "U" for g in gold),
        "error_categories": dict(counts), "confusion_order": list(LABELS), "confusion": confusion,
        "by_variant": scenario}
    if method in {"combined", "qwen_v4"}:
        result["conflicts"] = sum(bool(c.get("conflict")) for row in rows_by_id.values() for c in row.get("conditions", []))
        result["conflict_samples"] = sum(any(c.get("conflict") for c in row.get("conditions", [])) for row in rows_by_id.values())
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", type=pathlib.Path, required=True)
    p.add_argument("--pairs", type=pathlib.Path, required=True)
    p.add_argument("--predictions", type=pathlib.Path, required=True)
    p.add_argument("--qwen-reference", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    p.add_argument("--errors", type=pathlib.Path, required=True)
    a = p.parse_args()
    labels, pairs = read_jsonl(a.labels), read_jsonl(a.pairs)
    label_by_id = validate_pairs(labels, pairs)
    prediction_rows = read_jsonl(a.predictions) if a.predictions.exists() else []
    qwen_rows = read_jsonl(a.qwen_reference)
    predictions = unique_index(prediction_rows, "V5 predictions")
    qwen = unique_index(qwen_rows, "V4 direct predictions")
    expected_ids = set(label_by_id)
    if set(qwen) != expected_ids:
        raise ValueError("V4 saved prediction IDs do not exactly match frozen labels")
    if set(predictions) - expected_ids:
        raise ValueError("V5 prediction IDs include rows absent from the frozen labels")
    results = {"protocol": "exploratory_same_set_regression; V4 was used in method design",
               "sample_count": len(labels), "methods": {}}
    for method, rows in (("trialmatch_model_only", predictions), ("trialmatch", predictions),
                         ("combined", predictions), ("qwen_v4", qwen)):
        results["methods"][method] = method_metrics(labels, rows, method, pairs)
    if predictions:
        compared = [(label, predictions[label["sample_id"]]) for label in labels if label["sample_id"] in predictions]
        paired = [(label, row) for label, row in compared]
        a_ok = [row.get("trialmatch_applicability", "U") == label["label"] for label, row in paired]
        b_ok = [row.get("applicability", "U") == label["label"] for label, row in paired]
        results["paired_trialmatch_vs_combined"] = {
            "n": len(paired), "combined_rescued": sum(b and not a for a, b in zip(a_ok, b_ok)),
            "combined_regressed": sum(a and not b for a, b in zip(a_ok, b_ok)),
            "prediction_changed": sum(row.get("trialmatch_applicability", "U") != row.get("applicability", "U") for _, row in paired),
            "unknown_to_known": sum(row.get("trialmatch_applicability") == "U" and row.get("applicability") in {"T", "F"} for _, row in paired),
            "known_to_unknown": sum(row.get("trialmatch_applicability") in {"T", "F"} and row.get("applicability") == "U" for _, row in paired)}
    errors = []
    for label in labels:
        row = predictions.get(label["sample_id"])
        if row is None:
            continue
        for method in ("trialmatch_model_only", "trialmatch", "combined"):
            guess = predicted_state(row, method)
            evidence_ok = None if method == "trialmatch_model_only" else evidence_correct_ids(all_evidence_ids(row, method), label)
            decision_wrong = guess != label["label"]
            if decision_wrong or evidence_ok is False:
                category = ("known_to_unknown" if guess == "U" and label["label"] != "U" else
                    "unknown_to_known" if label["label"] == "U" and guess != "U" else
                    "false_rejection_T_to_F" if (label["label"], guess) == ("T", "F") else
                    "false_acceptance_F_to_T" if (label["label"], guess) == ("F", "T") else
                    "evidence_localization_error" if not decision_wrong else "known_polarity_error")
                errors.append({"sample_id": label["sample_id"], "rule_id": label["rule_id"],
                    "variant": label.get("variant"), "method": method,
                    "gold": label["label"], "prediction": guess,
                    "evidence_correct": evidence_ok, "error_category": category,
                    "predicted_evidence_ids": sorted(all_evidence_ids(row, method)) if evidence_ok is not None else None,
                    "sufficient_gold_evidence": label.get("acceptable_evidence_proofs", []),
                    "conflict_evidence_ids": label.get("conflict_evidence_ids", []),
                    "condition_diagnostics": [{k: condition.get(k) for k in (
                        "condition_id", "trialmatch_state", "trialmatch_parse_error", "trialmatch_truncated",
                        "numeric_status", "nli_evidence_state", "state", "resolution", "conflict")}
                        for condition in row.get("conditions", [])]})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    a.errors.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in errors) + ("\n" if errors else ""))
    print(json.dumps({name: {k: val for k, val in row.items() if k in ("status", "correct", "n", "accuracy", "macro_f1", "known_evidence_accuracy", "joint_accuracy", "counterfactual_pair_accuracy")} for name, row in results["methods"].items()}, ensure_ascii=False))

if __name__ == "__main__":
    main()
