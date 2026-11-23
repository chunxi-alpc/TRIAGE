#!/usr/bin/env python3
"""Offline gate-only replay of frozen V6 predictions; no model inference."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import random

from common import BASELINE_COMMIT, load_modules, read_jsonl, sha256, source_hashes, write_json, write_jsonl


def evidence_ids(prediction):
    return set(span["sentence_id"] for c in prediction["conditions"]
               for key in ("support_spans", "refute_spans", "numeric_evidence_spans")
               for span in c.get(key, []) if span.get("sentence_id") is not None)


def evidence_correct(ids, label):
    if label["label"] in ("T", "F"):
        return any(set(proof) <= ids for proof in label.get("acceptable_evidence_proofs", [])) and ids <= set(label.get("all_relevant_evidence_ids", []))
    conflicts = set(label.get("conflict_evidence_ids", []))
    return ids == conflicts if conflicts else not ids


def rates(rows, gate_key, evidence_key):
    selected = [r for r in rows if r[gate_key]]
    label_correct = sum(r["label_correct"] for r in selected)
    ev_correct = sum(r[evidence_key] for r in selected)
    joint = sum(r["label_correct"] and r[evidence_key] for r in selected)
    correct_t = sum(r["gold"] == r["prediction"] == "T" for r in rows)
    return {"n": len(rows), "opened": len(selected), "coverage": len(selected) / len(rows),
        "opened_label_correct": label_correct, "opened_label_accuracy": label_correct / len(selected) if selected else None,
        "opened_evidence_correct": ev_correct, "opened_evidence_accuracy": ev_correct / len(selected) if selected else None,
        "opened_joint_correct": joint, "opened_joint_accuracy": joint / len(selected) if selected else None,
        "incorrect_opening": len(selected) - label_correct,
        "correct_T_n": correct_t, "correct_T_opened": label_correct,
        "correct_T_gate_coverage": label_correct / correct_t if correct_t else None}


def cluster_interval(rows, gate_key, evidence_key, seed=20261019, draws=2000):
    groups = defaultdict(list)
    for row in rows:
        groups[row["anchor_id"]].append(row)
    anchors = sorted(groups)
    rng = random.Random(seed)
    precision, coverage = [], []
    for _ in range(draws):
        sample = [r for key in rng.choices(anchors, k=len(anchors)) for r in groups[key]]
        stat = rates(sample, gate_key, evidence_key)
        if stat["opened_joint_accuracy"] is not None:
            precision.append(stat["opened_joint_accuracy"])
        coverage.append(stat["coverage"])
    def interval(values):
        values = sorted(values)
        return [values[int((len(values)-1)*q)] for q in (.025, .975)] if values else [None, None]
    return {"cluster": "source_case_anchor_all_variants_together", "clusters": len(anchors), "draws": draws,
            "seed": seed, "opened_joint_accuracy_95pct": interval(precision), "coverage_95pct": interval(coverage),
            "interpretation": "descriptive_bootstrap_of_development_cache_not_independent_generalization"}


def evaluate(source, cohort, v7):
    label_name = "labels_public.jsonl" if cohort == "old_192" else "new_labels_100.jsonl"
    rule_name = "rules_v5.jsonl" if cohort == "old_192" else "new_rules_100.jsonl"
    predictions = read_jsonl(source / (cohort + "_predictions_v6.jsonl"))
    labels = {r["sample_id"]: r for r in read_jsonl(source / label_name)}
    rules = {r["rule_id"]: r for r in read_jsonl(source / rule_name)}
    if len(labels) != len(predictions) or set(labels) != {p["sample_id"] for p in predictions}:
        raise ValueError("prediction/label IDs must align exactly")
    rows = []
    for prediction in predictions:
        label = labels[prediction["sample_id"]]
        # The V6 recomposition policy, not the obsolete V5 condition status,
        # determines atom states. No gold atoms or evidence are used for gates.
        states = {c["condition_id"]: c["numeric_status"] if c.get("numeric_status") in ("T", "F") else c.get("trialmatch_state", "U")
                  for c in prediction["conditions"]}
        gate = v7.rule_evidence_gate(rules[prediction["rule_id"]], prediction["conditions"], states=states)
        mismatch = gate["state"] != prediction["C"]
        if mismatch:
            gate["allow_attention_intervention"] = False
            gate["reason"] = "cached_rule_state_reconstruction_mismatch"
        all_ids = evidence_ids(prediction)
        proof_ids = {s["sentence_id"] for s in gate["proof_evidence"]}
        variant = label["variant"]
        anchor = label.get("anchor_id") or label["sample_id"].removesuffix("_" + variant)
        rows.append({"sample_id": prediction["sample_id"], "anchor_id": anchor,
            "rule_id": prediction["rule_id"], "variant": variant,
            "gold": label["label"], "prediction": prediction["C"], "label_correct": prediction["C"] == label["label"],
            "saved_gate": bool(prediction["allow_attention_intervention"]),
            "v7_proof_gate": gate["allow_attention_intervention"], "gate_reason": gate["reason"],
            "state_reconstruction_matches_saved": not mismatch,
            "all_cached_evidence_ids": sorted(all_ids), "selected_proof_evidence_ids": sorted(proof_ids),
            "all_cached_evidence_correct": evidence_correct(all_ids, label),
            "selected_proof_evidence_correct": evidence_correct(proof_ids, label),
            "selected_proof": gate["selected_proof"],
            "condition_checks": {k: {"allowed": x["allowed"], "reason": x["reason"], "offset_verification": x["offset_verification"]} for k, x in gate["condition_checks"].items()}})
    metrics = {"n": len(rows), "labels_and_models_unchanged": True,
        "applicability_correct": sum(r["label_correct"] for r in rows),
        "state_reconstruction_mismatches": sum(not r["state_reconstruction_matches_saved"] for r in rows),
        "numeric_TF_atoms_in_original_cache": sum(c.get("numeric_status") in ("T", "F") for p in predictions for c in p["conditions"]),
        "numeric_atoms_total": sum(len(p["conditions"]) for p in predictions),
        "gate_changes": {"opened_to_closed": sum(r["saved_gate"] and not r["v7_proof_gate"] for r in rows),
                         "closed_to_opened": sum(not r["saved_gate"] and r["v7_proof_gate"] for r in rows)},
        "gate_reasons": dict(Counter(r["gate_reason"] for r in rows)), "policies": {}}
    for name, gate_key, evidence_key in [("saved_V6_gate_all_cached_evidence", "saved_gate", "all_cached_evidence_correct"),
        ("V7_proof_gate_all_cached_evidence", "v7_proof_gate", "all_cached_evidence_correct"),
        ("V7_proof_gate_selected_proof_evidence", "v7_proof_gate", "selected_proof_evidence_correct")]:
        metrics["policies"][name] = dict(rates(rows, gate_key, evidence_key),
            cluster_bootstrap=cluster_interval(rows, gate_key, evidence_key))
    return rows, metrics, [source / x for x in (label_name, rule_name, cohort + "_predictions_v6.jsonl")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True, help="Frozen experiments/care_module1_v6/data directory")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    _, v7 = load_modules()
    result = {"run_type": "offline_saved_prediction_gate_replay_not_new_model_inference",
        "baseline_commit": BASELINE_COMMIT, "created_at": datetime.now(timezone.utc).isoformat(),
        "source_hashes": source_hashes(), "raw_patient_text_available": False,
        "numeric_scope_guard_recomputed": False, "offset_validation": "structure_only",
        "gold_used_for_gate_decisions": False, "expert_review": False,
        "limitations": ["Evidence labels and model predictions are reused; V7 clinical effectiveness is not established.",
            "No raw patient text: numeric subject/time guard and exact evidence text cannot be replayed.",
            "This evaluates gate and proof selection, not the complete V7 inference pipeline.",
            "Development/cache data contain correlated variants; bootstrap groups variants by anchor."],
        "input_sha256": {}, "cohorts": {}}
    for cohort in ("old_192", "new_100"):
        rows, metrics, paths = evaluate(args.source_dir, cohort, v7)
        result["cohorts"][cohort] = metrics
        result["input_sha256"].update({p.name: sha256(p) for p in paths})
        write_jsonl(args.output / (cohort + "_gate_predictions.jsonl"), rows)
    write_json(args.output / "metrics.json", result)
    for cohort, metric in result["cohorts"].items():
        print(cohort, {name: {key: p[key] for key in ("opened", "opened_label_correct", "opened_joint_correct")} for name, p in metric["policies"].items()})
    if any(m["state_reconstruction_mismatches"] for m in result["cohorts"].values()):
        raise SystemExit("Some cached rule states do not reconstruct. Gates closed and marked; inspect before interpretation.")


if __name__ == "__main__":
    main()
