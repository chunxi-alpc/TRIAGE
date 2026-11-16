#!/usr/bin/env python3
"""Build VERIFIED patient-attribute flips without synthesizing clinical action gold.

Rule applicability here is only a numeric/text-template protocol check. It does
not certify the clinical recommendation or show parametric-context conflict.
"""
import argparse
import hashlib
import json
from pathlib import Path

from prepare import jsonl_bytes, read_jsonl, split_for

OPERATORS = {">=": lambda x, v: x >= v, ">": lambda x, v: x > v,
             "<=": lambda x, v: x <= v, "<": lambda x, v: x < v,
             "==": lambda x, v: x == v}


def generate(rows, seed=20261019):
    inputs, labels = [], []
    families = set()
    for r in rows:
        needed = ("family_id", "question", "rule_text", "patient_template", "condition", "values", "provenance")
        if any(k not in r for k in needed):
            raise ValueError("Missing verified conditional annotation fields")
        src = r["provenance"]
        if src.get("verified") is not True or not src.get("reviewer") or not src.get("source_url"):
            raise ValueError("Require explicit source URL and independent reviewed annotation; no invented clinical gold")
        family = str(r["family_id"])
        if family in families:
            raise ValueError(f"Duplicate conditional family {family}")
        families.add(family)
        pred = r["condition"]
        if not all(k in pred for k in ("field", "operator", "threshold", "unit")):
            raise ValueError("Numeric predicate incomplete")
        if pred["operator"] not in OPERATORS or isinstance(pred["threshold"], bool) or not isinstance(pred["threshold"], (int, float)):
            raise ValueError("Only explicit numeric predicates supported")
        if r["patient_template"].count("{value}") != 1:
            raise ValueError("Patient template must contain exactly one variable")
        if set(r["values"]) != {"T", "F"}:
            raise ValueError("Both minimal-pair sides T/F required")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in r["values"].values()):
            raise ValueError("Patient flip values must be numeric")
        test = OPERATORS[pred["operator"]]
        if not test(r["values"]["T"], pred["threshold"]) or test(r["values"]["F"], pred["threshold"]):
            raise ValueError("Counterfactual values do not flip applicability")
        split = split_for("cond:" + family, seed)
        for state in ("T", "F", "U"):
            value = r["values"].get(state)
            # The unknown variant is missing evidence, NOT a negative finding.
            observation = r["patient_template"].replace("{value}", str(value) if value is not None else "not recorded")
            question = r["question"]
            rule = r["rule_text"]
            prompt = (f"Question: {question}\nPatient evidence: {observation}\n"
                      f"Guideline rule: {rule}\n"
                      "Task: Does this rule's stated numeric condition apply to this patient? "
                      "Do not infer a clinical action if it is inapplicable or unknown.\n"
                      "Choices: A. applicable  B. inapplicable  C. unknown\nAnswer:")
            p0 = prompt.index(observation)
            r0 = prompt.index(rule, p0 + len(observation))
            item_id = hashlib.sha256(f"{family}|{state}".encode()).hexdigest()[:20]
            inputs.append({"id": item_id, "group": "cond:" + family, "split": split, "track": "conditional_applicability",
                           "prompt": prompt, "evidence_span": [p0, p0 + len(observation)],
                           "rule_span": [r0, r0 + len(rule)], "question": question,
                           "provenance": {"source_url": src["source_url"], "reviewer": src["reviewer"],
                                          "verified": True, "predicate": pred}})
            labels.append({"id": item_id, "group": "cond:" + family, "split": split,
                           "state": state, "correct_choice": {"T": "A", "F": "B", "U": "C"}[state],
                           "clinical_action_gold": None, "expert_action_gold_available": False})
    return inputs, labels


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    inp, lbl = generate(list(read_jsonl(a.annotations)))
    a.out.mkdir(parents=True, exist_ok=False)
    a.out.joinpath("conditional_inputs.jsonl").write_bytes(jsonl_bytes(inp))
    a.out.joinpath("conditional_labels.jsonl").write_bytes(jsonl_bytes(lbl))
    manifest = {"status": "VERIFIED_ANNOTATION_STRUCTURE_PREPARED_NOT_MODEL_EVALUATED",
                "families": len(inp) // 3, "rows": len(inp),
                "source_sha256": hashlib.sha256(a.annotations.read_bytes()).hexdigest(),
                "warning": "No clinical action gold; applicability only; no intervention was performed."}
    a.out.joinpath("conditional_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
