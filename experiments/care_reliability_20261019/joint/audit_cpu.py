#!/usr/bin/env python3
"""Audit frozen labels and V6/V7 numeric input compatibility without model calls."""
from __future__ import annotations

import hashlib
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT), str(EXPERIMENT.parents[1])]
from joint.context import prediction_input, full_context
from joint.evaluate import read_jsonl
from joint.logic import reference_structure
from src.module1.care_v6 import validate_v6_input, _patient_spans
from src.module1.care_v7 import scoped_numeric_observation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=EXPERIMENT / "joint/cpu_audit.json")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output report already exists; choose a new path. No file was overwritten.")
    path = EXPERIMENT / "data/mechanism_cases.jsonl"
    cases = read_jsonl(path)
    audit = []
    for case in cases:
        validate_v6_input(**prediction_input(case))
        if reference_structure(case["rules"], case["structured_facts"]) != case["reference_structure"]:
            raise AssertionError("Reference recomputation differs")
        if full_context(case)[0] != case["prompt"]:
            raise AssertionError("Complete prompt differs")
        spans = _patient_spans(case["patient"], case["patient"])
        atoms = {a["condition_id"]: a for r in case["rules"] for a in r["conditions"] + r["exceptions"] if a.get("numeric")}
        for cid, condition in atoms.items():
            measured = scoped_numeric_observation(condition, spans)
            reference = case["reference_structure"]["atoms"][cid]["state"]
            audit.append({"case_id": case["case_id"], "split": case["split"], "condition_id": cid,
                          "reference_state": reference, "numeric_state": measured["status"],
                          "numeric_reason": measured["reason"], "agrees": reference == measured["status"]})
    report = {
        "status": "completed_cpu_logic_schema_and_numeric_compatibility_audit",
        "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "v7_source_sha256": hashlib.sha256((EXPERIMENT.parents[1] / "src/module1/care_v7.py").read_bytes()).hexdigest(),
        "cases": len(cases), "schema_validated_cases": len(cases), "oracle_recomputed_cases": len(cases),
        "numeric_atomic_checks": len(audit), "numeric_reason_counts": dict(Counter(r["numeric_reason"] for r in audit)),
        "numeric_disagreements": [r for r in audit if not r["agrees"]],
        "numeric_compatibility_only": True, "model_inference_completed": False,
        "clinical_performance_measured": False,
        "no_selection_on_numeric_audit": "All cases and frozen family labels retained, including failures. This checks code/schema compatibility, not the learned module's performance.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "cases", "numeric_atomic_checks", "numeric_reason_counts")}, indent=2))
    print("numeric_disagreements", len(report["numeric_disagreements"]))


if __name__ == "__main__":
    main()
