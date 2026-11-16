#!/usr/bin/env python3
"""Execute deterministic numeric regressions; these are not model benchmarks."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import platform
import sys
import unittest

from common import HERE, BASELINE_COMMIT, load_modules, source_hashes, write_json, write_jsonl


def numeric_cases():
    rows = []
    def add(name, text, field="age", operator=">=", value=18, expected="T", unit=None, category="boundary"):
        spec = {"field": field, "operator": operator, "value": value}
        if unit: spec["unit"] = unit
        rows.append({"case_id": name, "category": category, "patient": text,
                     "condition": {"condition_id": "c", "field": field, "text": f"{field} {operator} {value}", "numeric": spec},
                     "expected_numeric_status": expected})
    for op, expected in [(">=", "T"), (">", "F"), ("<=", "T"), ("<", "F"), ("==", "T"), ("!=", "F")]:
        add("exact_age_" + {">=":"ge",">":"gt","<=":"le","<":"lt","==":"eq","!=":"ne"}[op], "The patient's current age is 18 years.", operator=op, expected=expected)
    add("below_age", "The patient's current age is 17 years.", expected="F")
    add("above_age", "The patient's current age is 19 years.")
    add("age_months", "The patient's current age is 216 months.", category="unit")
    add("age_implicit_word", "The patient is currently 19 years old.", category="wording")
    add("decimal_cm_to_mm", "The patient's current tumor diameter is 0.6 cm.", "tumor_cm", ">=", 6, "T", "mm", "unit")
    add("mm_to_cm", "The patient's current tumor diameter is 40 mm.", "tumor_cm", ">", 4, "F", "cm", "unit")
    add("inch_to_mm", "The patient's current surgical margin is 1 inch.", "margin_mm", "==", 25.4, "T", "mm", "unit")
    add("ngml_to_ngdl", "The patient's current serum testosterone is 0.5 ng/mL.", "testosterone", "<", 50, "F", "ng/dL", "unit")
    add("relapse_years", "The patient's current relapse interval is 0.5 years.", "relapse_months", "<", 6, "F", "month", "unit")
    add("range_cross", "The patient's current age is 17-19 years.", expected="U", category="range")
    add("range_all_true", "The patient's current age is 18-20 years.", category="range")
    add("range_all_false", "The patient's current age is 15 to 17 years.", expected="F", category="range")
    add("range_equality_inside", "The patient's current age is 17-19 years.", operator="==", expected="U", category="range")
    add("range_inequality_inside", "The patient's current age is 17-19 years.", operator="!=", expected="U", category="range")
    add("range_mixed_units", "The patient's current tumor diameter is 3 cm to 40 mm.", "tumor_cm", "<=", 4, "T", "cm", "range")
    add("current_conflict", "The patient's current age is 17 years. The patient's current age is 19 years.", expected="U", category="contradiction")
    add("current_negated_conflict", "The patient's current age is 19 years. The patient's current age is not 19 years.", expected="U", category="contradiction")
    add("current_unknown_conflict", "The patient's current age is 19 years. The patient's current age is unknown.", expected="U", category="contradiction")
    add("family_only", "The patient's mother has a current age of 65 years.", expected="U", category="subject")
    add("sibling_cross_module_regression", "The patient's sibling's current age is 24 years.", expected="U", category="subject")
    add("grandparent_only", "The patient's grandfather's current age is 72 years.", expected="U", category="subject")
    add("another_patient", "Another patient's current age is 24 years.", expected="U", category="subject")
    add("nested_unlisted_owner", "The patient's mentor's current age is 24 years.", expected="U", category="subject")
    add("nested_curly_apostrophe", "The patient’s mentor’s current age is 24 years.", expected="U", category="subject")
    add("mixed_family_sentence", "The patient's current age is 19 years and her mother's current age is 65 years.", expected="U", category="subject")
    add("no_subject", "Current age is 19 years.", expected="U", category="subject")
    add("history_only", "The patient's age was 19 years in 2020.", expected="U", category="time")
    add("time_unspecified", "The patient's age is 19 years.", expected="U", category="time")
    add("mixed_time_sentence", "The patient's current age is 19 years and previously age was 17 years.", expected="U", category="time")
    add("current_over_history", "The patient's previous age was 17 years. The patient's current age is 19 years.", category="time")
    add("separate_family_ignored", "The patient's mother's current age is 65 years. The patient's current age is 17 years.", expected="F", category="subject")
    add("unknown", "The patient's current age is unknown.", expected="U", category="missing")
    add("negated_measurement", "The patient's current age is not 19 years.", expected="U", category="negation")
    add("approximate_measurement", "The patient's current age is about 19 years.", expected="U", category="uncertain")
    add("missing_unit", "The patient's current tumor diameter is 6.", "tumor_cm", ">=", 6, "U", "mm", "unit")
    add("incompatible_unit", "The patient's current age is 19 cm.", expected="U", category="unit")
    add("negative_value", "The patient's current age is -19 years.", expected="U", category="invalid")
    add("unrelated_same_unit", "The patient's current tumor diameter is 3 cm and liver nodule is 8 cm.", "tumor_cm", ">=", 5, "F", "cm", "binding")
    add("liver_boundary", "The patient's current liver nodule diameter is 10 mm.", "liver_mm", ">=", 10, "T", None, "new_field")
    add("lung_boundary", "The patient's current pulmonary nodule diameter is 8 mm.", "lung_mm", "<=", 8, "T", None, "new_field")
    add("solid_unit", "The patient's current solid component is 0.6 cm.", "solid_mm", ">=", 6, "T", None, "new_field")
    add("thyroid_unit", "The patient's current primary thyroid carcinoma is 40 mm.", "thyroid_cm", ">", 4, "F", None, "new_field")
    add("multifocal", "The patient's current additional thyroid focus is 2 cm.", "multifocal_cm", ">", 1, "T", None, "new_field")
    add("pci", "The patient's current PCI is 10.", "pci", "<=", 10, "T", None, "new_field")
    add("vessels", "The patient's current vessel count is 4.", "vessels", ">=", 4, "T", None, "new_field")
    add("unsupported_field", "The patient's current ECOG is 1.", "ecog", "<=", 2, "U", None, "unsupported")
    add("unsupported_chemistry_unit", "The patient's current serum testosterone is 1.7 nmol/L.", "testosterone", "<", 50, "U", "ng/dL", "unsupported")
    add("stage_attached_digit", "The patient's current pT1 status is recorded.", "pt", ">=", 2, "U", None, "unsupported")
    add("stage_plain_number", "The patient's current pT is 2.", "pt", ">=", 2, "T", None, "new_field")
    add("malformed_decimal", "The patient's current age is 19.2.3 years.", expected="U", category="invalid")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing results are never overwritten")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    v6, v7 = load_modules()
    predictions = []
    for case in numeric_cases():
        spans = v6.sentences_with_offsets(case["patient"])
        old = v6._numeric_observation(case["condition"], spans)
        new = v7.scoped_numeric_observation(case["condition"], spans)
        predictions.append(dict(case, v6_numeric=old, v7_numeric=new,
            v6_status=old["status"] if old else "U", v7_status=new["status"] if new else "U"))
    suite = unittest.defaultTestLoader.discover(str(HERE / "tests"), pattern="test_v7.py")
    with (args.output / "unit_tests.txt").open("w") as log:
        tests = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    metrics = {"run_type": "deterministic_synthetic_cpu_diagnostic_not_model_inference",
        "baseline_commit": BASELINE_COMMIT, "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "source_hashes": source_hashes(),
        "n": len(predictions), "expert_review": False, "test_policy": "development_regression_not_held_out",
        "unit_tests": {"run": tests.testsRun, "failures": len(tests.failures), "errors": len(tests.errors)},
        "numeric_methods": {}, "category_counts": dict(Counter(x["category"] for x in predictions))}
    for version in ("v6", "v7"):
        correct = sum(x[version + "_status"] == x["expected_numeric_status"] for x in predictions)
        negative_scope = [x for x in predictions if x["expected_numeric_status"] == "U"]
        metrics["numeric_methods"][version] = {"correct": correct, "accuracy": correct / len(predictions),
            "resolved": sum(x[version + "_status"] in ("T", "F") for x in predictions),
            "unsafe_override_in_abstention_cases": sum(x[version + "_status"] != "U" for x in negative_scope),
            "abstention_cases_n": len(negative_scope),
            "by_category": {c: {"n": sum(x["category"] == c for x in predictions),
                "correct": sum(x["category"] == c and x[version + "_status"] == x["expected_numeric_status"] for x in predictions)} for c in metrics["category_counts"]}}
    write_jsonl(args.output / "numeric_cases_and_predictions.jsonl", predictions)
    write_json(args.output / "metrics.json", metrics)
    print({"numeric": {k: (v["correct"], metrics["n"]) for k, v in metrics["numeric_methods"].items()}, "unit_tests": metrics["unit_tests"]})
    if not tests.wasSuccessful() or metrics["numeric_methods"]["v7"]["correct"] != len(predictions):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
