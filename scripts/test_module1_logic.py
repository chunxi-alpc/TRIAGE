#!/usr/bin/env python3
"""Small CPU-only checks; the stub verifies control flow, not model quality."""
import json
import re
import sys
import tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.module1 import Module1
from src.module1.conditions import match_conditions

class StubBackend:
    cache_identity = {"model": "cpu_logic_test_stub"}
    def __init__(self):
        self.calls = []
    def score_pairs(self, question, clauses):
        return [0.8] * len(clauses)
    def json(self, system, user, max_new_tokens=0):
        self.calls.append({"purpose": "CPU logic stub"})
        if "Independently judge" in system or "Judge whether each complete" in system:
            data=json.loads(user)
            return {row["condition_id"]:{"condition":row["condition"],
                "predicate_truth":"true" if "diabetes" in row["condition"].lower() else "unknown",
                "evidence_relation":"entails" if "diabetes" in row["condition"].lower() else "insufficient",
                "evidence_sentence_ids":[0] if "diabetes" in row["condition"].lower() else [],
                "reason":"Controlled explicit denial"} for row in data["conditions"]}
        if "Compare two independently" in system:
            data = json.loads(user)
            return {"rules": [{"rule_id": row["rule_id"], "prior_support": "oppose",
                "guided_support": "support", "reason": "Synthetic test decision contrast"}
                for row in data["rule_actions"]]}
        if "clinical decision model" in system:
            return {"decision": "Synthetic test decision", "actions": ["Synthetic action"],
                "status": "decided", "rationale": "CPU control flow verification"}
        section = user.split("Here are the guideline inclusion conditions:\n", 1)[1].split("\n\nPlain JSON")[0]
        output = {}
        for index, criterion in re.findall(r"(?m)^(\d+)\. (.*)$", section):
            if "diabetes" in criterion.lower():
                output[index] = ["Explicit denial in patient sentence", [0], "included"]
            else:
                output[index] = ["No supporting patient evidence", [], "not enough information"]
        return output

def condition(cid, field, operator, value, unit):
    return {"condition_id": cid, "text": f"{field} {operator} {value} {unit}",
        "numeric": {"field": field, "operator": operator, "value": value, "unit": unit}}

def rule(rid, conditions, logic="AND", exceptions=None):
    return {"rule_id": rid, "text": f"Clinical test rule {rid}", "conditions": conditions,
        "logic": logic, "exceptions": exceptions or [], "action": f"Action {rid}"}

def check_span_positions(patient, spans):
    for span in spans:
        assert patient[span["start"]:span["end"]] == span["text"]

def main():
    backend = StubBackend()
    records = []
    cases = [
        ("age_with_unrelated_duration", "A 58-year-old patient became unwell 2 days ago.",
            [condition("age", "age", ">=", 18, "years")], "satisfied"),
        ("numeric_below_threshold", "The patient is 16 years old.",
            [condition("age", "age", ">=", 18, "years")], "not_satisfied"),
        ("glucose_unit_conversion", "Glucose is 180 mg/dL.",
            [condition("glucose", "glucose", ">=", 9, "mmol/L")], "satisfied"),
        ("numeric_missing", "A patient has fatigue; laboratory results are pending.",
            [condition("egfr", "egfr", ">=", 60, "mL/min/1.73 m2")], "unknown"),
        ("explicit_negation", "The patient denies diabetes.",
            [{"condition_id": "absence", "text": "No diabetes", "negated": False}], "satisfied"),
        ("semantic_missing", "The patient reports fatigue.",
            [{"condition_id": "missing", "text": "History of heart failure"}], "unknown"),
    ]
    for name, patient, conditions, expected in cases:
        before = len(backend.calls)
        relations, applicability, spans = match_conditions([rule(name, conditions)], patient, backend)
        assert relations[0]["status"] == expected, (name, relations)
        check_span_positions(patient, spans)
        if name.startswith("numeric") or name in {"age_with_unrelated_duration", "glucose_unit_conversion"}:
            assert len(backend.calls) == before, "Numeric checks must not invoke language models"
        records.append({"case": name, "relations": relations, "applicability": applicability, "passed": True})

    before = len(backend.calls)
    negative_patient = "The patient denies diabetes."
    shared_rules = [rule("absence", [{"condition_id": "c", "text": "No diabetes", "negated": False}]),
        rule("inverse_absence", [{"condition_id": "c", "text": "No diabetes", "negated": True}])]
    relations, applicability, spans = match_conditions(shared_rules, negative_patient, backend)
    assert [row["status"] for row in relations] == ["satisfied", "not_satisfied"]
    assert len(backend.calls) == before + 2
    assert relations[0]["evidence"] == relations[1]["evidence"]
    check_span_positions(negative_patient, spans)
    records.append({"case": "shared_condition_and_complement", "relations": relations,
        "applicability": applicability, "passed": True})

    patient = "The patient is 58 years old."
    adult = condition("adult", "age", ">=", 18, "years")
    child = condition("child", "age", "<", 18, "years")
    missing = condition("egfr", "egfr", ">=", 60, "mL/min/1.73 m2")
    rules = [
        rule("and_unknown", [adult, missing]),
        rule("or_satisfied", [adult, missing], "OR"),
        rule("excluded", [adult], exceptions=[condition("exception_age", "age", ">=", 50, "years")]),
        rule("not_applicable", [child]),
    ]
    with tempfile.TemporaryDirectory(prefix="module1_cpu_") as cache:
        result = Module1(top_k=4, backend=backend, cache_dir=cache).run(
            guideline={"rules": rules}, patient=patient, question="What action is appropriate?")
    states = {row["rule_id"]: row["status"] for row in result["applicability"]}
    assert states == {"and_unknown": "unknown", "or_satisfied": "applicable",
        "excluded": "not_applicable", "not_applicable": "not_applicable"}, states
    assert result["action_gate"]["or_satisfied"] > 0
    for rid in ("and_unknown", "excluded", "not_applicable"):
        assert result["action_gate"][rid] == 0, (rid, result["action_gate"])
    assert result["binding_gate"]["and_unknown"]["egfr"] == 0
    assert result["binding_gate"]["not_applicable"]["child"] > 0
    assert any(span["rule_id"] == "not_applicable" for span in result["critical_spans"])
    check_span_positions(patient, result["critical_spans"])
    json.dumps(result, allow_nan=False)
    records.append({"case": "full_interface_logic_and_gates", "applicability": result["applicability"],
        "action_gate": result["action_gate"], "binding_gate": result["binding_gate"], "passed": True})
    output = {"run_kind": "CPU logic unit tests", "models_loaded": False,
        "stub_backend_only_for_logic_tests": True, "passed": True, "cases": records}
    destination = ROOT / "results/module1_cpu_checks.json"
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(f"Passed {len(records)} necessary CPU checks; saved {destination}")

if __name__ == "__main__":
    main()

