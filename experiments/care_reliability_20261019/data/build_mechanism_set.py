#!/usr/bin/env python3
"""Freeze independent invented routing families before any model is evaluated.

No prediction log, existing test result, or external clinical recommendation is
read. The reference program derives labels from explicit synthetic observations.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))
from joint.context import full_context
from joint.logic import evaluate_tree, reference_structure, reference_answer, convert

DATASET_ID = "sealed_mechanism_20261019"
BASE_COMMIT = "7557d3150214a3894f2d340df6d9964de75efce1"
ACTION_TEXT = {"cedar": "Route Cedar", "maple": "Route Maple", "juniper": "Route Juniper", "review": "Request review"}


def AND(*args): return {"op": "AND", "args": list(args)}
def OR(*args): return {"op": "OR", "args": list(args)}
def NOT(arg): return {"op": "NOT", "args": [arg]}


FAMILIES = [
    ("conjunction", AND("C1", "C2")),
    ("disjunction", OR("C1", "C2")),
    ("negation", NOT("C1")),
    ("conjunction_nested_disjunction", AND("C1", OR("C2", "C3"))),
    ("disjunction_nested_conjunction", OR(AND("C1", "C2"), "C3")),
    ("negated_entry_conjunction", AND(NOT("C1"), "C2")),
    ("negated_entry_disjunction", OR(NOT("C1"), "C2")),
    ("negated_conjunction", NOT(AND("C1", "C2"))),
    ("negated_disjunction", NOT(OR("C1", "C2"))),
    ("entry_with_two_exclusions", AND("C1", NOT(OR("C2", "C3")))),
    ("alternative_conjunctions", OR(AND("C1", "C2"), AND("C3", "C4"))),
    ("joint_alternatives", AND(OR("C1", "C2"), OR("C3", "C4"))),
    ("exclusive_entry", OR(AND("C1", NOT("C2")), AND(NOT("C1"), "C2"))),
    ("conditional_requirement", OR(NOT("C1"), "C2")),
    ("three_requirements", AND("C1", "C2", "C3")),
    ("three_alternatives", OR("C1", "C2", "C3")),
    ("three_exclusions", NOT(OR("C1", "C2", "C3"))),
    ("negated_three_requirements", NOT(AND("C1", "C2", "C3"))),
    ("nested_negated_alternatives", OR(AND("C1", NOT("C2")), NOT(OR("C3", "C4")))),
    ("two_paths_with_exclusions", OR(AND("C1", "C2", NOT("C3")), AND("C3", NOT("C4")))),
    ("shared_requirement", AND("C1", OR(AND("C2", "C3"), "C4"))),
    ("five_condition_composition", AND(OR("C1", "C2"), NOT(AND("C3", "C4")), "C5")),
    ("exceptional_alternative", OR(AND("C1", NOT(OR("C2", "C3"))), AND("C4", "C5"))),
    ("negated_nested_composition", NOT(OR(AND("C1", "C2"), AND(NOT("C3"), "C4", "C5")))),
]


def leaves(tree):
    if isinstance(tree, str):
        return {tree}
    return {cid for arg in tree["args"] for cid in leaves(arg)}


def stable_int(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)


def atom_definitions(index):
    age = 18 + index % 9
    size = 1.0 + (index % 8) * .5
    interval = 12 * (1 + index % 3)
    return {
        "C1": {"condition_id": "C1", "field": "age",
               "text": f"The patient's current age is at least {age} years.",
               "numeric": {"field": "age", "operator": ">=", "value": age, "unit": "year"}},
        "C2": {"condition_id": "C2", "field": "authorization", "expected": True,
               "text": "The current patient has routing authorization."},
        "C3": {"condition_id": "C3", "field": "tumor_cm",
               "text": f"The patient's current tumor diameter is at most {size:g} cm.",
               "numeric": {"field": "tumor_cm", "operator": "<=", "value": size, "unit": "cm"}},
        "C4": {"condition_id": "C4", "field": "audit_certificate", "expected": True,
               "text": "The current patient has an audit certificate."},
        "C5": {"condition_id": "C5", "field": "relapse_months",
               "text": f"The patient's current relapse interval is greater than {interval} months.",
               "numeric": {"field": "relapse_months", "operator": ">", "value": interval, "unit": "month"}},
        "E1": {"condition_id": "E1", "field": "suspension", "expected": True,
               "text": "The current patient has a routing suspension flag."},
        "E2": {"condition_id": "E2", "field": "waiver", "expected": True,
               "text": "The current patient has a suspension waiver."},
    }


def make_rules(tree, definitions):
    conditions = [copy.deepcopy(definitions[c]) for c in sorted(leaves(tree))]
    exceptions = [copy.deepcopy(definitions[c]) for c in ("E1", "E2")]
    exception_tree = AND("E1", NOT("E2"))
    return [
        {"rule_id": "R_TARGET", "action_id": "cedar", "action": ACTION_TEXT["cedar"],
         "conditions": conditions, "logic": tree, "exceptions": exceptions, "exception_logic": exception_tree},
        {"rule_id": "R_ALTERNATIVE", "action_id": "maple", "action": ACTION_TEXT["maple"],
         "conditions": copy.deepcopy(conditions), "logic": NOT(copy.deepcopy(tree)),
         "exceptions": copy.deepcopy(exceptions), "exception_logic": copy.deepcopy(exception_tree)},
        {"rule_id": "R_EXCEPTION", "action_id": "juniper", "action": ACTION_TEXT["juniper"],
         "conditions": copy.deepcopy(exceptions), "logic": copy.deepcopy(exception_tree),
         "exceptions": [], "exception_logic": OR()},
    ]


def fact_for(atom, truth, *, suffix="", subject="patient", time="current"):
    field = atom["field"]
    if "numeric" in atom:
        spec = atom["numeric"]
        step = {"age": 1, "tumor_cm": .1, "relapse_months": 12}[field]
        boundary = spec["value"]
        if spec["operator"] == ">=": value = boundary if truth else boundary - step
        elif spec["operator"] == "<=": value = boundary if truth else boundary + step
        elif spec["operator"] == ">": value = boundary + step if truth else boundary
        else: raise ValueError(spec)
        fact = {"evidence_id": f"obs_{field}{suffix}", "field": field,
                "value": round(value, 9), "unit": spec["unit"], "subject": subject, "time": time}
    else:
        fact = {"evidence_id": f"obs_{field}{suffix}", "field": field,
                "value": bool(truth), "subject": subject, "time": time}
    return fact


def fact_text(fact):
    current = fact["time"] == "current"
    who = "patient" if fact["subject"] == "patient" else "patient's sibling"
    prefix = f"The {who}'s {'current' if current else 'historical'}"
    if fact["field"] in {"age", "tumor_cm", "relapse_months"}:
        name = {"age": "age", "tumor_cm": "tumor diameter", "relapse_months": "relapse interval"}[fact["field"]]
        unit = fact["unit"]
        unit = {"year": "years", "month": "months"}.get(unit, unit)
        return f"{prefix} {name} {'is' if current else 'was'} {fact['value']:g} {unit}."
    name = {"authorization": "routing authorization", "audit_certificate": "an audit certificate",
            "suspension": "a routing suspension flag", "waiver": "a suspension waiver"}[fact["field"]]
    return f"The {'current' if current else 'historical'} {who} {'has' if fact['value'] else 'does not have'} {name}."


def assignment_facts(assignment, definitions):
    return [fact_for(definitions[cid], state == "T") for cid, state in sorted(assignment.items()) if state != "U"]


def option_order(pair_key):
    permutation = list(itertools.permutations(ACTION_TEXT))[stable_int(pair_key) % 24]
    return {letter: ACTION_TEXT[action] for letter, action in zip("ABCD", permutation)}


def build_case(family_id, split, rules, definitions, assignment, pair_key, side, variant,
               *, changed_atom=None, uncertain_mode=None, paired=True):
    facts = assignment_facts(assignment, definitions)
    if uncertain_mode:
        cid = changed_atom
        facts = [f for f in facts if f["field"] != definitions[cid]["field"]]
        if uncertain_mode == "historical_only":
            facts.append(fact_for(definitions[cid], True, time="historical"))
        elif uncertain_mode == "relative_only":
            facts.append(fact_for(definitions[cid], True, subject="relative"))
        elif uncertain_mode == "contradictory_current":
            facts += [fact_for(definitions[cid], value, suffix=f"_{i}") for i, value in enumerate((True, False))]
        elif uncertain_mode != "missing":
            raise ValueError(uncertain_mode)
    if variant == "unit_equivalent":
        for fact in facts:
            if "unit" in fact:
                other = {"year": "month", "month": "year", "cm": "mm"}[fact["unit"]]
                fact["value"] = round(convert(fact["value"], fact["unit"], other), 9)
                fact["unit"] = other
    elif variant == "scope_and_order":
        # Fixed distractors across both pair members: the minimal current
        # patient-state change is preserved.
        facts += [fact_for(definitions["C1"], True, suffix="_relative", subject="relative"),
                  fact_for(definitions["C1"], False, suffix="_old", time="historical")]
        facts.reverse()
    ref = reference_structure(rules, facts)
    action_id = reference_answer(ref)
    target_state = next(r["state"] for r in ref["rules"] if r["rule_id"] == "R_TARGET")
    options = option_order(pair_key)
    gold_answer = next(letter for letter, text in options.items() if text == ACTION_TEXT[action_id])
    target_answer = next(letter for letter, text in options.items() if text == ACTION_TEXT["cedar"])
    notes = ""
    if uncertain_mode == "missing":
        notes = " The current value of " + definitions[changed_atom]["field"] + " is not documented."
    row = {
        "dataset_id": DATASET_ID, "case_id": f"{pair_key}-{variant}-{side}",
        "pair_id": f"{pair_key}-{variant}" if paired else None,
        "variant_group_id": pair_key, "family_id": family_id, "group_id": family_id,
        "split": split, "outer_fold": stable_int(family_id) % 3,
        "outer_fold_purpose": "fixed legacy-asset assignment; never selection on new labels",
        "variant": variant, "uncertainty_mode": uncertain_mode,
        "question": "Which routing action follows from the patient's current observations and all of the rules?",
        "patient": " ".join(fact_text(f) for f in facts) + notes,
        "rules": copy.deepcopy(rules), "structured_facts": facts,
        "reference_structure": ref, "reference_state": target_state,
        "applicable": {"T": True, "F": False, "U": None}[target_state],
        "pair_eligible": paired and target_state in {"T", "F"},
        "target_rule_id": "R_TARGET", "target_action_id": "cedar",
        "target_action": ACTION_TEXT["cedar"], "target_answer": target_answer,
        "inapplicable_action": target_answer, "option_map": options,
        "gold_action_id": action_id, "gold_label": action_id, "gold_answer": gold_answer,
        "rule_id": family_id, "changed_atom": changed_atom,
        "is_clinical": False, "independent_expert_review": False,
    }
    row["prompt"], row["action_spans"] = full_context(row)
    row["spans"] = {"action": row["action_spans"]["R_TARGET"]}
    row["guidance"] = row["prompt"].split("Guidance:\n", 1)[1].split("\nOptions:", 1)[0]
    row["query"] = row["question"]
    # Legacy diagnostic loaders sometimes expect a scalar. It describes the
    # manipulated atom, not an independent clinical numeric variable.
    row["value"] = 1.0 if assignment.get(changed_atom) == "T" else 0.0
    return row


def generate():
    rows, split_map, family_info = [], {}, []
    pattern = ["train", "confirmation", "train", "dev", "confirmation", "train"]
    for index, (name, tree) in enumerate(FAMILIES):
        family_id = f"mechanism_{index + 1:02d}_{name}"
        split = pattern[index % len(pattern)]
        split_map[family_id] = split
        defs = atom_definitions(index)
        active = sorted(leaves(tree))
        rules = make_rules(tree, defs)
        truth_assignments = []
        for bits in itertools.product(("F", "T"), repeat=len(active)):
            assignment = dict(zip(active, bits)) | {"E1": "F", "E2": "F"}
            truth_assignments.append(assignment)
        edges = []
        for app in truth_assignments:
            if evaluate_tree(tree, app) != "T":
                continue
            for cid in active:
                non = app | {cid: "F" if app[cid] == "T" else "T"}
                if evaluate_tree(tree, non) == "F":
                    edges.append((cid, app, non))
        assert edges, family_id
        edges.sort(key=lambda x: ("numeric" not in defs[x[0]], x[0], json.dumps(x[1], sort_keys=True)))
        chosen = [edges[0]]
        additional = next((e for e in edges[1:] if e[0] != edges[0][0]), None)
        if additional is None and len(edges) == 1:
            cid, edge_app, edge_non = edges[0]
            # A one-atom policy has only one Boolean condition edge. Use a
            # distinct explicit waiver context rather than duplicate that case.
            additional = (cid, edge_app | {"E2": "T"}, edge_non | {"E2": "T"})
        chosen.append(additional or edges[-1])
        app = copy.deepcopy(chosen[0][1])
        non = app | {"E1": "T"}
        chosen.append(("E1", app, non))
        for pair_index, (cid, app, non) in enumerate(chosen):
            pair_key = f"{family_id}-p{pair_index + 1:02d}"
            for variant in ("canonical", "unit_equivalent", "scope_and_order"):
                for side, assignment in (("app", app), ("non", non)):
                    row = build_case(family_id, split, rules, defs, assignment, pair_key, side, variant, changed_atom=cid)
                    assert row["reference_state"] == ("T" if side == "app" else "F")
                    rows.append(row)
        pivot, app, _ = edges[0]
        for mode in ("missing", "historical_only", "relative_only", "contradictory_current"):
            key = f"{family_id}-u-{mode}"
            row = build_case(family_id, split, rules, defs, app, key, "unknown", "canonical",
                             changed_atom=pivot, uncertain_mode=mode, paired=False)
            assert row["reference_state"] == "U" and row["gold_action_id"] == "review"
            rows.append(row)
        # Incomplete observations can still suffice for a T/F proof. Prefer
        # masking a non-decisive core atom; otherwise mask the irrelevant waiver
        # under an explicitly absent suspension flag.
        for target in ("T", "F"):
            found = None
            for assignment in truth_assignments:
                if evaluate_tree(tree, assignment) != target:
                    continue
                for cid in active + ["E2"]:
                    unknown = assignment | {cid: "U"}
                    ref = reference_structure(rules, assignment_facts(unknown, defs))
                    if ref["rules"][0]["state"] == target:
                        found = (cid, unknown)
                        break
                if found:
                    break
            assert found is not None
            cid, assignment = found
            key = f"{family_id}-short-circuit-{target}"
            row = build_case(family_id, split, rules, defs, assignment, key, target, "short_circuit", changed_atom=cid, paired=False)
            assert row["reference_state"] == target
            rows.append(row)
        family_info.append({"family_id": family_id, "split": split, "logic": tree,
                            "exception_logic": AND("E1", NOT("E2")),
                            "condition_ids": active, "counterfactual_changed_atoms": [e[0] for e in chosen]})
    return rows, split_map, family_info


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, split_map, family_info = generate()
    files = []
    for name, subset in [
        ("mechanism_cases.jsonl", rows),
        ("module2_sealed.jsonl", [r for r in rows if r["split"] == "confirmation" and r["pair_eligible"]]),
    ]:
        path = args.output / name
        write_jsonl(path, subset)
        files.append(path)
    splits = args.output / "family_split.json"
    splits.write_text(json.dumps({"assignment_policy": "fixed six-position pattern repeated; no model output consulted",
                                 "families": split_map, "definitions": family_info}, indent=2) + "\n")
    files.append(splits)
    counts = {split: dict(Counter(r["reference_state"] for r in rows if r["split"] == split))
              for split in ("train", "dev", "confirmation")}
    manifest = {
        "dataset_id": DATASET_ID, "schema_version": "care.synthetic_mechanism.v1",
        "status": "frozen_before_model_execution", "base_repository_commit": BASE_COMMIT,
        "generator_sha256": digest(Path(__file__)), "is_clinical": False,
        "reference_label_provenance": "executable strong-Kleene logic over deliberately invented routing facts",
        "independent_expert_review": False, "existing_model_outputs_read": False,
        "method_development_blind": False,
        "development_exposure": {
            "timing": "frozen before model forward execution; not blind throughout method development",
            "exposed_material": ["schema", "case wording"],
            "use": "CPU development regression checks and fixes, including relative-subject recognition",
            "case_label_or_split_changes_based_on_model_outputs": False,
            "interpretation": "new synthetic mechanism test with prespecified family holdout; not independently blinded confirmation or clinical validation",
        },
        "existing_patient_or_rule_samples_reused": False, "cases": len(rows),
        "families": len(split_map), "family_counts": dict(Counter(split_map.values())),
        "case_counts_by_reference_state": counts,
        "pair_counts": dict(Counter(r["split"] for r in rows if r["pair_eligible"] and r["applicable"])),
        "module2_sealed_scope": "confirmation split; complete T/F pairs only; fixed target action",
        "outer_fold_policy": "SHA256(family_id) modulo 3; fixed selection among previously trained legacy assets",
        "primary_analysis": "confirmation canonical pairs; all variants and U/proof subsets reported separately",
        "prohibited_selection": ["filtering cases by model confidence or effect", "selecting methods on confirmation labels",
                                 "calling random projections learned subspaces", "dropping parsing failures from denominators"],
        "limitations": ["invented rules and templated simulated observations", "no clinical validity or clinician adjudication",
                        "schema and wording were exposed during method-development regression fixes; not blind throughout development",
                        "syntactic family holdout does not establish open-world semantic generalization",
                        "surface variants share counterfactual groups and are not independent samples"],
        "files": {path.name: {"sha256": digest(path), "bytes": path.stat().st_size} for path in files},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("status", "cases", "families", "family_counts", "pair_counts")}, indent=2))


if __name__ == "__main__":
    main()
