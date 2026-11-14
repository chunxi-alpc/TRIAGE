"""Executable reference labels; never used as a learned prediction model.

This module deliberately has no model, network, clinical lookup, or training code.
Its inputs are invented facts in a finite routing simulation. Unknown observations
are not false, and another subject's or historical facts do not establish a
current fact about the evaluated subject.
"""
from __future__ import annotations

import math
from typing import Any

STATES = {"T", "F", "U"}
UNIT_SCALE = {
    "year": ("time", 12.0), "years": ("time", 12.0),
    "month": ("time", 1.0), "months": ("time", 1.0),
    "cm": ("length", 10.0), "mm": ("length", 1.0),
}


def negate(state: str) -> str:
    return {"T": "F", "F": "T", "U": "U"}[state]


def evaluate_tree(tree: Any, labels: dict[str, str]) -> str:
    """Strong Kleene logic, including empty conjunction/disjunction identities."""
    if isinstance(tree, str):
        if tree not in labels:
            raise ValueError(f"Unknown atomic condition: {tree}")
        if labels[tree] not in STATES:
            raise ValueError(f"Invalid truth value for {tree}")
        return labels[tree]
    if not isinstance(tree, dict) or set(tree) != {"op", "args"}:
        raise ValueError("Logic nodes must contain exactly op and args")
    op, args = tree["op"], tree["args"]
    if not isinstance(args, list):
        raise ValueError("Logic args must be a list")
    vals = [evaluate_tree(arg, labels) for arg in args]
    if op == "NOT":
        if len(vals) != 1:
            raise ValueError("NOT needs exactly one child")
        return negate(vals[0])
    if op == "AND":
        return "F" if "F" in vals else "U" if "U" in vals else "T"
    if op == "OR":
        return "T" if "T" in vals else "U" if "U" in vals else "F"
    raise ValueError(f"Unsupported logical operator: {op}")


def convert(value: float, source: str, target: str) -> float:
    if source not in UNIT_SCALE or target not in UNIT_SCALE:
        raise ValueError(f"Unknown unit conversion {source} -> {target}")
    source_dim, source_scale = UNIT_SCALE[source]
    target_dim, target_scale = UNIT_SCALE[target]
    if source_dim != target_dim:
        raise ValueError("Incompatible dimensions")
    return float(value) * source_scale / target_scale


def compare(value: float, operator: str, threshold: float) -> bool:
    if not math.isfinite(value) or not math.isfinite(threshold):
        raise ValueError("Non-finite numeric fact")
    return {
        ">": lambda: value > threshold,
        ">=": lambda: value >= threshold,
        "<": lambda: value < threshold,
        "<=": lambda: value <= threshold,
        "==": lambda: math.isclose(value, threshold, rel_tol=0, abs_tol=1e-9),
    }[operator]()


def evaluate_atom(atom: dict, facts: list[dict]) -> dict:
    observed = [f for f in facts if f["field"] == atom["field"]]
    eligible = [f for f in observed
                if f.get("subject") == "patient" and f.get("time") == "current"]
    result = {
        "condition_id": atom["condition_id"], "state": "U",
        "support_evidence_ids": [], "refute_evidence_ids": [],
        "observed_evidence_ids": [f["evidence_id"] for f in observed],
        "reason": "no_current_patient_observation",
    }
    if not eligible:
        return result
    values, truth = [], []
    try:
        for fact in eligible:
            if "numeric" in atom:
                spec = atom["numeric"]
                val = convert(fact["value"], fact["unit"], spec["unit"])
                values.append(round(val, 9))
                truth.append(compare(val, spec["operator"], spec["value"]))
            else:
                if type(fact["value"]) is not bool:
                    raise ValueError("Boolean fact expected")
                values.append(fact["value"])
                truth.append(fact["value"] == atom.get("expected", True))
    except (KeyError, TypeError, ValueError):
        result["reason"] = "uninterpretable_or_incompatible_observation"
        return result
    result["support_evidence_ids"] = [f["evidence_id"] for f, t in zip(eligible, truth) if t]
    result["refute_evidence_ids"] = [f["evidence_id"] for f, t in zip(eligible, truth) if not t]
    # Different current values are unresolved contradictory records, even if
    # both happen to lie on the same side of this particular threshold.
    if len(set(values)) > 1:
        result["reason"] = "conflicting_current_patient_observations"
        return result
    result["state"] = "T" if truth[0] else "F"
    result["reason"] = "current_patient_evidence"
    return result


def sufficient_proof(tree: Any, labels: dict[str, str], wanted: str) -> list[str] | None:
    """Small deterministic witness set; an OR proof need not cite every branch."""
    if wanted not in {"T", "F"} or evaluate_tree(tree, labels) != wanted:
        return None
    if isinstance(tree, str):
        return [tree]
    op, args = tree["op"], tree["args"]
    if op == "NOT":
        return sufficient_proof(args[0], labels, negate(wanted))
    need_all = (op == "AND" and wanted == "T") or (op == "OR" and wanted == "F")
    candidates = [sufficient_proof(arg, labels, wanted) for arg in args]
    if need_all:
        if any(p is None for p in candidates):
            return None
        return sorted({leaf for proof in candidates for leaf in proof})
    valid = [p for p in candidates if p is not None]
    return min(valid, key=lambda p: (len(p), tuple(p))) if valid else None


def reference_structure(rules: list[dict], facts: list[dict]) -> dict:
    all_atoms = {a["condition_id"]: a
                 for r in rules for a in r["conditions"] + r.get("exceptions", [])}
    atoms = {cid: evaluate_atom(atom, facts) for cid, atom in all_atoms.items()}
    labels = {cid: item["state"] for cid, item in atoms.items()}
    outputs = []
    for rule in rules:
        full_tree = {"op": "AND", "args": [
            rule["logic"], {"op": "NOT", "args": [rule.get("exception_logic", {"op": "OR", "args": []})]}
        ]}
        state = evaluate_tree(full_tree, labels)
        proof = sufficient_proof(full_tree, labels, state)
        evidence = sorted({eid for cid in (proof or []) for eid in
                           atoms[cid]["support_evidence_ids" if labels[cid] == "T" else "refute_evidence_ids"]})
        outputs.append({
            "rule_id": rule["rule_id"], "action_id": rule["action_id"],
            "state": state, "applicability": state,
            "confidence": 1.0 if state in {"T", "F"} else 0.0,
            "allow_attention_intervention": state in {"T", "F"} and proof is not None,
            "proof_condition_ids": proof or [], "proof_evidence_ids": evidence,
            "support_evidence_ids": evidence if state == "T" else [],
            "refute_evidence_ids": evidence if state == "F" else [],
        })
    return {"status": "ok", "source": "reference", "diagnostic_only": True,
            "rules": outputs, "atoms": atoms}


def reference_answer(structure: dict) -> str:
    applicable = [r["action_id"] for r in structure["rules"] if r["state"] == "T"]
    if len(applicable) > 1:
        raise ValueError("Generated policy has multiple simultaneously applicable actions")
    return applicable[0] if applicable else "review"
