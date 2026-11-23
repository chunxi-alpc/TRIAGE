"""Normalize real V6/V7 outputs without inventing labels or bypassing gates."""
from __future__ import annotations

import copy
import importlib

from .context import prediction_input


def normalize_structure(raw, case, source):
    expected = {rule["rule_id"]: rule for rule in case["rules"]}
    if not isinstance(raw, dict) or not isinstance(raw.get("rules"), list):
        return {"status": "parse_failure", "source": source, "rules": [],
                "parse_failure": True, "error": "missing structured rule output"}
    if any(not isinstance(rule, dict) for rule in raw["rules"]):
        return {"status": "parse_failure", "source": source, "rules": [],
                "parse_failure": True, "error": "non-object rule output"}
    outputs, errors = [], []
    received = [r.get("rule_id") for r in raw["rules"] if isinstance(r, dict)]
    if len(received) != len(set(received)) or set(received) != set(expected):
        return {"status": "parse_failure", "source": source, "rules": [],
                "parse_failure": True, "error": "rule IDs missing, duplicated, or unexpected"}
    for rule in raw["rules"]:
        rid = rule["rule_id"]
        state = rule.get("state", rule.get("applicability"))
        if state not in {"T", "F", "U"}:
            return {"status": "parse_failure", "source": source, "rules": [],
                    "parse_failure": True, "error": f"invalid applicability state for {rid}"}
        # Preserve the actual implementation's gate. In particular, V6's
        # positive-only rule gate must not silently become a negative edit gate.
        allowed = bool(rule.get("allow_attention_intervention", False)) and state in {"T", "F"}
        confidence = rule.get("confidence")
        score = confidence if confidence is not None else (1.0 if allowed else 0.0)
        if not isinstance(score, (int, float)) or not 0 <= score <= 1:
            allowed, score, confidence = False, 0.0, None
            errors.append({"rule_id": rid, "error": "invalid confidence"})
        for atom in rule.get("conditions", []) + rule.get("exceptions", []):
            trialmatch = atom.get("trialmatch", {})
            status = str(trialmatch.get("parse_status", atom.get("generation_status", ""))).casefold()
            # The strict parser has one accepted terminal status. Its current
            # failures include invalid_output, ambiguous_target, invalid_label,
            # truncated, ERROR and unparsed; future nonaccepted values remain
            # errors rather than silently escaping this accounting.
            if atom.get("technical_error") or trialmatch.get("parse_error") or status not in {"", "accepted"}:
                errors.append({"rule_id": rid, "condition_id": atom.get("condition_id"),
                               "error": atom.get("technical_error") or trialmatch.get("parse_error") or status})
        outputs.append({
            "rule_id": rid, "action_id": expected[rid]["action_id"],
            "state": state, "confidence": confidence, "gate_weight": float(score),
            "allow_attention_intervention": allowed,
            "support_evidence": copy.deepcopy(rule.get("support_evidence", [])),
            "refute_evidence": copy.deepcopy(rule.get("refute_evidence", [])),
            "selected_proof": copy.deepcopy(rule.get("selected_proof")),
            "proof_evidence": copy.deepcopy(rule.get("proof_evidence", [])),
        })
    return {"status": "ok", "source": source, "rules": outputs,
            "parse_failure": bool(errors), "condition_parse_failures": errors,
            "raw_schema_version": raw.get("schema_version"),
            "scores_are_calibrated": bool(raw.get("metadata", {}).get("scores_are_calibrated", False))}


class StructureProvider:
    """Models are loaded lazily and only raw case text/rules are passed to them."""
    def __init__(self, source, *, include_reference_diagnostic=False, model_options=None):
        if source not in {"reference", "reference_positive", "v6", "v7"}:
            raise ValueError(source)
        if source.startswith("reference") and not include_reference_diagnostic:
            raise ValueError("Reference structure requires explicit diagnostic mode")
        self.source = source
        self.options = model_options or {}
        self.instance = None

    def load(self, shared_adapters=None):
        if self.source.startswith("reference") or self.instance is not None:
            return
        module = importlib.import_module(f"src.module1.care_{self.source}")
        cls = getattr(module, f"CAREModule1{self.source.upper()}")
        self.instance = cls(**shared_adapters) if shared_adapters is not None else cls.from_pretrained(**self.options)

    def predict(self, case):
        if self.source.startswith("reference"):
            out = copy.deepcopy(case["reference_structure"])
            out["source"] = self.source
            out["gate_policy"] = "T_and_F" if self.source == "reference" else "T_only"
            if self.source == "reference_positive":
                for rule in out["rules"]:
                    rule["allow_attention_intervention"] = bool(rule["allow_attention_intervention"] and rule["state"] == "T")
            out["parse_failure"] = False
            return out, None
        self.load()
        raw = self.instance.run(**prediction_input(case))
        return normalize_structure(raw, case, self.source), raw


def signed_targets(structure, action_spans):
    if structure.get("status") != "ok":
        return []
    targets = []
    for rule in structure["rules"]:
        if not rule.get("allow_attention_intervention") or rule["state"] == "U":
            continue
        magnitude = float(rule.get("gate_weight", rule.get("confidence") or 0.0))
        if magnitude <= 0:
            continue
        targets.append({"rule_id": rule["rule_id"], "span": action_spans[rule["rule_id"]],
                        "sign": 1 if rule["state"] == "T" else -1, "magnitude": magnitude})
    return targets
