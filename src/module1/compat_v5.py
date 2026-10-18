"""V5 adapter for the unchanged module-two/three ``module1.v1`` contract.

``confidence`` is evidence strength, not calibrated correctness: a conflict-free
numeric comparison is 1; otherwise it is the largest actual MedNLI probability
for the final relation. Unknown/conflicting or unsupported decisions receive 0.
The legacy action-prior conflict path stays disabled because V5 never asks Qwen
to generate or compare module-one decisions.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
from typing import Any

from .care_v5 import (
    BIOLINK_MEDNLI, TRIALMATCH_ADAPTER, TRIALMATCH_BASE, CAREModule1V5,
    FALSE, TRUE, UNKNOWN, eval_logic,
)

ROOT = Path(__file__).resolve().parents[2]
_STATUS = {TRUE: "satisfied", FALSE: "not_satisfied", UNKNOWN: "unknown"}
_STATE = {value: key for key, value in _STATUS.items()}


def _logic_proof(expr: Any, states: dict[str, str]) -> tuple[str, list[str]]:
    """Find condition IDs that establish, refute, or leave an expression unknown."""
    state = eval_logic(expr, states)
    if isinstance(expr, str):
        return state, [expr]
    if not isinstance(expr, dict):
        return state, []
    children = [_logic_proof(arg, states) for arg in expr.get("args", [])]
    op = str(expr.get("op", "AND")).upper()
    if op == "AND" and state == FALSE:
        children = [child for child in children if child[0] == FALSE]
    elif op == "OR" and state == TRUE:
        children = [child for child in children if child[0] == TRUE]
    elif state == UNKNOWN:
        children = [child for child in children if child[0] == UNKNOWN]
    return state, list(dict.fromkeys(cid for _, ids in children for cid in ids))


def _evidence_confidence(condition: dict[str, Any]) -> tuple[float, str]:
    state = condition.get("status", UNKNOWN)
    if state not in (TRUE, FALSE) or condition.get("conflict"):
        return 0.0, "unknown_or_conflicting"
    numeric = condition.get("numeric") or {}
    if (numeric.get("status") == state and numeric.get("evidence") and
            numeric.get("reason") == "deterministic_numeric_comparison"):
        return 1.0, "deterministic_numeric_flag"
    relation = "entailment" if state == TRUE else "contradiction"
    key = "support_evidence" if state == TRUE else "refute_evidence"
    scores = [row.get("scores", {}).get(relation) for row in condition.get(key, [])]
    scores = [float(score) for score in scores if isinstance(score, (int, float)) and
              math.isfinite(score) and 0.0 <= score <= 1.0]
    return (max(scores), "uncalibrated_mednli_probability") if scores else (0.0, "no_matching_evidence")


def as_module1_v1(result: dict[str, Any]) -> dict[str, Any]:
    """Add the established downstream fields without changing V5 predictions."""
    result["schema_version"] = "module1.v1"
    result["implementation_version"] = "module1.v5"
    rule_by_id = {rule["rule_id"]: rule for rule in result["candidate_rules"]}
    details = {(rule["rule_id"], condition["condition_id"]): condition
               for rule in result["rules"]
               for condition in rule.get("conditions", []) + rule.get("exceptions", [])}
    relation_by_id = {}
    for row in result["condition_evidence_relations"]:
        key = row["rule_id"], row["condition_id"]
        detail = details[key]
        confidence, definition = _evidence_confidence(detail)
        row["criterion_kind"] = row.get("role", "inclusion")
        row["role"] = "exception" if row["criterion_kind"] == "exclusion" else "condition"
        row["confidence"], row["confidence_kind"] = confidence, definition
        row["method"] = "care_v5:" + row.get("resolution", "unknown")
        # Unknown candidate positions remain in unknown_evidence, rather than
        # becoming evidence for a known predicate merely because they are nearby.
        if row["status"] in {"satisfied", "not_satisfied"}:
            field = "support_evidence" if row["status"] == "satisfied" else "refute_evidence"
            row["evidence"] = list(row.get(field, []))
        relation_by_id[key] = row
    for app in result["applicability"]:
        rid = app["rule_id"]
        rule = rule_by_id[rid]
        states = {cid: _STATE[row["status"]] for (rule_id, cid), row in relation_by_id.items()
                  if rule_id == rid}
        base, base_ids = _logic_proof(rule["logic"], states)
        excluded, exception_ids = _logic_proof(rule["exception_logic"], states)
        state = app["state"]
        if state == FALSE:
            decisive = (base_ids if base == FALSE else []) + (exception_ids if excluded == TRUE else [])
        elif state == UNKNOWN:
            decisive = (base_ids if base == UNKNOWN else []) + (exception_ids if excluded == UNKNOWN else [])
        else:
            decisive = base_ids + exception_ids
        decisive = list(dict.fromkeys(decisive))
        app.update(conditions_status=_STATUS[base], exceptions_status=_STATUS[excluded],
                   decisive_condition_ids=decisive,
                   blockers=decisive if state == FALSE else [],
                   supporters=decisive if state == TRUE else [])
        for (rule_id, cid), row in relation_by_id.items():
            if rule_id == rid:
                row["decisive"] = cid in decisive
    # No closed-book/guideline Qwen decisions are generated in V5. Model-evidence
    # disagreement is kept separately, and must not masquerade as that old signal.
    result["model_evidence_conflicts"] = result.get("conflict_scores", [])
    result["conflict_scores"] = [{"rule_id": rule["rule_id"], "score": 0.0,
        "status": "unknown", "prior_support": "unknown", "guided_support": "unknown",
        "reason": "V5 does not generate or compare closed-book and guideline Qwen decisions"}
        for rule in result["candidate_rules"]]
    for row in result["relation_weights"]:
        relation = relation_by_id[row["rule_id"], row["condition_id"]]
        gate = relation["confidence"] if relation.get("evidence") else 0.0
        sign = 1 if relation["status"] == "satisfied" else -1 if relation["status"] == "not_satisfied" else 0
        row.update(role=relation["role"], evidence=relation["evidence"], confidence=relation["confidence"],
                   confidence_kind=relation["confidence_kind"], decisive=relation["decisive"],
                   weight=gate, binding_gate=gate, signed_weight=sign * gate,
                   prior_competition_weight=0.0)
        result["binding_gate"][row["rule_id"]][row["condition_id"]] = gate
    result["decisive_relations"] = [row for row in result["relation_weights"] if row["decisive"]]
    result["critical_spans"] = [{"rule_id": row["rule_id"], "condition_id": row["condition_id"], **span}
        for row in result["condition_evidence_relations"] for span in row.get("evidence", [])]
    result.setdefault("metadata", {}).update({
        "compatibility_schema": "module1.v1", "implementation": "CAREModule1V5",
        "confidence_definition": "conflict-free numeric evidence=1; otherwise maximum actual MedNLI probability for the final relation; unknown/conflict/unsupported=0; uncalibrated",
        "decisive_ids_definition": "IDs establishing/refuting the full prerequisite AND NOT exception expression, or leaving it unknown; supporters establish applicability, blockers refute applicability",
        "gate_formula": "binding=evidence confidence for known relations; action=1 iff applicable; prior_competition=0 (no prior-decision comparison)",
        "prior_conflict_path": "disabled: no Qwen calls or fabricated action-prior disagreement",
        "missing_action_rule_ids": [rule["rule_id"] for rule in result["candidate_rules"] if not rule.get("action")],
    })
    return result


class Module1:
    """Default, lazy V5 implementation with the previous constructor/run shape.

    Settings use CARE_MODULE1_V5_CONFIG or experiments/care_module1_v5/config.json.
    Explicit model-path environment variables take precedence over JSON settings.
    ``backend`` is accepted for the shared final-generation pipeline; only its
    device attribute is read. Its methods and model are never used by module one.
    Text guidelines require an exact match in the offline structured rule registry.
    """
    def __init__(self, config_path=None, device="cuda:0", top_k=12, cache_dir=None,
                 backend=None, *, engine=None, rules_path=None):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        path = Path(config_path or os.environ.get("CARE_MODULE1_V5_CONFIG") or
                    ROOT / "experiments/care_module1_v5/config.json")
        self.config_path = path
        self.config = json.loads(path.read_text()) if path.is_file() else {}
        if (config_path is not None or os.environ.get("CARE_MODULE1_V5_CONFIG")) and not path.is_file():
            raise FileNotFoundError(path)
        self.device = os.environ.get("CARE_MODULE1_V5_DEVICE") or getattr(backend, "device", None) or device
        self.cache_dir = cache_dir or os.environ.get("CARE_MODULE1_V5_CACHE")
        self.top_k = top_k
        self.rules_path = Path(rules_path or os.environ.get("CARE_MODULE1_V5_RULES") or
            self.config.get("rules_path") or ROOT / "experiments/care_module1_v5/data/rules_v5.jsonl")
        self.engine = engine
        self._registry = None

    def _load(self):
        if self.engine is None:
            models = self.config.get("models", {})
            model_paths = self.config.get("model_paths", {})
            def model_path(key, variable, directory, fallback):
                configured = os.environ.get(variable) or model_paths.get(key)
                if configured:
                    return configured
                local = Path("/home/gcx/models/care_module1_v5") / directory
                return str(local) if local.is_dir() else models.get(key, fallback)
            self.engine = CAREModule1V5.from_pretrained(device=self.device, cache_dir=self.cache_dir,
                base_model=model_path("trialmatch_base", "CARE_MODULE1_V5_BASE_MODEL", "phi-4", TRIALMATCH_BASE),
                adapter_model=model_path("trialmatch_adapter", "CARE_MODULE1_V5_ADAPTER_MODEL", "trialmatchai-phi4-reasoning-lora", TRIALMATCH_ADAPTER),
                evidence_model=model_path("evidence_model", "CARE_MODULE1_V5_EVIDENCE_MODEL", "biolinkbert-mednli", BIOLINK_MEDNLI),
                nli_batch_size=int(self.config.get("nli_batch_size", 32)),
                candidate_limit=int(self.config.get("candidate_sentence_limit", 8)))
        return self.engine

    def _exact_rule(self, text):
        if self._registry is None:
            if not self.rules_path.is_file():
                raise ValueError("Free-text guidelines require an offline structured rule registry; "
                                 "set CARE_MODULE1_V5_RULES or supply structured rules directly")
            self._registry = [json.loads(line) for line in self.rules_path.read_text().splitlines() if line.strip()]
        matches = [row for row in self._registry if text in (row.get("text"), row.get("source_text"))]
        if len(matches) != 1:
            raise ValueError("Guideline text has no unique exact match in the offline rule registry; "
                             "supply a prepared structured rule; Qwen parsing is disabled")
        return copy.deepcopy(matches[0])

    def _prepared(self, guideline):
        if isinstance(guideline, str):
            return [self._exact_rule(guideline)]
        if isinstance(guideline, (list, tuple)):
            return [self._exact_rule(row) if isinstance(row, str) else copy.deepcopy(row) for row in guideline]
        if isinstance(guideline, dict) and "rules" in guideline:
            prepared = copy.deepcopy(guideline)
            prepared["rules"] = self._prepared(guideline["rules"])
            return prepared
        if isinstance(guideline, dict) and "conditions" not in guideline and isinstance(guideline.get("text"), str):
            return [self._exact_rule(guideline["text"])]
        return copy.deepcopy(guideline)

    def run(self, guideline, patient, question):
        prepared = self._prepared(guideline)
        output = self._load().run(guideline=prepared, patient=patient, question=question)
        output.setdefault("metadata", {}).update({"module1_backend": "TrialMatchAI + BioLinkBERT-MedNLI",
            "qwen_backend_used": False, "rule_selection": "caller supplied structured rules or exact offline registry match; top_k retained for constructor compatibility; no truncation"})
        return output
