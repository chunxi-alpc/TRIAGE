"""Experimental CARE V7: complete logical evidence proofs and scoped numerics.

V6 and the package default are deliberately unchanged. This version does not
claim that NLI evidence is clinically correct: the proof gate checks structural
completeness, evidence direction, offsets and absence of conflicting evidence.
Numeric extraction is deliberately conservative and abstains outside its
documented English-language, explicit-patient, explicit-current scope.
"""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal, InvalidOperation
import itertools
import math
import re
import time
from typing import Any

from . import care_v6 as v6

TRUE, FALSE, UNKNOWN = v6.TRUE, v6.FALSE, v6.UNKNOWN
VERSION = "module1.v7.experimental"


class ProofLimitExceeded(ValueError):
    """The exact proof set exceeded a declared resource limit; never truncate."""


def _minimal(proofs: list[frozenset], limit: int) -> list[frozenset]:
    unique = sorted(set(proofs), key=lambda p: (len(p), sorted(p)))
    result = []
    for proof in unique:
        if any(existing <= proof for existing in result):
            continue
        result.append(proof)
        if len(result) > limit:
            raise ProofLimitExceeded("minimal proof count exceeds max_proofs")
    return result


def _proofs(expr: Any, states: dict[str, str], limit: int) -> list[frozenset]:
    """All subset-minimal sufficient proofs for this assignment's known value."""
    state = v6.eval_logic(expr, states)
    if state == UNKNOWN:
        return []
    if isinstance(expr, str):
        return [frozenset(((expr, state),))]
    op, args = expr["op"].upper(), expr["args"]
    if op == "NOT":
        return _proofs(args[0], states, limit)
    # AND true and OR false need every child. The other two cases need a
    # single decisive child. NOT retains the atom's actual sign in the proof.
    need_all = (op == "AND" and state == TRUE) or (op == "OR" and state == FALSE)
    if not need_all:
        choices = []
        for child in args:
            if v6.eval_logic(child, states) == state:
                choices.extend(_proofs(child, states, limit))
        return _minimal(choices, limit)
    combined = [frozenset()]
    for child in args:
        child_proofs = _proofs(child, states, limit)
        if not child_proofs:
            return []
        if len(combined) * len(child_proofs) > limit * limit:
            raise ProofLimitExceeded("intermediate proof product exceeds max_proofs squared")
        combined = _minimal([a | b for a, b in itertools.product(combined, child_proofs)], limit)
    return combined


def validate_logic(expr: Any, condition_ids: set[str]) -> None:
    if isinstance(expr, str):
        if expr not in condition_ids:
            raise ValueError(f"logic refers to unknown condition {expr!r}")
        return
    if not isinstance(expr, dict) or set(expr) - {"op", "args"}:
        raise ValueError("logic must be a condition ID or an op/args object")
    op, args = str(expr.get("op", "")).upper(), expr.get("args")
    if op not in {"AND", "OR", "NOT"} or not isinstance(args, list):
        raise ValueError("logic requires AND/OR/NOT and an args list")
    if op == "NOT" and len(args) != 1:
        raise ValueError("NOT requires exactly one argument")
    if op in {"AND", "OR"} and not args:
        raise ValueError("empty logical connectives require an explicitly unconditional rule")
    for child in args:
        validate_logic(child, condition_ids)


def full_rule_logic(rule: dict[str, Any]) -> Any:
    main = rule.get("logic", {"op": "AND", "args": [c["condition_id"] for c in rule["conditions"]]})
    if rule.get("exceptions"):
        exceptions = rule.get("exception_logic", {"op": "OR", "args": [c["condition_id"] for c in rule["exceptions"]]})
        if not rule.get("conditions") and rule.get("explicitly_unconditional"):
            return {"op": "NOT", "args": [exceptions]}
        return {"op": "AND", "args": [main, {"op": "NOT", "args": [exceptions]}]}
    return main


def sufficient_proofs(expr: Any, states: dict[str, str], max_proofs: int = 4096) -> dict[str, Any]:
    """Return exact subset-minimal proofs, or an explicit fail-closed result."""
    if max_proofs < 1:
        raise ValueError("max_proofs must be positive")
    if any(value not in {TRUE, FALSE, UNKNOWN} for value in states.values()):
        raise ValueError("atom states must be T, F or U")
    validate_logic(expr, set(states))
    try:
        proofs = _proofs(expr, states, max_proofs)
    except ProofLimitExceeded as exc:
        return {"state": v6.eval_logic(expr, states), "proofs": [], "complete": False, "reason": str(exc)}
    return {"state": v6.eval_logic(expr, states), "proofs": [
        [{"condition_id": cid, "required_state": state} for cid, state in sorted(proof)]
        for proof in proofs], "complete": True,
        "reason": "unknown_has_no_sufficient_known_proof" if not proofs else "exact_minimal_proofs"}


def _valid_span(span: Any, patient_text: str | None = None) -> bool:
    if not isinstance(span, dict):
        return False
    start, end, sid = span.get("start"), span.get("end"), span.get("sentence_id")
    if not all(isinstance(x, int) and not isinstance(x, bool) for x in (start, end, sid)):
        return False
    if start < 0 or end <= start or sid < 1:
        return False
    if patient_text is not None:
        if end > len(patient_text):
            return False
        if span.get("text") is not None and patient_text[start:end] != span["text"]:
            return False
    return True


def _dedup(spans: list[dict]) -> list[dict]:
    result = {}
    for span in spans:
        result.setdefault((span.get("sentence_id"), span.get("start"), span.get("end")), span)
    return sorted(result.values(), key=lambda x: (x.get("start", -1), x.get("end", -1)))


def condition_evidence(row: dict[str, Any], state: str | None = None,
                       patient_text: str | None = None) -> dict[str, Any]:
    """Check evidence for one signed atom, including normalized cached rows."""
    state = state or row.get("applicability", row.get("status", UNKNOWN))
    support = list(row.get("support_evidence", row.get("support_spans", [])))
    refute = list(row.get("refute_evidence", row.get("refute_spans", [])))
    numeric = row.get("numeric") or {}
    numeric_state = numeric.get("status", row.get("numeric_status", UNKNOWN))
    numeric_spans = numeric.get("evidence", row.get("numeric_evidence_spans", []))
    if numeric_state in (TRUE, FALSE):
        (support if numeric_state == TRUE else refute).extend(numeric_spans)
    invalid_n = sum(not _valid_span(x, patient_text) for x in support + refute)
    support = _dedup([x for x in support if _valid_span(x, patient_text)])
    refute = _dedup([x for x in refute if _valid_span(x, patient_text)])
    conflict = bool(support and refute) or numeric.get("reason") == "numeric_range_or_values_cross_threshold" or bool(numeric.get("context_disputed"))
    matched = support if state == TRUE else refute if state == FALSE else []
    opposite = refute if state == TRUE else support if state == FALSE else []
    if state == UNKNOWN:
        reason = "unknown_atom"
    elif conflict or opposite:
        reason = "conflicting_or_opposite_evidence"
    elif not matched:
        reason = "missing_matching_evidence"
    elif invalid_n:
        reason = "invalid_evidence_offsets"
    else:
        reason = "consistent_complete_atom_evidence"
    allowed = reason == "consistent_complete_atom_evidence"
    return {"allowed": allowed, "reason": reason, "state": state,
            "evidence_status": "disputed" if conflict or opposite else "supported" if matched else "missing",
            "support_evidence": support, "refute_evidence": refute,
            "matched_evidence": matched if allowed else [], "invalid_evidence_n": invalid_n,
            "offset_verification": "exact_text" if patient_text is not None else "structure_only"}


def rule_evidence_gate(rule: dict[str, Any], rows: list[dict[str, Any]],
                       states: dict[str, str] | None = None,
                       patient_text: str | None = None, max_proofs: int = 4096) -> dict[str, Any]:
    """Open the action gate only if a complete sufficient true proof is grounded."""
    by_id = {r["condition_id"]: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError("duplicate condition IDs in evidence rows")
    states = states or {cid: row.get("applicability", row.get("status", UNKNOWN)) for cid, row in by_id.items()}
    if not rule.get("conditions") and rule.get("explicitly_unconditional") and not rule.get("exceptions"):
        return {"state": TRUE, "allow_attention_intervention": False, "selected_proof": [],
                "proof_evidence": [], "proofs": [], "proofs_complete": True,
                "reason": "unconditional_rule_has_no_patient_evidence_target", "condition_checks": {}}
    proof_result = sufficient_proofs(full_rule_logic(rule), states, max_proofs)
    checks = {cid: condition_evidence(row, states.get(cid, UNKNOWN), patient_text) for cid, row in by_id.items()}
    grounded = [proof for proof in proof_result["proofs"] if proof and all(
        checks.get(lit["condition_id"], {}).get("allowed", False)
        and checks[lit["condition_id"]]["state"] == lit["required_state"] for lit in proof)]
    grounded.sort(key=lambda p: (len(p), sum(len(checks[x["condition_id"]]["matched_evidence"]) for x in p), str(p)))
    selected = grounded[0] if grounded else []
    evidence = _dedup([span for lit in selected for span in checks[lit["condition_id"]]["matched_evidence"]])
    allow = proof_result["complete"] and proof_result["state"] == TRUE and bool(selected)
    reason = ("complete_consistent_true_proof" if allow else "proof_enumeration_incomplete" if not proof_result["complete"]
              else "rule_not_true" if proof_result["state"] != TRUE else "no_complete_consistent_evidence_proof")
    return {"state": proof_result["state"], "allow_attention_intervention": allow,
            "selected_proof": selected, "proof_evidence": evidence,
            "proofs": proof_result["proofs"], "proofs_complete": proof_result["complete"],
            "reason": reason, "condition_checks": checks}


# Unit factors are exact decimals. Clinical chemistry conversions requiring a
# molecular-weight assumption are intentionally unsupported.
_FACTORS = {"length": {"mm": "1", "cm": "10", "m": "1000", "in": "25.4"},
            "duration": {"month": "1", "year": "12"},
            "count": {"count": "1"}, "stage": {"stage": "1"},
            "testosterone": {"ng/dl": "1", "ng/ml": "100"},
            "tmb": {"mut/mb": "1"}}
_NUMERIC_FIELDS = {
    "age": ("duration", "year", (r"age", r"aged")),
    "relapse_months": ("duration", "month", (r"relapse interval", r"time to relapse", r"interval to relapse", r"relapse")),
    "tumor_cm": ("length", "cm", (r"tumou?r diameter", r"tumou?r size", r"invasive tumou?r")),
    "margin_mm": ("length", "mm", (r"resection margin", r"surgical margin", r"closest margin")),
    "liver_mm": ("length", "mm", (r"largest liver nodule", r"liver nodule(?: diameter)?")),
    "lung_mm": ("length", "mm", (r"largest solid pulmonary nodule", r"pulmonary nodule(?: diameter)?", r"lung nodule")),
    "solid_mm": ("length", "mm", (r"solid component(?: of the part-solid pulmonary nodule)?",)),
    "thyroid_cm": ("length", "cm", (r"primary thyroid carcinoma", r"thyroid carcinoma(?: diameter)?")),
    "multifocal_cm": ("length", "cm", (r"additional macroscopic thyroid tumou?r focus", r"additional thyroid focus")),
    "pci": ("count", "count", (r"peritoneal cancer index", r"PCI")),
    "testosterone": ("testosterone", "ng/dl", (r"serum testosterone", r"testosterone")),
    "vessels": ("count", "count", (r"vessel invasion count", r"invaded vessel count", r"vessel count", r"invaded vessels")),
    "pt": ("stage", "stage", (r"pT", r"pathologic(?:al)? T stage")),
    "pn": ("stage", "stage", (r"pN", r"pathologic(?:al)? N stage")),
    "ct": ("stage", "stage", (r"cT", r"clinical T stage")),
    "cn": ("stage", "stage", (r"cN", r"clinical N stage")),
    "ypn": ("stage", "stage", (r"ypN", r"post-neoadjuvant N stage")),
    "tmb": ("tmb", "mut/mb", (r"tumou?r mutational burden", r"TMB")),
    "sln_count": ("count", "count", (r"positive sentinel(?: lymph)? node count", r"sentinel node count")),
    "ax_nodes": ("count", "count", (r"positive axillary(?: lymph)? node count", r"axillary node count")),
}
_NUMBER = r"[-+]?\d+(?:\.\d+)?"
_UNIT = r"(?:ng\s*/\s*(?:dL|mL)|(?:mutations?|mut)\s*/\s*Mb|years?|months?|yrs?|mos?|y/o|yo|inches|inch|mm|cm|in|m)"
_MEASUREMENT = re.compile(r"(?P<lo>" + _NUMBER + r")\s*(?P<u1>" + _UNIT + r")?"
    r"(?:\s*(?:[-–—]|\bto\b|\bthrough\b)\s*(?P<hi>" + _NUMBER + r")\s*(?P<u2>" + _UNIT + r")?)?", re.I)
_SUBJECT = re.compile(r"\b(?:patient(?:'s)?|he|she|his|her)\b|患者|病人", re.I)
_FAMILY = re.compile(r"\b(?:mother|father|sisters?|brothers?|siblings?|daughter|son|aunt|uncle|wife|husband|relatives?|grandmother|grandfather|grandparents?|stepmother|stepfather|nephew|niece|cousins?|partner|spouse|caregiver|companion|family history|(?:another|other|different) patient|someone else)\b|母亲|父亲|姐姐|妹妹|哥哥|弟弟|兄弟姐妹|祖父|祖母|家族史|另一位患者|其他患者", re.I)
_NESTED_OWNER = re.compile(r"\bpatient['’]s\s+(?:[a-z-]+\s+){0,3}[a-z-]+['’]s\b", re.I)
_CURRENT = re.compile(r"\b(?:current|currently|today|now|latest|this visit|present measurement)\b|当前|目前|本次|今日", re.I)
_HISTORICAL = re.compile(r"\b(?:previous(?:ly)?|historical|formerly|baseline|at diagnosis|last (?:year|month)|\d+ (?:years?|months?) ago|in (?:19|20)\d{2})\b|既往|去年|此前|过去", re.I)
_UNCERTAIN = re.compile(r"\b(?:no|not|unknown|uncertain|unavailable|missing|pending|may|might|possibly|approximately|about|withdrawn|corrected|erroneous)\b|未测|不详|未知|撤回", re.I)


def _unit_name(value: str) -> str:
    value = re.sub(r"\s+", "", str(value).lower())
    return {"years": "year", "yr": "year", "yrs": "year", "y/o": "year", "yo": "year",
            "months": "month", "mo": "month", "mos": "month", "inches": "in", "inch": "in",
            "mutations/mb": "mut/mb", "mutation/mb": "mut/mb"}.get(value, value)


def scoped_numeric_observation(condition: dict[str, Any], spans: list[dict[str, Any]]) -> dict[str, Any] | None:
    """An explicit-current patient value may override; all other scopes abstain.

    This checks the patient's own current value in the matching sentence. A
    family or historical mention in the same sentence rejects that entire
    sentence, even if a human could resolve its clauses. No subject or date is
    inferred from unrelated sentences, model guesses, filenames or gold facts.
    """
    spec = condition.get("numeric")
    if not isinstance(spec, dict):
        return None
    field = condition.get("field", spec.get("field"))
    result = {"status": UNKNOWN, "reason": "no_eligible_current_patient_value", "evidence": [],
              "scope_policy": "explicit_patient_and_current_in_same_sentence_v1", "rejected": [], "values": []}
    if field not in _NUMERIC_FIELDS:
        result["reason"] = "unsupported_numeric_field"
        return result
    dimension, canonical, aliases = _NUMERIC_FIELDS[field]
    target_unit = _unit_name(spec.get("unit") or canonical)
    try:
        threshold = Decimal(str(spec["value"]))
    except (KeyError, InvalidOperation, TypeError, ValueError):
        result["reason"] = "invalid_numeric_spec"
        return result
    op = spec.get("operator")
    if not threshold.is_finite() or op not in {"<", "<=", ">", ">=", "==", "!="}:
        result["reason"] = "invalid_numeric_spec"
        return result
    if target_unit not in _FACTORS[dimension]:
        result["reason"] = "unsupported_target_unit"
        return result
    alias_end = r"(?=\b|\d)" if dimension == "stage" else r"\b"
    alias_pattern = re.compile(r"\b(?:" + "|".join(sorted(aliases, key=len, reverse=True)) + r")" + alias_end, re.I)
    observations: list[tuple[tuple[Decimal, ...], dict]] = []
    for span in spans:
        sentence = span["text"]
        anchors = list(alias_pattern.finditer(sentence))
        # Explicit patient age expressed without the word "age".
        if field == "age" and not anchors and re.search(r"\b(?:years?|months?)[ -]old\b|\by/o\b", sentence, re.I):
            anchors = [re.search(r"\b(?:patient|he|she)\b", sentence, re.I)]
            anchors = [x for x in anchors if x is not None]
        if not anchors:
            continue
        rejection = ("non_patient_or_mixed_subject" if _FAMILY.search(sentence) or _NESTED_OWNER.search(sentence) else
                     "unresolved_subject" if not _SUBJECT.search(sentence) else
                     "historical_or_mixed_time" if _HISTORICAL.search(sentence) else
                     "unresolved_time" if not _CURRENT.search(sentence) else
                     "uncertain_negated_or_corrected_value" if _UNCERTAIN.search(sentence) else None)
        if rejection:
            result["rejected"].append({"sentence_id": span["sentence_id"], "reason": rejection})
            # A contradictory/uncertain *current patient* statement cannot be
            # ignored merely because a different sentence contains a point.
            # Clear family-only and historical-only sentences remain excluded.
            if rejection == "uncertain_negated_or_corrected_value":
                result["context_disputed"] = True
            continue
        seen = set()
        for anchor in anchors:
            tail = sentence[anchor.end():]
            # A short, nonnumeric grammatical bridge ties the measurement to
            # this field. Never scan every same-unit number in the sentence.
            bridge = re.match(r"\s*(?:is|was|measures?|measured|equals?|of|:|=|currently|now|today|at|has|a|an|diameter|size|score|count|'s|\s){0,12}\s*", tail, re.I)
            offset = anchor.end() + (bridge.end() if bridge else 0)
            match = _MEASUREMENT.match(sentence, offset)
            if match is None or match.span() in seen:
                continue
            seen.add(match.span())
            after = sentence[match.end():match.end() + 16]
            if re.match(r"\.\d", after) or (re.match(r"\s*(?:/|\d|[a-zA-Z])", after) and not re.match(r"\s*(?:old\b|and\b|or\b|on\b|at\b|with\b|for\b|;|\.)", after, re.I)):
                result["rejected"].append({"sentence_id": span["sentence_id"], "reason": "unparsed_measurement_suffix"})
                continue
            units = (match.group("u1"), match.group("u2"))
            endpoints = (match.group("lo"), match.group("hi"))
            converted = []
            for index, value in enumerate(endpoints):
                if value is None:
                    continue
                source_unit = _unit_name(units[index] or units[1-index] or (canonical if dimension in {"count", "stage"} else ""))
                if source_unit not in _FACTORS[dimension]:
                    converted = []
                    break
                val = Decimal(value)
                if val < 0:
                    converted = []
                    break
                converted.append(val * Decimal(_FACTORS[dimension][source_unit]) / Decimal(_FACTORS[dimension][target_unit]))
            if converted:
                observations.append((tuple(sorted(converted)), span))
            else:
                result["rejected"].append({"sentence_id": span["sentence_id"], "reason": "missing_incompatible_unit_or_negative_value"})
    if not observations:
        return result
    if result.get("context_disputed"):
        result["reason"] = "numeric_context_disputed"
        result["evidence"] = _dedup([span for _, span in observations])
        return result
    compare = {"<": lambda x: x < threshold, "<=": lambda x: x <= threshold,
               ">": lambda x: x > threshold, ">=": lambda x: x >= threshold,
               "==": lambda x: x == threshold, "!=": lambda x: x != threshold}[op]
    outcomes = {compare(value) for endpoints, _ in observations for value in endpoints}
    if op in {"==", "!="} and any(len(vals) == 2 and vals[0] < threshold < vals[1] for vals, _ in observations):
        outcomes.add(compare(threshold))
    result["status"] = TRUE if outcomes == {True} else FALSE if outcomes == {False} else UNKNOWN
    result["reason"] = "deterministic_numeric_comparison" if result["status"] != UNKNOWN else "numeric_range_or_values_cross_threshold"
    result["evidence"] = _dedup([span for _, span in observations])
    result["values"] = [[float(value) for value in vals] for vals, _ in observations]
    result["unit"] = target_unit
    return result


class CAREModule1V7(v6.CAREModule1V6):
    """Separate experimental entry point; retains V6's model adapter contract."""

    def _evaluate_condition(self, patient: str, spans: list[dict[str, Any]], condition: dict[str, Any], kind: str) -> dict[str, Any]:
        # Reuse exactly one existing TrialMatchAI and one NLI evaluation. Drop
        # V6's numeric-derived evidence and recompute the number with V7 scope.
        row = super()._evaluate_condition(patient, spans, condition, kind)
        started = time.perf_counter()
        numeric = scoped_numeric_observation(condition, spans)
        row["numeric_v7_seconds"] = time.perf_counter() - started
        row["numeric"] = numeric
        state = numeric["status"] if numeric and numeric["status"] in (TRUE, FALSE) else row["trialmatch"]["state"]
        row["status"] = row["applicability"] = state
        row["applicability_source"] = "deterministic_numeric_v7" if numeric and numeric["status"] in (TRUE, FALSE) else "trialmatch"
        row["resolution"] = row["applicability_source"]
        row["support_evidence"] = [x for x in row["nli_candidates"] if x["nli_relation"] == "entailment"]
        row["refute_evidence"] = [x for x in row["nli_candidates"] if x["nli_relation"] == "contradiction"]
        check = condition_evidence(row, state, patient)
        row.update({"support_evidence": check["support_evidence"], "refute_evidence": check["refute_evidence"],
                    "evidence_status": check["evidence_status"], "allow_attention_intervention": check["allowed"],
                    "conflict": check["reason"] == "conflicting_or_opposite_evidence", "evidence_gate_check": check,
                    "evidence_state": TRUE if check["support_evidence"] and not check["refute_evidence"] else FALSE if check["refute_evidence"] and not check["support_evidence"] else UNKNOWN})
        return row

    def run(self, guideline: Any, patient: Any, question: str) -> dict[str, Any]:
        rules = v6._rule_list(deepcopy(guideline))
        ids = [rule["rule_id"] for rule in rules]
        if len(ids) != len(set(ids)):
            raise ValueError("V7 requires unique rule IDs")
        for rule in rules:
            conditions = rule["conditions"] + rule["exceptions"]
            cids = [c["condition_id"] for c in conditions]
            if len(cids) != len(set(cids)):
                raise ValueError("V7 requires unique condition IDs within each rule")
            if conditions:
                validate_logic(full_rule_logic(rule), set(cids))
        result = super().run(rules, patient, question)
        patient_text = v6._patient_text(patient)
        by_rule = {r["rule_id"]: r for r in rules}
        for rule in result["rules"]:
            gate = rule_evidence_gate(by_rule[rule["rule_id"]], rule["conditions"] + rule["exceptions"], patient_text=patient_text)
            rule.update({"allow_attention_intervention": gate["allow_attention_intervention"],
                         "selected_proof": gate["selected_proof"], "proof_evidence": gate["proof_evidence"],
                         "proof_gate": gate})
            result["action_gate"][rule["rule_id"]] = float(gate["allow_attention_intervention"])
        outputs = {r["rule_id"]: r for r in result["rules"]}
        result["proof_binding_gate"] = {}
        for rid, rule in outputs.items():
            proof_ids = {x["condition_id"] for x in rule["selected_proof"]}
            result["proof_binding_gate"][rid] = {c["condition_id"]: float(rule["allow_attention_intervention"] and c["condition_id"] in proof_ids)
                for c in rule["conditions"] + rule["exceptions"]}
        for item in result["applicability"]:
            rule = outputs[item["rule_id"]]
            item["allow_attention_intervention"] = rule["allow_attention_intervention"]
            item["decisive_condition_ids"] = [x["condition_id"] for x in rule["selected_proof"]]
        for item in result["relation_weights"]:
            item["action_gate"] = result["action_gate"][item["rule_id"]]
            item["decisive"] = any(lit["condition_id"] == item["condition_id"] for lit in outputs[item["rule_id"]]["selected_proof"])
            item["proof_binding_gate"] = result["proof_binding_gate"][item["rule_id"]][item["condition_id"]]
        result["schema_version"] = VERSION
        result["metadata"].update({"algorithm": "complete_evidence_proof_v1", "default_v6_changed": False,
            "scope_policy": "explicit_patient_and_current_in_same_sentence_v1", "proofs_are_clinically_verified": False,
            "numeric_timing_includes_v6_candidate_pass": True, "scores_are_calibrated": False})
        return result
