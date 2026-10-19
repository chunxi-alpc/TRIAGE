"""Unified semantic applicability interface for the frozen PR11 Qwen experiment.

States come only from semantic model records. Action positions come only from
original rule IDs; evidence quote matching is diagnostic and never gates a state.
"""
from __future__ import annotations
from collections import Counter


def prediction_rules(record):
    if not isinstance(record, dict):
        return []
    if isinstance(record.get("rules"), list):
        return record["rules"]
    for key in ("prediction", "parsed", "structure", "output"):
        value = record.get(key)
        if isinstance(value, dict) and isinstance(value.get("rules"), list):
            return value["rules"]
        if isinstance(value, list):
            return value
    return []


def normalize_structure(case, record):
    """Keep every source rule. Technical failures are null, never medical U."""
    raw = prediction_rules(record)
    counts = Counter(r.get("rule_id") for r in raw if isinstance(r, dict))
    by_id = {r.get("rule_id"): r for r in raw if isinstance(r, dict)}
    rows = []
    for rule in case["rules"]:
        rid = rule["rule_id"]
        got = by_id.get(rid, {})
        state = got.get("state")
        valid = counts[rid] == 1 and state in ("T", "F", "U")
        rows.append({
            "rule_id": rid,
            "state": state if valid else None,
            "evidence_quote": got.get("evidence_quote", ""),
            "reason_short": got.get("reason_short", ""),
            "parse_status": "accepted" if valid else "technical_failure",
        })
    ok = all(r["state"] in ("T", "F", "U") for r in rows)
    return {"status": "ok" if ok else "parse_failure", "parse_failure": not ok,
            "source": "semantic_v8", "rules": rows}


def build_interface(case, prompt, action_spans, structure, offsets=None):
    """Character spans refer to the full Qwen prompt; all intervals are [start,end)."""
    by_id = {r["rule_id"]: r for r in structure["rules"]}
    patient_start = prompt.index("\nPatient: ") + len("\nPatient: ")
    rows = []
    for rule in case["rules"]:
        rid = rule["rule_id"]
        result = by_id[rid]
        action_span = list(action_spans[rid])
        assert prompt[action_span[0]:action_span[1]] == rule["action"]
        row = {"rule_id": rid, "state": result["state"],
               "action_char_span": action_span}
        quote = result.get("evidence_quote", "")
        if isinstance(quote, str) and quote:
            local = case["patient"].find(quote)
            if local >= 0:
                row["optional_evidence_char_span"] = [
                    patient_start + local, patient_start + local + len(quote)]
                row["evidence_location_status"] = "exact_patient_substring"
                row["evidence_occurrences"] = case["patient"].count(quote)
            else:
                row["evidence_location_status"] = "not_exact_patient_substring"
        else:
            row["evidence_location_status"] = "empty_or_invalid_quote"
        if offsets is not None:
            for name, span in (("action_token_indices", action_span),
                               ("optional_evidence_token_indices",
                                row.get("optional_evidence_char_span"))):
                if span is not None:
                    a, b = span
                    row[name] = [i for i, (x, y) in enumerate(offsets)
                                 if y > a and x < b and y > x]
        rows.append(row)
    return rows


def signed_action_targets(interface):
    """Only the preselected C intervention: T positive, F negative, U/error no-op."""
    return [{"rule_id": r["rule_id"], "span": r["action_char_span"],
             "sign": 1 if r["state"] == "T" else -1, "kind": "action",
             "token_indices": r.get("action_token_indices", [])}
            for r in interface if r["state"] in ("T", "F")]
