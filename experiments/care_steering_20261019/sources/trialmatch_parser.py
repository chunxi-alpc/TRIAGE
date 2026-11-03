"""Parse one TrialMatchAI condition without models, heuristics, or gold labels.

Only a unique, verbatim target Criterion and an exact official Classification
supply the decision. Presentation and the trial-wide Final Decision do not.
"""
from __future__ import annotations

import json
import re
from typing import Any

PARSER_VERSION = "trialmatch-condition-json-v2.1"
INCLUSION_LABELS = {"Met": "T", "Not Met": "F", "Unclear": "U", "Irrelevant": "U"}
EXCLUSION_LABELS = {"Violated": "T", "Not Violated": "U", "Unclear": "U", "Irrelevant": "U"}
ARRAY_KEYS = {"inclusion": "Inclusion_Criteria_Evaluation", "exclusion": "Exclusion_Criteria_Evaluation"}
FINAL_DECISIONS = {"Eligible", "Likely Eligible", "Likely Ineligible", "Ineligible",
    "Likely Eligible (leaning toward inclusion)", "Likely Ineligible (leaning toward exclusion)"}


def strip_thinking(raw: str) -> str:
    # Phi-4 sometimes uses a second <think> as its closing delimiter. Never
    # accept a label found only inside a complete or unfinished thinking block.
    clean = re.sub(r"<think>.*?(?:</think>|<think>)", "", raw, flags=re.DOTALL)
    unfinished = clean.find("<think>")
    return (clean[:unfinished] if unfinished != -1 else clean).strip()


def _unique_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate_json_key:" + key)
        value[key] = item
    return value


def _invalid_constant(value):
    raise ValueError("invalid_json_constant:" + value)


def _container_end(text: str, start: int) -> int:
    """Skip a malformed JSON container whole; never salvage its nested keys."""
    stack, in_string, escaped = [], False, False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "[{":
            stack.append(char)
        elif char in "]}":
            if not stack or (stack.pop(), char) not in (("{", "}"), ("[", "]")):
                return index + 1
            if not stack:
                return index + 1
    return len(text)


def _schema_objects(text: str):
    """Read complete root objects amid fences/prose, preserving ambiguity.

    A successfully decoded outer object/list is consumed whole, so JSON-looking
    text inside a Justification cannot masquerade as a second final response.
    Duplicate keys or non-JSON constants are rejected rather than overwritten.
    """
    decoder = json.JSONDecoder(object_pairs_hook=_unique_keys, parse_constant=_invalid_constant)
    objects, failures, cursor = [], [], 0
    while cursor < len(text):
        match = re.search(r"[\[{]", text[cursor:])
        if not match:
            break
        start = cursor + match.start()
        try:
            value, end = decoder.raw_decode(text, start)
        except (json.JSONDecodeError, ValueError) as exc:
            failures.append(str(exc))
            cursor = max(start + 1, _container_end(text, start))
            continue
        if isinstance(value, dict) and any(key in value for key in ARRAY_KEYS.values()):
            objects.append((value, start, end))
        cursor = end
    return objects, failures


def parse_trialmatch_response(raw: str, criterion_text: str, criterion_kind: str = "inclusion",
                             *, truncated: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {"parser_version": PARSER_VERSION, "raw_label": None,
        "state": "U", "parse_error": None, "parse_status": "invalid_output",
        "unknown_reason": None, "truncated": bool(truncated), "parse_warnings": []}

    def reject(error, status="invalid_output", detail=None):
        result.update(parse_error=error, parse_status=status, unknown_reason=status)
        if detail is not None:
            result["parse_error_detail"] = detail
        return result

    if criterion_kind not in ARRAY_KEYS:
        return reject("invalid_criterion_kind")
    if not isinstance(raw, str) or not isinstance(criterion_text, str) or not criterion_text.strip():
        return reject("invalid_parser_input")
    if truncated:
        return reject("generation_truncated", "truncated")
    clean = strip_thinking(raw)
    schemas, failures = _schema_objects(clean)
    if any(error.startswith(("duplicate_json_key:", "invalid_json_constant:")) for error in failures):
        return reject("invalid_json_response", detail="; ".join(failures))
    if not schemas:
        return reject("invalid_json_response" if failures or not clean.startswith(("{", "[")) else "missing_schema_fields")
    if len(schemas) != 1:
        return reject("multiple_schema_objects", "ambiguous_target")
    data, start, end = schemas[0]
    result["json_object_span"] = {"start": start, "end": end, "source": "thinking_removed_output"}
    if clean[:start].strip() or clean[end:].strip():
        result["parse_warnings"].append("presentation_text_outside_json")
    target = data.get(ARRAY_KEYS[criterion_kind])
    if not isinstance(target, list):
        return reject("missing_or_invalid_target_array", "ambiguous_target")
    candidates = [(index, item) for index, item in enumerate(target)
                  if isinstance(item, dict) and isinstance(item.get("Criterion"), str)]
    matches = [(index, item) for index, item in candidates
               if item["Criterion"].strip() == criterion_text.strip()]
    match_kind = "exact"
    if not matches:
        # Only JSON whitespace reflow is normalized. Wording, punctuation and
        # case remain exact; paraphrases are never treated as the requested atom.
        compact = lambda value: re.sub(r"\s+", " ", value).strip()
        matches = [(index, item) for index, item in candidates
                   if compact(item["Criterion"]) == compact(criterion_text)]
        match_kind = "whitespace_normalized_exact"
    result["target_array_count"] = len(target)
    result["exact_target_count"] = len(matches)
    result["target_match_kind"] = match_kind if matches else None
    if not matches:
        return reject("criterion_text_mismatch" if target else "target_criterion_missing", "ambiguous_target")
    if len(matches) != 1:
        return reject("duplicate_target_criterion", "ambiguous_target")
    index, entry = matches[0]
    label = entry.get("Classification")
    allowed = EXCLUSION_LABELS if criterion_kind == "exclusion" else INCLUSION_LABELS
    if not isinstance(label, str) or label not in allowed:
        return reject("invalid_classification", "invalid_label")
    other_kind = "exclusion" if criterion_kind == "inclusion" else "inclusion"
    other = data.get(ARRAY_KEYS[other_kind], [])
    if not isinstance(other, list):
        return reject("invalid_non_target_array")
    placeholder = re.compile(rf"no (?:specific )?{other_kind} criteria(?: (?:were |are |have been )?"
                             r"(?:provided|specified|listed|mentioned|given))?\.?", re.IGNORECASE)
    ignored = [item for item in other if isinstance(item, dict)
        and item.get("Classification") == "Irrelevant" and isinstance(item.get("Criterion"), str)
        and placeholder.fullmatch(item["Criterion"].strip())]
    # The opposite-side array is outside this one-atom request. Never use it as
    # evidence for this label, and do not reject a sound unique target because
    # the model copied additional trial-level criteria there.
    result["non_target_entry_count"] = len(other)
    if ignored:
        result["ignored_non_target_placeholders"] = ignored
    if len(other) > len(ignored):
        result["parse_warnings"].append("non_target_entries_ignored")
    if len(target) > 1:
        result["parse_warnings"].append("additional_target_array_entries_ignored")
        result["ignored_target_entry_count"] = len(target) - 1
    if not isinstance(entry.get("Justification"), str):
        result["parse_warnings"].append("missing_or_invalid_justification")
    decision = data.get("Final Decision")
    if not isinstance(decision, str) or decision not in FINAL_DECISIONS:
        result["parse_warnings"].append("invalid_or_missing_final_decision_not_used")
    unknown = {"Unclear": "medical_unknown", "Irrelevant": "irrelevant_condition",
               "Not Violated": "exclusion_not_violated"}.get(label)
    result.update(raw_label=label, state=allowed[label], parse_status="accepted", unknown_reason=unknown,
        justification=entry.get("Justification"), parsed_response=data,
        matched_criterion_index=index, matched_criterion=entry["Criterion"])
    return result
