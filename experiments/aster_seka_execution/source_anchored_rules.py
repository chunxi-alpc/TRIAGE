"""Recover a rule passage from exact original anchors; never rewrite clinical text."""
from bisect import bisect_left
from dataclasses import replace
import json

from clinical_core import parse_quoted_rules, parse_quoted_rules_individually, sha

VERSION = "original_source_span_reconstruction_v1"


def occurrences(text, quote):
    if not isinstance(quote, str) or not quote:
        raise ValueError("Nonempty original quotation required")
    result, offset = [], 0
    while True:
        start = text.find(quote, offset)
        if start < 0:
            break
        result.append(start)
        offset = start + 1
    if not result:
        raise ValueError("Quotation is absent from original source")
    return result


def enclosing_source_span(guideline, anchors):
    """Shortest original interval containing an occurrence of each exact anchor.

    Equal-length alternatives use the earliest source position, never outcomes.
    The original intervening qualifiers and text are retained in full.
    """
    anchors = list(dict.fromkeys(anchors))
    if not anchors:
        raise ValueError("At least one anchor required")
    locations = [occurrences(guideline, q) for q in anchors]
    best = None
    for left in sorted({i for group in locations for i in group}):
        selected = []
        for quote, starts in zip(anchors, locations):
            index = bisect_left(starts, left)
            if index == len(starts):
                break
            selected.append((starts[index], starts[index] + len(quote)))
        if len(selected) != len(anchors):
            continue
        start, end = min(x[0] for x in selected), max(x[1] for x in selected)
        candidate = (end - start, start, end)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise ValueError("Cannot bind all original anchors")
    return best[1], best[2]


def parse_source_anchored_rules(raw, guideline):
    old_rules, audits = parse_quoted_rules_individually(raw, guideline)
    rules = {rule.rule_id: rule for rule in old_rules}
    seen = {(r.condition_quote, r.action_quote, r.exception_quotes, r.rule_quote): r.rule_id
            for r in old_rules}
    for entry in audits:
        entry["source_span_parser_version"] = VERSION
        if entry["status"] != "rejected" or entry.get("truncated"):
            continue
        original_error = entry.get("error")
        item = entry.get("raw_item")
        if not isinstance(item, dict) or set(item) != {"condition_quote", "action_quote", "exception_quotes", "rule_quote"}:
            continue
        try:
            c, a, exceptions = item["condition_quote"], item["action_quote"], item["exception_quotes"]
            if not isinstance(c, str) or not isinstance(a, str) or not c or not a or not isinstance(exceptions, list):
                raise ValueError("Invalid source anchor fields")
            if a in c:
                raise ValueError("Condition still contains the complete proposed action")
            start, end = enclosing_source_span(guideline, [c, a, *exceptions])
            recovered = {**item, "rule_quote": guideline[start:end]}
            rule = parse_quoted_rules(json.dumps([recovered], ensure_ascii=False), guideline)[0]
            rule = replace(rule, rule_id=entry["rule_id"])
            assert rule.start == start and rule.end == end
            key = (rule.condition_quote, rule.action_quote, rule.exception_quotes, rule.rule_quote)
            entry.update(original_status="rejected", original_error=original_error,
                         source_span=[start, end], reconstructed_passage_sha256=sha(rule.rule_quote),
                         original_generated_passage_sha256=sha(str(item["rule_quote"])),
                         reconstruction="Exact source anchors; original intervening text retained",
                         semantic_correctness="not established by source matching")
            if key in seen:
                entry.update(status="duplicate", duplicate_of=seen[key])
            else:
                seen[key] = rule.rule_id
                rules[rule.rule_id] = rule
                entry["status"] = "original_source_span_reconstructed"
        except Exception as error:
            entry["source_span_reconstruction_error"] = str(error)
    return tuple(sorted(rules.values(), key=lambda x: x.rule_id)), audits
