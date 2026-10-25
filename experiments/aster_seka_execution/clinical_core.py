"""Label-free clinical rule quoting, lexical evidence and exact chat span mapping.

No reference trees, reference paths, applicability labels or answer keys enter
these functions. Exact quotation validation does not prove clinical correctness.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import re
from typing import Any

from aster_seka_adapter import PreparedPrompt, PredictedSpans, Span


SELECTOR_VERSION = "aster.lexical_quotes.v1"
EXTRACTION_PARSER_VERSION = "complete_array_items_literal_invalid_escapes_v2"
EXTRACTION_INSTRUCTION = """Extract conditional clinical rules from the supplied guideline only.
Return only a JSON array. Each item has exactly these keys:
condition_quote: a verbatim condition, without its proposed action;
action_quote: the corresponding verbatim recommended action;
exception_quotes: a list of verbatim exceptions, or [];
rule_quote: a contiguous verbatim passage containing the condition, action and exceptions.
Keep original spelling and punctuation. Do not invent missing conditions or actions.
Do not use external knowledge. Preserve every rule you can identify, including exceptions.
This is an unverified extraction, not a clinical decision. The guideline follows:\n"""
STOP = frozenset("a an the and or of to in on for with without is are was were be been as at by from if then than that this these those patient patients should may can must do does not no have has had its it their them they".split())


def sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


@dataclass(frozen=True)
class QuotedRule:
    rule_id: str
    condition_quote: str
    action_quote: str
    exception_quotes: tuple[str, ...]
    rule_quote: str
    start: int
    end: int
    duplicate_quote_occurrences: int


def parse_quoted_rules(raw: str, guideline: str) -> tuple[QuotedRule, ...]:
    cleaned = raw.strip()
    if cleaned.startswith("```json") and cleaned.endswith("```"):
        cleaned = cleaned[7:-3].strip()
    elif cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned[3:-3].strip()
    parsed = json.loads(cleaned)
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("Extraction must be a nonempty JSON array")
    keys = {"condition_quote", "action_quote", "exception_quotes", "rule_quote"}
    rules, seen = [], set()
    for item in parsed:
        if not isinstance(item, dict) or set(item) != keys:
            raise ValueError("Extraction rule schema mismatch")
        exceptions = item["exception_quotes"]
        if not isinstance(exceptions, list):
            raise ValueError("exception_quotes must be a list")
        quoted = [item["condition_quote"], item["action_quote"], item["rule_quote"], *exceptions]
        if any(not isinstance(q, str) or not q or q not in guideline for q in quoted):
            raise ValueError("Every quotation must occur verbatim in the original guideline")
        passage = item["rule_quote"]
        if any(q not in passage for q in [item["condition_quote"], item["action_quote"], *exceptions]):
            raise ValueError("Rule passage must contain its condition, action and exceptions")
        if item["action_quote"] in item["condition_quote"]:
            raise ValueError("Condition field includes the complete proposed action")
        key = (item["condition_quote"], item["action_quote"], tuple(exceptions), passage)
        if key in seen:
            continue
        seen.add(key)
        start = guideline.index(passage)
        rules.append(QuotedRule(f"R{len(rules) + 1:04d}", item["condition_quote"], item["action_quote"],
                                tuple(exceptions), passage, start, start + len(passage), guideline.count(passage)))
    return tuple(rules)


def _extraction_array_items(raw: str):
    """Split array elements without interpreting their clinical text or guessing tails."""
    cleaned = raw.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:].strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
    if not cleaned.startswith("["):
        raise ValueError("Extraction must be a nonempty JSON array")
    items, start, depth, in_string, escaped, closed = [], 1, 0, False, False, False
    for index in range(1, len(cleaned)):
        char = cleaned[index]
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
            depth += 1
        elif char == "]" and depth == 0:
            if cleaned[start:index].strip():
                items.append((start, index, cleaned[start:index], True))
            closed = True
            break
        elif char in "}]":
            depth -= 1
        elif char == "," and depth == 0:
            items.append((start, index, cleaned[start:index], True))
            start = index + 1
    if not closed and cleaned[start:].strip():
        items.append((start, len(cleaned), cleaned[start:], depth == 0 and not in_string))
    if not items:
        raise ValueError("Extraction must contain at least one complete or truncated array item")
    return items, closed


def _literal_invalid_json_escapes(raw: str):
    """Escape only invalid JSON backslashes; valid short/Unicode escapes survive."""
    output, repairs, index, in_string = [], [], 0, False
    while index < len(raw):
        char = raw[index]
        if char == '"':
            in_string = not in_string
            output.append(char)
            index += 1
        elif char == "\\" and in_string:
            following = raw[index + 1:index + 2]
            if following and following in '"\\/bfnrt':
                output.append(raw[index:index + 2])
                index += 2
            elif following == "u" and len(raw[index + 2:index + 6]) == 4 and all(
                    digit in "0123456789abcdefABCDEF" for digit in raw[index + 2:index + 6]):
                output.append(raw[index:index + 6])
                index += 6
            else:
                output.append("\\\\")
                repairs.append({"item_character_offset": index, "original_escape_prefix": raw[index:index + 2],
                                "repair": "literal_backslash_only"})
                index += 1
        else:
            output.append(char)
            index += 1
    return "".join(output), repairs


def parse_quoted_rules_individually(raw: str, guideline: str):
    """Validate completed items independently; preserve indices and reject tails.

    Invalid JSON backslashes become literal backslashes, never guessed clinical
    content. Every recovered quotation still passes the unchanged exact-source
    validator. Raw text, repairs, malformed items and truncation are all audited.
    """
    items, closed = _extraction_array_items(raw)
    valid, audits, seen = [], [], {}
    for index, (start, end, raw_item, complete) in enumerate(items, 1):
        entry = {"original_array_index": index, "rule_id": f"R{index:04d}",
                 "parser_version": EXTRACTION_PARSER_VERSION, "array_closed": closed,
                 "cleaned_array_character_range": [start, end], "raw_item_json": raw_item,
                 "raw_item_json_sha256": sha(raw_item), "item_sha256": sha(raw_item),
                 "truncated": not complete, "invalid_escape_repairs": []}
        if not complete:
            entry.update(status="rejected", error_type="TruncatedArrayItem",
                         error="Incomplete trailing array item; no missing content was reconstructed")
            audits.append(entry)
            continue
        try:
            repaired, repairs = _literal_invalid_json_escapes(raw_item)
            entry.update(invalid_escape_repairs=repairs, repaired_item_json_sha256=sha(repaired))
            item = json.loads(repaired)
            serialized = json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            entry.update(raw_item=item, item_sha256=sha(serialized))
            rule = parse_quoted_rules(json.dumps([item], ensure_ascii=False), guideline)[0]
            rule = replace(rule, rule_id=f"R{index:04d}")
            key = (rule.condition_quote, rule.action_quote, rule.exception_quotes, rule.rule_quote)
            if key in seen:
                entry.update(status="duplicate", duplicate_of=seen[key])
            else:
                seen[key] = rule.rule_id
                valid.append(rule)
                entry["status"] = "quotes_validated"
        except Exception as error:
            entry.update(status="rejected", error_type=type(error).__name__, error=str(error))
        audits.append(entry)
    return tuple(valid), audits


def terms(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9_-]*|[\u4e00-\u9fff]", text)
            if w.lower() not in STOP}


def sentence_ranges(text: str):
    for match in re.finditer(r"[^.!?。！？;；\n]+[.!?。！？;；\n]*", text):
        if match.group().strip():
            yield match.start(), match.end()


def lexical_patient_spans(patient: str, condition: str, rule_id: str) -> tuple[Span, ...]:
    vocabulary = terms(condition)
    return tuple(Span("patient_history", start, end, rule_id)
                 for start, end in sentence_ranges(patient)
                 if vocabulary & terms(patient[start:end]))


def assemble(header: str, fields: list[tuple[str, str]], tail: str,
             evidence: PredictedSpans, stage: str, output_kind: str) -> PreparedPrompt:
    text, offsets = header, {}
    for name, content in fields:
        text += f"\n[{name}]\n"
        offsets[name] = (len(text), content)
        text += content
    return PreparedPrompt(text + tail, offsets, evidence, stage, output_kind)


def empty_evidence() -> PredictedSpans:
    return PredictedSpans(sha(SELECTOR_VERSION), ())


def condition_prompt(patient: str, rules: tuple[QuotedRule, ...], rule_id: str) -> PreparedPrompt:
    rule = next(r for r in rules if r.rule_id == rule_id)
    # Every supplied rule field contains only its quoted condition, never action or
    # rule passage. Treatment history in the original patient text stays visible.
    evidence = lexical_patient_spans(patient, rule.condition_quote, rule_id) + (
        Span(f"condition:{rule_id}", 0, len(rule.condition_quote), rule_id),)
    return assemble(
        "Judge whether the named condition is true for this patient, false, or unknown. "
        "T means supported by the patient history; F means contradicted; U means insufficient information. "
        "Do not infer missing facts or choose any treatment. Other conditions are reference text only.",
        [("patient_history", patient)] + [(f"condition:{r.rule_id}", r.condition_quote) for r in rules],
        f"\nAssess condition {rule_id}. Answer with exactly one token: T, F, or U.\n",
        PredictedSpans(sha(SELECTOR_VERSION), evidence), "condition", "tfu")


def chat_prompt(prompt: PreparedPrompt, tokenizer: Any) -> PreparedPrompt:
    messages = [{"role": "system", "content": prompt.text}]
    kwargs = {"add_generation_prompt": True, "enable_thinking": False}
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
    if rendered.count(prompt.text) != 1:
        raise ValueError("Chat template must embed the complete original content verbatim exactly once")
    expected = tokenizer.apply_chat_template(messages, tokenize=True, **kwargs)
    actual = tokenizer(rendered, add_special_tokens=True)["input_ids"]
    if actual != expected:
        raise ValueError("Chat template token IDs disagree with the StageSEKAAdapter encoding")
    shift = rendered.index(prompt.text)
    return replace(prompt, text=rendered,
                   fields={name: (offset + shift, text) for name, (offset, text) in prompt.fields.items()})


def official_prompt(template: str, guideline: str, patient: str,
                    evidence: PredictedSpans | None = None, tfu: list[dict] | None = None) -> PreparedPrompt:
    substitutions = {"Guideline": ("guideline", guideline), "Case": ("patient_history", patient)}
    text, fields, last = "", {}, 0
    for match in re.finditer(r"\{(Guideline|Case)\}", template):
        text += template[last:match.start()]
        name, value = substitutions[match.group(1)]
        if name in fields:
            raise ValueError("Repeated template source placeholder needs an explicit multi-span adaptation")
        fields[name] = (len(text), value)
        text += value
        last = match.end()
    text += template[last:]
    if set(fields) != {"guideline", "patient_history"}:
        raise ValueError("Official template must include the original guideline and patient")
    if tfu is not None:
        text += ("\nAuxiliary predicted condition assessments (unverified; unknown is not false). "
                 "Use the complete original guideline and patient history above. Preserve the required output format.\n"
                 + json.dumps(tfu, ensure_ascii=False, sort_keys=True))
    return PreparedPrompt(text, fields, evidence or empty_evidence(), "execution", "path")


def support_spans(rules: tuple[QuotedRule, ...], tfu: list[dict]) -> tuple[Span, ...]:
    true_ids = {p["rule_id"] for p in tfu if p["label"] == "T"}
    # Unknown rules remain in the full original context; no gold path is cropped.
    seen, selected = set(), []
    for rule in rules:
        interval = (rule.start, rule.end)
        if rule.rule_id in true_ids and interval not in seen:
            seen.add(interval)
            selected.append(Span("guideline", *interval, rule.rule_id))
    return tuple(selected)


def token_groups(prompt: PreparedPrompt, tokenizer: Any, pool: tuple[Span, ...], excluded: set[int]):
    encoded = tokenizer(prompt.text, return_offsets_mapping=True, add_special_tokens=True)
    offsets = encoded["offset_mapping"]
    # Unicode byte pieces may share character offsets; keep those tokens together.
    groups = {}
    for span in pool:
        offset, source = prompt.fields[span.source]
        a, b = offset + span.start, offset + span.end
        for i, (start, end) in enumerate(offsets):
            if end > start and a <= start < end <= b:
                key = (span.source, start - offset, end - offset)
                groups.setdefault(key, (set(), span.rule_id))[0].add(i)
    return [(key, ids, rule_id) for key, (ids, rule_id) in groups.items() if not ids & excluded]


def counted_control_spans(prompt: PreparedPrompt, adapter: Any, pool: tuple[Span, ...],
                          disjoint: bool) -> tuple[Span, ...]:
    _, _, original = adapter._encode(prompt, prompt.evidence.spans)
    target = int(original.sum())
    excluded = set(original[0].nonzero().flatten().tolist()) if disjoint else set()
    selected, count = [], 0
    for (source, start, end), indices, rule_id in token_groups(prompt, adapter.operator.tok, pool, excluded):
        if count + len(indices) <= target:
            selected.append(Span(source, start, end, rule_id))
            count += len(indices)
        if count == target:
            break
    if not selected or count != target:
        return ()
    _, _, changed = adapter._encode(prompt, selected)
    if int(changed.sum()) != target or bool((changed == original).all()) or (disjoint and bool((changed & original).any())):
        return ()
    return tuple(selected)


def attach_controls(prompt: PreparedPrompt, adapter: Any, rules: tuple[QuotedRule, ...],
                    target_rule: str | None = None) -> PreparedPrompt:
    all_fields = tuple(Span(name, 0, len(value), "wrong")
                       for name, (_, value) in prompt.fields.items() if value)
    wrong = counted_control_spans(prompt, adapter, all_fields, disjoint=True)
    rotation = {rule.rule_id: rules[(index + 1) % len(rules)] for index, rule in enumerate(rules)}
    misplaced = []
    if len(rules) > 1:
        if target_rule is not None:
            replacement = rotation[target_rule]
            misplaced = [Span(f"condition:{replacement.rule_id}", 0,
                              len(replacement.condition_quote), target_rule)]
            misplaced += list(lexical_patient_spans(prompt.fields["patient_history"][1],
                                                    replacement.condition_quote, target_rule))
        else:
            for original in prompt.evidence.spans:
                replacement = rotation[original.rule_id]
                misplaced.append(Span("guideline", replacement.start, replacement.end, original.rule_id))
    misaligned = counted_control_spans(prompt, adapter, tuple(misplaced), disjoint=False) if misplaced else ()
    return replace(prompt, evidence=replace(prompt.evidence, wrong_spans=wrong, misaligned_rule_spans=misaligned))


def verify_tfu_tokens(tokenizer: Any) -> dict[str, int]:
    mapping = {}
    for label in ("T", "F", "U"):
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1 or tokenizer.decode(ids) != label:
            raise ValueError("T/F/U must each have one distinct exact token representation")
        mapping[label] = ids[0]
    if len(set(mapping.values())) != 3:
        raise ValueError("T/F/U token IDs collide")
    return mapping
