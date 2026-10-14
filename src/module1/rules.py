"""Question/patient retrieval and cached guideline parsing (no answer inputs)."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from pathlib import Path

PARSER_VERSION = "module1-rule-parser-v1"
SYSTEM_PROMPT = """You parse clinical guideline clauses into executable conditions. Return one JSON object only.
Schema: {"conditions":[{"condition_id":"c1","text":"verbatim or faithful full condition","numeric":{"field":"age","operator":">=","value":18,"unit":"years"},"negated":false}],"logic":{"op":"AND","args":["c1"]},"exceptions":[],"exception_logic":{"op":"OR","args":[]},"action":"complete recommended action"}.
The numeric object is optional. Use field, operator (<, <=, >, >=, ==, !=), numeric value, and unit only for explicit thresholds. Preserve all conditions, exclusions, negations, and nested AND/OR/NOT relations. Do not invent patient facts or use reference answers. Every condition and exception needs a unique ID; exception IDs start with e. Exceptions override applicability. The logical args must reference the corresponding IDs or nested logical objects. If there are no applicability restrictions, return conditions=[] and logic={"op":"AND","args":[]}. Only do this for an explicitly unconditional recommendation. Never omit a condition to shorten the response. Return {"parse_error":"reason"} if the full clause cannot be represented. Do not put the recommended action in conditions. negated=true means logical inversion of the provided underlying condition text and numeric test. Do not encode negation twice: if text or numeric operator already expresses the complete negative condition, set negated=false. Prefer complete numeric operators with negated=false. For inverted decision-tree branches use positive underlying text with negated=true."""


def _clauses(guideline):
    if isinstance(guideline, dict):
        if "rules" in guideline:
            entries = guideline["rules"]
        elif "conditions" in guideline or "action" in guideline:
            entries = [guideline]
        elif isinstance(guideline.get("text"), str):
            return _clauses(guideline["text"])
        else:
            raise ValueError("guideline requires rules, text, or a structured rule")
    elif isinstance(guideline, str):
        # Preserve every original character offset; no truncation before parsing.
        pieces = list(re.finditer(r"\S(?:[^\n]|\n(?!\s*\n))*", guideline))
        if len(pieces) <= 1:
            pieces = list(re.finditer(r"[^\n]+", guideline))
        result = []
        for match in pieces:
            text = match.group().strip()
            if not text:
                continue
            start = match.start() + len(match.group()) - len(match.group().lstrip())
            result.append({"rule_id": f"r{len(result)+1}", "text": text,
                           "source_span": {"start": start, "end": start+len(text),
                                           "unit": "character", "index": len(result)}})
        return result
    elif isinstance(guideline, (list, tuple)):
        entries = guideline
    else:
        raise TypeError("guideline must be text, a list, or a rule dictionary")
    if not isinstance(entries, (list, tuple)):
        raise TypeError("guideline.rules must be a list")
    result = []
    used_ids = set()
    for index, entry in enumerate(entries):
        if isinstance(entry, str):
            entry = {"text": entry}
        if not isinstance(entry, dict):
            raise TypeError("each guideline clause must be text or a dictionary")
        item = copy.deepcopy(entry)
        text = item.get("text") or item.get("rule_text") or item.get("recommendation")
        if not isinstance(text, str) or not text.strip():
            if "conditions" not in item:
                raise ValueError("guideline clause has no text or structured conditions")
            text = json.dumps({k: v for k, v in item.items() if k != "source_span"}, ensure_ascii=False)
        item["text"] = text
        rule_id = str(item.get("rule_id") or item.get("id") or f"r{index+1}")
        if rule_id in used_ids:
            rule_id = f"{rule_id}_{index+1}"
        used_ids.add(rule_id)
        item["rule_id"] = rule_id
        item.setdefault("source_span", {"start": 0, "end": len(text), "unit": "character", "index": index})
        result.append(item)
    return result


def _conditions(items, prefix):
    if items is None:
        return []
    if not isinstance(items, list):
        raise ValueError("conditions and exceptions must be lists")
    result = []
    for index, item in enumerate(items):
        if isinstance(item, str):
            item = {"text": item}
        if not isinstance(item, dict) or not isinstance(item.get("text"), str) or not item["text"].strip():
            raise ValueError("condition requires nonempty text")
        condition = copy.deepcopy(item)
        condition["condition_id"] = str(condition.get("condition_id") or condition.get("id") or f"{prefix}{index+1}")
        condition["negated"] = bool(condition.get("negated", False))
        numeric = condition.get("numeric")
        if numeric is not None:
            if not isinstance(numeric, dict) or not all(k in numeric for k in ("field", "operator", "value")):
                raise ValueError("numeric condition requires field, operator, and value")
            numeric["operator"] = {"gt": ">", "gte": ">=", "lt": "<", "lte": "<=", "eq": "==", "ne": "!=", "=": "=="}.get(numeric["operator"], numeric["operator"])
            if numeric["operator"] not in ("<", "<=", ">", ">=", "==", "!="):
                raise ValueError("unsupported numeric operator")
            numeric["value"] = float(numeric["value"])
            if not math.isfinite(numeric["value"]):
                raise ValueError("numeric threshold must be finite")
            numeric.setdefault("unit", "")
        result.append(condition)
    ids = [item["condition_id"] for item in result]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate condition IDs")
    return result


def _logic(value, ids, default):
    if value is None:
        value = default
    if isinstance(value, str):
        if value in ids:
            return value
        value = {"op": value.upper(), "args": list(ids)}
    if not isinstance(value, dict):
        raise ValueError("logic must be AND, OR, NOT, a condition ID, or an object")
    op = str(value.get("op", "")).upper()
    args = value.get("args", list(ids))
    if op not in ("AND", "OR", "NOT") or not isinstance(args, list):
        raise ValueError("unsupported logical expression")
    if op == "NOT" and len(args) != 1:
        raise ValueError("NOT requires exactly one argument")
    converted = []
    for arg in args:
        if isinstance(arg, str):
            if arg not in ids:
                raise ValueError(f"logic refers to unknown condition {arg}")
            converted.append(arg)
        else:
            converted.append(_logic(arg, ids, default))
    return {"op": op, "args": converted}


def _refs(logic):
    if isinstance(logic, str):
        return {logic}
    return set().union(*(_refs(arg) for arg in logic["args"])) if logic["args"] else set()


def _normalize(raw, source):
    if not isinstance(raw, dict):
        raise ValueError("rule parser must return an object")
    if raw.get("parse_error"):
        raise ValueError(str(raw["parse_error"]))
    if "conditions" not in raw:
        raise ValueError("rule parser omitted conditions")
    conditions = _conditions(raw["conditions"], "c")
    exceptions = _conditions(raw.get("exceptions", []), "e")
    ids = [c["condition_id"] for c in conditions]
    exception_ids = [c["condition_id"] for c in exceptions]
    if set(ids) & set(exception_ids):
        raise ValueError("condition and exception IDs overlap")
    logic = _logic(raw.get("logic"), ids, "AND")
    exception_logic = _logic(raw.get("exception_logic"), exception_ids, "OR")
    if _refs(logic) != set(ids) or _refs(exception_logic) != set(exception_ids):
        raise ValueError("logical expression omitted conditions")
    action = raw.get("action") or raw.get("recommended_action")
    if not isinstance(action, str) or not action.strip():
        raise ValueError("rule parser omitted recommended action")
    return {"rule_id": source["rule_id"], "text": source["text"],
            "source_span": source["source_span"], "conditions": conditions,
            "logic": logic, "exceptions": exceptions, "exception_logic": exception_logic,
            "action": action, "parse_status": "ok"}


def _identity(backend):
    identity = {"parser_version": PARSER_VERSION, "backend_class": type(backend).__qualname__}
    for attr in ("cache_identity", "model_identity", "model_path", "qwen_path", "model_version"):
        value = getattr(backend, attr, None)
        if value is not None and not callable(value):
            identity[attr] = str(value)
    # A config change at the same model path invalidates the parsing cache.
    for attr in ("model_path", "qwen_path"):
        value = getattr(backend, attr, None)
        if isinstance(value, (str, Path)):
            config = Path(value) / "config.json"
            if config.is_file():
                identity[f"{attr}_config_sha256"] = hashlib.sha256(config.read_bytes()).hexdigest()
    return identity


def select_and_parse(guideline, question, backend, top_k=12, cache_dir=None, patient=None):
    """Preserve small guideline coverage; union question and patient retrieval otherwise."""
    clauses = _clauses(guideline)
    if not clauses:
        return [], []
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    texts = [clause["text"] for clause in clauses]
    def score(query):
        values = [float(x) for x in backend.score_pairs(query, texts)]
        if len(values) != len(clauses) or not all(math.isfinite(x) for x in values):
            raise ValueError("relevance scorer returned invalid scores")
        return [max(0.0, min(1.0, x)) for x in values]
    question_scores = score(question)
    patient_scores = list(question_scores)
    if patient:
        # Separate complete sentences avoid burying patient facts behind the question
        # in the cross-encoder query token budget. No reference/path labels are used.
        from .trialgpt_matching import sentences_from_patient
        sentences, _ = sentences_from_patient(patient)
        for sentence in sentences:
            values = score(question + "\nPatient evidence: " + sentence)
            patient_scores = [max(a, b) for a, b in zip(patient_scores, values)]
    scores = patient_scores
    question_order = sorted(range(len(clauses)), key=lambda i: (-question_scores[i], i))
    combined_order = sorted(range(len(clauses)), key=lambda i: (-scores[i], i))
    if patient and len(clauses) <= 24:
        # Small diagnosis runs can inspect the whole guideline before applicability.
        selected = combined_order
    else:
        selected = question_order[:min(4, top_k)]
        selected += [i for i in combined_order if i not in selected][:max(0, top_k-len(selected))]
    question_rank = {index: rank+1 for rank, index in enumerate(question_order)}
    combined_rank = {index: rank+1 for rank, index in enumerate(combined_order)}
    rules, relevance = [], []
    cache_root = Path(cache_dir) if cache_dir else None
    if cache_root:
        cache_root.mkdir(parents=True, exist_ok=True)
    identity = _identity(backend)
    for index in selected:
        clause = clauses[index]
        relevance.append({"rule_id": clause["rule_id"], "score": scores[index],
            "question_score": question_scores[index], "question_rank": question_rank[index],
            "combined_rank": combined_rank[index], "retrieval_method": "question_patient_union" if patient else "question_only",
            "all_small_guideline_rules_retained": bool(patient and len(clauses) <= 24)})
        key_data = {"identity": identity, "text": clause["text"]}
        cache_key = hashlib.sha256(json.dumps(key_data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        cache_file = cache_root / f"{cache_key}.json" if cache_root else None
        raw = None
        cache_hit = False
        try:
            if "conditions" in clause:
                raw = clause
            elif cache_file and cache_file.is_file():
                try:
                    cached = json.loads(cache_file.read_text())
                    if cached.get("key_data") == key_data:
                        raw = cached["parsed"]
                        cache_hit = True
                except (OSError, ValueError, KeyError, TypeError):
                    raw = None
            if raw is None:
                raw = backend.json(SYSTEM_PROMPT, "Parse this complete guideline clause:\n"+clause["text"], max_new_tokens=2048)
            rule = _normalize(raw, clause)
            if cache_file and "conditions" not in clause and not cache_hit:
                temp = cache_file.with_suffix(f".{os.getpid()}.tmp")
                try:
                    temp.write_text(json.dumps({"key_data": key_data, "parsed": raw}, ensure_ascii=False, indent=2))
                    temp.replace(cache_file)
                except OSError:
                    if temp.exists():
                        temp.unlink()
                    rule["cache_warning"] = "Could not write rule parsing cache"
            rule["parse_cache_hit"] = cache_hit
        except Exception as error:
            rule = {"rule_id": clause["rule_id"], "text": clause["text"],
                    "source_span": clause["source_span"], "conditions": [],
                    "logic": {"op": "AND", "args": []}, "exceptions": [],
                    "exception_logic": {"op": "OR", "args": []},
                    "action": str(clause.get("action", "")), "parse_status": "error",
                    "parse_error": f"{type(error).__name__}: {error}", "parse_cache_hit": cache_hit}
        rules.append(rule)
    return rules, relevance
