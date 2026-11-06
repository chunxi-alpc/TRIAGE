"""Automatic numeric C proposals using recorded facts and complete predicates.

Only an explicit comparison already present in the current condition is eligible.
Patients remain unchanged. Every returned proposal still passes construct.validate.
"""
from __future__ import annotations
import copy
import math
import re
from decimal import Decimal

from .common import evaluate
from .data import canonical_guide, condition

COMPARISONS = {"<": "<", "<=": "<=", ">": ">", ">=": ">=",
               "≤": "<=", "≥": ">=", "＜": "<", "＞": ">"}
NUMBER_COMPARISON = re.compile(
    r"(?P<op><=|>=|[<>≤≥＜＞])(?P<gap>\s*)(?P<number>[+-]?(?:\d+(?:\.\d+)?|\.\d+))(?![\d.])"
)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _leaves(expression):
    op = expression.get("op")
    if op in ("and", "or"):
        for child in expression["args"]:
            yield from _leaves(child)
    elif op == "not":
        yield from _leaves(expression["arg"])
    else:
        yield expression


def _replace_threshold(expression, field, op, old, new):
    """Change this comparison wherever repeated; retain all other leaves."""
    result = copy.deepcopy(expression)
    changed = 0
    for leaf in _leaves(result):
        if (leaf.get("field") == field and leaf.get("op") == op
                and _number(leaf.get("value")) and float(leaf["value"]) == old):
            leaf["value"] = int(new) if float(new).is_integer() else float(new)
            changed += 1
    return result, changed


def _text_replacer(op, old, new):
    rendered = str(int(new)) if float(new).is_integer() else format(float(new), ".15g")
    def replace(text):
        def match_value(match):
            if COMPARISONS[match["op"]] == op and float(match["number"]) == old:
                return match["op"] + match["gap"] + rendered
            return match.group(0)
        return NUMBER_COMPARISON.sub(match_value, text)
    return replace


def _thresholds(old, observed):
    """Try actual observations first, then one unit of recorded precision."""
    candidates = sorted(set(float(x) for x in observed), key=lambda x: (abs(x-old), x))
    exponents = [Decimal(str(x)).as_tuple().exponent for x in [old] + list(observed)]
    step = 10.0 ** min(0, *exponents)
    nearby = {x + sign*step for x in candidates for sign in (-1, 1)}
    for value in candidates + sorted(nearby, key=lambda x: (abs(x-old), x)):
        if value != old and math.isfinite(value):
            yield value


def numeric_condition_candidate(group, value):
    """Return a C-schema candidate or None; never mutate group or value.

    The same comparator/threshold is synchronized in all supplied condition
    text, definition quotations and terminal payloads. Whole original fields
    become edits, so structural IDs and 800 are unaffected by an 80 change.
    """
    if group.get("cohort") != "main":
        return None
    try:
        facts = value["facts"]
        original = value["predicate"]
        if len(facts) != 2 or [int(evaluate(original, f)) for f in facts] != [0, 1]:
            return None
        target = condition(group)
        explicit = {(COMPARISONS[m["op"]], float(m["number"]))
                    for m in NUMBER_COMPARISON.finditer(target)}
        tree = canonical_guide(group["guideline"])
        texts = [node["condition"] for node in tree["nodes"]]
        texts += [d["quote"] for d in group.get("definitions", [])]
        texts += list(tree["leaves"].values())
        ancestor_originals = {a["gid"]: a["predicate"]
                              for a in value.get("ancestor_predicates", [])}
        for ancestor in group.get("ancestors", []):
            expression = ancestor_originals[ancestor["gid"]]
            if any(int(evaluate(expression, f)) != int(ancestor["label"]) for f in facts):
                return None
        attempted = set()
        for leaf in _leaves(original):
            op, old, field = leaf.get("op"), leaf.get("value"), leaf.get("field")
            if op not in ("<", "<=", ">", ">=") or not _number(old):
                continue
            old = float(old)
            if (op, old) not in explicit or (field, op, old) in attempted:
                continue
            attempted.add((field, op, old))
            observed = [f[field] for f in facts]
            if not all(_number(x) for x in observed):
                continue
            for new in _thresholds(old, observed):
                predicate, _ = _replace_threshold(original, field, op, old, new)
                labels = [int(evaluate(predicate, f)) for f in facts]
                states = [s for s in (0, 1) if labels[s] != s]
                if not states:
                    continue
                substitute = _text_replacer(op, old, new)
                if substitute(target) == target:
                    continue
                # Existing ancestor meanings remain required. When a referenced
                # threshold is affected, update the corresponding complete
                # expression only if its field/operator/value match is explicit.
                updates = []
                ancestors_ok = True
                for ancestor in group.get("ancestors", []):
                    ae = ancestor_originals[ancestor["gid"]]
                    changed, n_changed = _replace_threshold(ae, field, op, old, new)
                    text_touched = substitute(ancestor["condition"]) != ancestor["condition"]
                    if text_touched and n_changed == 0:
                        ancestors_ok = False
                        break
                    if n_changed:
                        if any(int(evaluate(changed, f)) != int(ancestor["label"]) for f in facts):
                            ancestors_ok = False
                            break
                        updates.append({"gid": ancestor["gid"], "predicate": changed})
                if not ancestors_ok:
                    continue
                edits = []
                for text in dict.fromkeys(texts):
                    new_text = substitute(text)
                    if new_text != text:
                        edits.append({"old": text, "new": new_text})
                # Replacing longest fields first handles literal nested
                # quotations consistently with replace_all's sequential edits.
                edits.sort(key=lambda edit: (-len(edit["old"]), edit["old"]))
                state = states[0]
                return {
                    "valid": True, "base_state": state, "edits": edits,
                    "predicate": predicate, "ancestor_predicates": updates,
                    "reason": "An explicit numeric threshold is changed on the fixed patient; the complete predicate flips and all recorded ancestors are preserved.",
                    "numeric_derivation": {
                        "field": field, "operator": op, "old_threshold": old,
                        "new_threshold": new, "original_labels": [0, 1],
                        "modified_labels": labels, "facts_changed": False,
                        "source": "deterministic_complete_predicate_threshold_construction",
                    },
                }
    except (KeyError, ValueError, TypeError, IndexError, OverflowError):
        return None
    return None
