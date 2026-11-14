"""Full-context prompts. No branch is removed using a predicted or true state."""
from __future__ import annotations

import hashlib
import json

PREAMBLE = (
    "This is an invented routing simulation for logic research. The numerical "
    "fields and routing rules below are fictional and are not medical guidance. "
    "Use only current observations about the patient. Historical or family-member "
    "observations do not establish the patient's current state. Missing or "
    "conflicting current observations are unknown, not false. Apply the rules "
    "using three-valued logic. If a unique route cannot be established, request review."
)


def render_tree(tree):
    if isinstance(tree, str):
        return tree
    op, args = tree["op"], tree["args"]
    if not args:
        return "TRUE" if op == "AND" else "FALSE"
    if op == "NOT":
        return f"NOT ({render_tree(args[0])})"
    return "(" + f" {op} ".join(render_tree(arg) for arg in args) + ")"


def build_guidance(rules):
    atoms = {a["condition_id"]: a
             for rule in rules for a in rule["conditions"] + rule.get("exceptions", [])}
    lines = ["Atomic conditions:"] + [f"{key}: {atom['text']}" for key, atom in sorted(atoms.items())]
    spans = {}
    text = "\n".join(lines) + "\nRouting rules:\n"
    for rule in rules:
        phrase = f"{rule['rule_id']}: IF {render_tree(rule['logic'])}"
        exceptions = rule.get("exception_logic", {"op": "OR", "args": []})
        if exceptions.get("args") if isinstance(exceptions, dict) else True:
            phrase += f" UNLESS {render_tree(exceptions)}"
        phrase += ", assign "
        start = len(text) + len(phrase)
        text += phrase + rule["action"] + ".\n"
        spans[rule["rule_id"]] = [start, start + len(rule["action"])]
    text += "If no unique route is established, Request review."
    return text, spans


def full_context(case, structure=None):
    guidance, local_spans = build_guidance(case["rules"])
    before = PREAMBLE + "\n\nQuestion: " + case["question"] + "\nPatient: " + case["patient"] + "\nGuidance:\n"
    prompt = before + guidance + "\nOptions:\n"
    prompt += "\n".join(f"{letter}. {answer}" for letter, answer in case["option_map"].items())
    spans = {rid: [len(before) + start, len(before) + end]
             for rid, (start, end) in local_spans.items()}
    if structure is not None:
        prompt += "\nApplicability assessment (T=true, F=false, U=unknown):\n"
        if structure.get("status") != "ok":
            prompt += "Assessment parsing failed; no applicability conclusion is supplied.\n"
        else:
            for rule in structure["rules"]:
                prompt += f"{rule['rule_id']}: {rule['state']}\n"
    prompt += "\nReturn exactly one option letter.\nAnswer:"
    return prompt, spans


def prompt_sha256(prompt):
    return hashlib.sha256(prompt.encode()).hexdigest()


def prediction_input(case):
    """Positive allowlist excludes labels, structured facts, and oracle structure."""
    rules = []
    allowed_rule = {"rule_id", "action", "conditions", "logic", "exceptions", "exception_logic"}
    for rule in case["rules"]:
        rules.append({key: json.loads(json.dumps(value))
                      for key, value in rule.items() if key in allowed_rule})
    return {"guideline": rules, "patient": case["patient"], "question": case["question"]}
