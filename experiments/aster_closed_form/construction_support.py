"""Mechanical C-side repair and exact text sources for reference updates.

These helpers never rewrite clinical facts or condition expressions. Repair is
only possible while the generator still proposes condition_change.valid=True.
"""
from .common import evaluate
from .data import canonical_guide


def source_edit_text(group):
    """Text that a C edit may update, excluding IDs and provenance metadata.

    The caller separately requires a change in the current condition or a
    necessary definition. Terminal payloads are included so redundant condition
    thresholds there can be synchronized with that primary edit.
    """
    tree = canonical_guide(group["guideline"])
    parts = [node["condition"] for node in tree["nodes"]]
    parts.extend(definition["quote"] for definition in group.get("definitions", []))
    parts.extend(tree["leaves"].values())
    return "\n".join(parts)


def repair_condition_state(group, value):
    """Select the other fixed patient state only if its complete truth flips.

    Mutates only condition_change.base_state and adds repaired_base_state.
    Returns whether a repair was made. A rejected/unavailable C is never enabled.
    Both original states must reproduce their historical 0/1 truth and original
    ancestor outcomes; the chosen changed-condition state must also preserve
    every ancestor, using updated predicates when provided.
    """
    change = value.get("condition_change", {})
    if change.get("valid") is not True:
        return False
    try:
        state = int(change["base_state"])
        if state not in (0, 1):
            return False
        facts = value["facts"]
        if len(facts) != 2:
            return False
        original_labels = [int(evaluate(value["predicate"], fact)) for fact in facts]
        if original_labels != [0, 1]:
            return False
        changed_labels = [int(evaluate(change["predicate"], fact)) for fact in facts]
        if changed_labels[state] != state:
            return False
        candidate = 1 - state
        if changed_labels[candidate] == candidate:
            return False
        originals = {entry["gid"]: entry["predicate"]
                     for entry in value.get("ancestor_predicates", [])}
        updates = {entry["gid"]: entry["predicate"]
                   for entry in change.get("ancestor_predicates", [])}
        for ancestor in group.get("ancestors", []):
            gid, expected = ancestor["gid"], int(ancestor["label"])
            original = originals[gid]
            if any(int(evaluate(original, fact)) != expected for fact in facts):
                return False
            touched = any(edit.get("old") and edit["old"] in ancestor["condition"]
                          for edit in change.get("edits", []))
            if touched and gid not in updates:
                return False
            updated = updates.get(gid, original)
            if int(evaluate(updated, facts[candidate])) != expected:
                return False
    except (KeyError, ValueError, TypeError, IndexError):
        return False
    change["base_state"] = candidate
    change["repaired_base_state"] = {
        "from": state, "to": candidate,
        "source": "complete_predicate_evaluation_on_unchanged_facts_and_ancestor_predicates",
        "original_labels": original_labels,
        "modified_labels": changed_labels,
        "facts_changed": False, "predicate_changed": False,
    }
    return True
