"""Verified, counterbalanced applicability triplets from development trees.

Only tree-predicate logic is labelled. These explicit synthetic findings are not
validated clinical vignettes. No module-one outputs or benchmark answers enter.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Iterable


VARIANTS = ("positive", "negative", "neutral")
DEVELOPMENT_SPLITS = {"train", "train_val", "development", "dev", "calibration"}
DEFAULT_REVISION = "8f058c88ce1ed43693785b435ad0ff99907a2890"


def _development_only(split: str) -> None:
    if split not in DEVELOPMENT_SPLITS:
        raise ValueError(f"Expected an explicit development source split, got {split!r}")


def load_development_trees(path, source_split="train_val"):
    """Read original JSONL, rejecting a named test path even if mislabelled."""
    _development_only(source_split)
    path = Path(path)
    if any(part.lower() in {"test", "testing", "test.jsonl"} for part in path.parts):
        raise ValueError("Module-two fitting must not read a test path")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _validate_tree(node, indices=None):
    indices = set() if indices is None else indices
    if not isinstance(node, dict) or "index" not in node:
        raise ValueError("Every tree node must have an original index")
    key = str(node["index"])
    if key in indices:
        raise ValueError("Duplicate node index")
    indices.add(key)
    if node.get("leaf") is True:
        if not isinstance(node.get("conclusion"), str) or not node["conclusion"].strip():
            raise ValueError("Leaf requires an action")
        return
    if not isinstance(node.get("description"), str) or not node["description"].strip():
        raise ValueError("Branch requires a predicate")
    for branch in ("yes", "no"):
        _validate_tree(node.get(branch), indices)


def execute_tree(node, facts):
    """Independent three-valued tree execution; missing facts yield no leaf."""
    while not node.get("leaf"):
        value = facts.get(str(node["index"]))
        if value is None:
            return None
        if not isinstance(value, bool):
            raise ValueError("Facts must be Boolean or unknown")
        node = node["yes" if value else "no"]
    return node["index"]


def rule_applicability(conditions, facts):
    """A false required predicate dominates unknown; otherwise use AND logic."""
    values = [facts.get(str(c["node_index"])) for c in conditions]
    if any(v is not None and v != c["required"] for c, v in zip(conditions, values)):
        return False
    return None if any(v is None for v in values) else True


def _terminal_siblings(node, ancestors=(), max_depth=6):
    if node.get("leaf") or len(ancestors) >= max_depth:
        return
    predicate = {"node_index": str(node["index"]), "text": node["description"].strip()}
    children = (node["yes"], node["no"])
    texts = [c["text"].strip().casefold() for c in ancestors] + [predicate["text"].casefold()]
    if all(child.get("leaf") is True for child in children) and len(texts) == len(set(texts)):
        # Identical actions cannot identify action routing and are not swapped.
        if children[0]["conclusion"].strip() != children[1]["conclusion"].strip():
            yield {"parent": node, "ancestors": list(ancestors), "predicate": predicate}
    for branch in (True, False):
        child = node["yes" if branch else "no"]
        yield from _terminal_siblings(child, (*ancestors, {**predicate, "required": branch}), max_depth)


def split_guidelines(group_ids, seed=261027):
    """Deterministic 50/25/25 split, at the source-guideline level."""
    groups = sorted(set(group_ids), key=lambda key: (_digest([seed, key]), key))
    if len(groups) < 3:
        raise ValueError("Need at least three independent guidelines for fit/select/confirm")
    n_select = max(1, len(groups) // 4)
    n_confirm = max(1, len(groups) // 4)
    n_fit = len(groups) - n_select - n_confirm
    return {group: ("fit" if i < n_fit else "select" if i < n_fit + n_select else "confirm")
            for i, group in enumerate(groups)}


def _render_prompt(conditions, facts, rules, order, target_id):
    chunks = []
    offset = 0
    def append(text):
        nonlocal offset
        start = offset
        chunks.append(text)
        offset += len(text)
        return [start, offset]
    append("Patient findings (UNKNOWN means not recorded):\n")
    patient_start = offset
    for condition in conditions:
        value = facts[str(condition["node_index"])]
        word = "UNKNOWN" if value is None else "YES" if value else "NO"
        append(f"- {condition['text']} Finding: {word}.\n")
    patient_span = [patient_start, offset]
    append("\nQuestion:\n")
    question_span = append("Which rule applies to the patient? Answer with only A or B.")
    append("\n\nGuideline (all conditions in a rule are required):\n")
    guide_start = offset
    rule_spans = {}
    for rule_id in order:
        rule = rules[rule_id]
        condition_text = " AND ".join(f"[{c['text']}] = {'YES' if c['required'] else 'NO'}"
                                       for c in rule["conditions"])
        append("Conditions: ")
        condition_span = append(condition_text)
        append(". Rule ")
        id_span = append(rule_id)
        append(". Action: ")
        action_span = append(rule["action"])
        append("\n")
        rule_spans[rule_id] = {"condition": [condition_span], "id": [id_span], "action": [action_span]}
    guide_span = [guide_start, offset]
    append("\nAnswer:")
    spans = {**rule_spans[target_id], "patient": [patient_span], "question": [question_span],
             "guide": [guide_span], "rules": rule_spans}
    return "".join(chunks), spans


def build_minimal_pairs(records: Iterable[dict], *, source_split="train_val", seed=261027,
                        max_groups=None, max_pairs_per_tree=1, max_trees_per_group=1, max_depth=6, language="en",
                        source_revision=DEFAULT_REVISION):
    """Return triplets and an audit; each sibling group contributes eight rows.

    The eight rows cross target branch, two action permutations and two rule
    orders. All variants of the original source guideline stay in one split.
    """
    _development_only(source_split)
    if min(max_pairs_per_tree, max_trees_per_group, max_depth) < 1 or (max_groups is not None and max_groups < 3):
        raise ValueError("Positive limits and at least three source guidelines are required")
    candidates, skipped = [], Counter()
    source_count = 0
    seen = set()
    for record in records:
        source_count += 1
        if record.get("split", record.get("source_split", source_split)) != source_split:
            raise ValueError("Mixed or mislabeled source splits are forbidden")
        if record.get("language") != language:
            skipped["other_language"] += 1
            continue
        tree_id = record.get("filename")
        guideline_id = record.get("guideline_id")
        if not tree_id or guideline_id is None:
            skipped["missing_source_identity"] += 1
            continue
        if tree_id in seen:
            skipped["duplicate_tree"] += 1
            continue
        seen.add(tree_id)
        try:
            payload = json.loads(record["text"]) if isinstance(record.get("text"), str) else record["text"]
            root = payload["decision_tree"]
            _validate_tree(root)
            pairs = list(_terminal_siblings(root, max_depth=max_depth))
        except (ValueError, TypeError, KeyError):
            skipped["invalid_tree"] += 1
            continue
        if not pairs:
            skipped["no_eligible_terminal_siblings"] += 1
            continue
        pairs.sort(key=lambda pair: _digest([seed, tree_id, pair["parent"]["index"]]))
        candidates.append({"tree_id": str(tree_id), "group_id": f"guideline:{guideline_id}",
                           "guideline_id": guideline_id, "root": root,
                           "source_tree_sha256": _digest(payload),
                           "pairs": pairs[:max_pairs_per_tree]})
    all_groups = sorted({item["group_id"] for item in candidates}, key=lambda key: _digest([seed, "sample", key]))
    selected_groups = set(all_groups if max_groups is None else all_groups[:max_groups])
    assignments = split_guidelines(selected_groups, seed=seed)
    by_group = defaultdict(list)
    for source in candidates:
        by_group[source["group_id"]].append(source)
    selected_trees = set()
    for group in selected_groups:
        choices = sorted(by_group[group], key=lambda item: _digest([seed, "tree", item["tree_id"]]))
        selected_trees.update(item["tree_id"] for item in choices[:max_trees_per_group])
    rows = []
    for source in sorted(candidates, key=lambda row: row["tree_id"]):
        if source["tree_id"] not in selected_trees:
            continue
        for pair in source["pairs"]:
            parent, ancestors, predicate = pair["parent"], pair["ancestors"], pair["predicate"]
            cluster_id = f"{source['tree_id']}:node:{parent['index']}"
            yes_id = "A" if int(_digest([seed, cluster_id])[-1], 16) % 2 == 0 else "B"
            ids = {True: yes_id, False: "B" if yes_id == "A" else "A"}
            for permutation in (0, 1):
                rules = {}
                for branch in (True, False):
                    leaf = parent["yes" if branch else "no"]
                    action_leaf = parent["yes" if branch != bool(permutation) else "no"]
                    rules[ids[branch]] = {"leaf_index": leaf["index"], "action_source_leaf": action_leaf["index"],
                        "conditions": [*ancestors, {**predicate, "required": branch}],
                        "action": action_leaf["conclusion"].strip()}
                for order_index in (0, 1):
                    order = [ids[True], ids[False]]
                    if order_index:
                        order.reverse()
                    for target_branch in (True, False):
                        target_id = ids[target_branch]
                        prompts, spans, patient_facts, oracle = {}, {}, {}, {}
                        for variant, decisive_value in zip(VARIANTS, (target_branch, not target_branch, None)):
                            facts = {str(c["node_index"]): c["required"] for c in ancestors}
                            facts[str(predicate["node_index"])] = decisive_value
                            leaf = execute_tree(source["root"], facts)
                            applicable = rule_applicability(rules[target_id]["conditions"], facts)
                            expected = True if variant == "positive" else False if variant == "negative" else None
                            expected_leaf = (parent["yes" if decisive_value else "no"]["index"]
                                             if decisive_value is not None else None)
                            if applicable is not expected or leaf != expected_leaf:
                                raise ValueError("Independent tree and rule oracles disagree")
                            prompts[variant], spans[variant] = _render_prompt(
                                [*ancestors, predicate], facts, rules, order, target_id)
                            patient_facts[variant] = facts
                            oracle[variant] = {"target_applicable": applicable, "selected_leaf": leaf,
                                "selected_rule_id": ids[decisive_value] if decisive_value is not None else None}
                        rows.append({"schema_version": "module2.triplet.v1",
                            "id": f"{cluster_id}:swap{permutation}:order{order_index}:target{target_id}",
                            "tree_id": source["tree_id"], "group_id": source["group_id"],
                            "pair_cluster_id": cluster_id, "target_rule_id": target_id,
                            "source_split": source_split, "split": assignments[source["group_id"]],
                            "prompts": prompts, "spans": spans, "patient_facts": patient_facts,
                            "rules": rules, "oracle": oracle,
                            "decisive_node_index": str(predicate["node_index"]),
                            "controls": {"action_permutation": permutation, "rule_order": order,
                                "rule_order_index": order_index, "target_branch": target_branch},
                            "provenance": {"dataset": "MEGA-CDP", "source_revision": source_revision,
                                "guideline_id": source["guideline_id"], "source_tree_sha256": source["source_tree_sha256"],
                                "synthetic_patient": True, "predicate_level_logic_only": True,
                                "module1_outputs_used": False, "benchmark_answers_used": False,
                                "local_sibling_guideline": True, "full_source_tree_retained_in_input": True}})
    audit = audit_triplets(rows)
    audit.update({"source_records": source_count, "eligible_guidelines": len(all_groups),
                  "skipped": dict(skipped), "seed": seed, "source_revision": source_revision,
                  "max_depth": max_depth, "max_pairs_per_tree": max_pairs_per_tree,
                  "max_trees_per_group": max_trees_per_group,
                  "group_split_assignments": assignments})
    return rows, audit


def audit_triplets(rows):
    """Validate construction invariants independently of model/tokenization."""
    if not rows:
        raise ValueError("No triplets to audit")
    split_groups, clusters, seen = defaultdict(set), defaultdict(list), set()
    for row in rows:
        if row["id"] in seen:
            raise ValueError("Duplicate triplet identifier")
        seen.add(row["id"])
        _development_only(row["source_split"])
        split_groups[row["split"]].add(row["group_id"])
        clusters[row["pair_cluster_id"]].append(row)
        positive = row["patient_facts"]["positive"]
        negative = row["patient_facts"]["negative"]
        neutral = row["patient_facts"]["neutral"]
        decisive = row["decisive_node_index"]
        if set(positive) != set(negative) or set(positive) != set(neutral):
            raise ValueError("Patient fact sets differ")
        if [key for key in positive if positive[key] != negative[key]] != [decisive]:
            raise ValueError("Pair must change exactly the decisive predicate")
        if neutral[decisive] is not None or any(neutral[k] != positive[k] for k in positive if k != decisive):
            raise ValueError("Neutral case changes a non-decisive fact")
        guide_texts, questions = set(), set()
        for variant in VARIANTS:
            text, spans = row["prompts"][variant], row["spans"][variant]
            if not text.endswith("\nAnswer:"):
                raise ValueError("Decision cue must follow the complete guideline")
            for role in ("action", "id", "condition", "patient", "question", "guide"):
                for start, end in spans[role]:
                    if not 0 <= start < end <= len(text):
                        raise ValueError("Invalid character span")
            for rid, rule_spans in spans["rules"].items():
                condition_end = rule_spans["condition"][-1][1]
                id_start, id_end = rule_spans["id"][0]
                action_start, action_end = rule_spans["action"][0]
                if not condition_end <= id_start < id_end <= action_start:
                    raise ValueError("Rule ID and action must follow complete conditions")
                if text[id_start:id_end] != rid or text[action_start:action_end] != row["rules"][rid]["action"]:
                    raise ValueError("Rule span text differs from its annotation")
            if spans["patient"][-1][1] > spans["question"][0][0] or spans["question"][-1][1] > spans["guide"][0][0]:
                raise ValueError("Expected patient -> question -> guideline")
            if any(spans[role] != spans["rules"][row["target_rule_id"]][role] for role in ("action", "id", "condition")):
                raise ValueError("Top-level target spans do not match the target rule")
            guide_texts.add(text[slice(*spans["guide"][0])])
            questions.add(text[slice(*spans["question"][0])])
            expected = True if variant == "positive" else False if variant == "negative" else None
            if rule_applicability(row["rules"][row["target_rule_id"]]["conditions"], row["patient_facts"][variant]) is not expected:
                raise ValueError("Applicability verification failed")
        if len(guide_texts) != 1 or len(questions) != 1:
            raise ValueError("The guideline and question must remain identical within a triplet")
    split_names = list(split_groups)
    for i, left in enumerate(split_names):
        for right in split_names[i + 1:]:
            if split_groups[left] & split_groups[right]:
                raise ValueError("A source guideline leaks across development partitions")
    expected_cells = {(target, swap, order) for target in (False, True) for swap in (0, 1) for order in (0, 1)}
    for cluster in clusters.values():
        cells = {(row["controls"]["target_branch"], row["controls"]["action_permutation"],
                  row["controls"]["rule_order_index"]) for row in cluster}
        if len(cluster) != 8 or cells != expected_cells:
            raise ValueError("Incomplete target/action/order counterbalancing block")
    return {"schema_version": "module2.pair_audit.v1", "status": "passed", "triplets": len(rows),
            "prompt_instances": 3 * len(rows), "source_guidelines": len(set().union(*split_groups.values())),
            "unique_prompts": len({prompt for row in rows for prompt in row["prompts"].values()}),
            "source_trees": len({row["tree_id"] for row in rows}), "sibling_groups": len(clusters),
            "partitions": {name: {"guidelines": len(groups),
                "triplets": sum(row["split"] == name for row in rows)} for name, groups in sorted(split_groups.items())},
            "variants_are_not_independent_samples": True, "token_alignment": "not_checked_until_tokenization",
            "clinical_fact_consistency": "not_validated; source tree predicates treated as explicit Boolean findings"}
