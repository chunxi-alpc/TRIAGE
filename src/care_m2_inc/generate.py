"""Deterministic synthetic mechanism data for the new N1--N4 tracks."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

from .io import canonical_json, sha256_obj, write_json, write_jsonl, hash_manifest
from .oracle import atoms, canonical_ast, dq_label, evaluate

SEED = 261014
N1_SPLITS = {"train": set(range(1, 13)), "dev": set(range(13, 17)), "test": set(range(17, 25))}
N3_SPLITS = {"train": set(range(1, 13)), "dev": set(range(13, 17)), "test": set(range(17, 25))}
N4_SPLITS = {"train": set(range(1, 3)), "dev": set(range(3, 5)), "test": set(range(5, 9))}


def _split(index: int, table: dict[str, set[int]]) -> str:
    return next(name for name, members in table.items() if index in members)


def _templates(prefix: str, count: int) -> list[dict[str, Any]]:
    """Create structurally different ASTs; family-specific atoms avoid aliases."""
    patterns: list[tuple[str, Any]] = [
        ("and3", {"op": "AND", "args": ["a", "b", "c"]}),
        ("or3", {"op": "OR", "args": ["a", "b", "c"]}),
        ("and_or", {"op": "AND", "args": ["a", {"op": "OR", "args": ["b", "c"]}]}),
        ("or_and", {"op": "OR", "args": ["a", {"op": "AND", "args": ["b", "c"]}]}),
        ("not_and", {"op": "NOT", "args": [{"op": "AND", "args": ["a", "b"]}]}),
        ("not_or", {"op": "NOT", "args": [{"op": "OR", "args": ["a", "b"]}]}),
        ("xor3", {"op": "XOR", "args": ["a", "b", "c"]}),
        ("equiv", {"op": "EQUIV", "args": ["a", "b", "c"]}),
        ("atleast2", {"op": "ATLEAST", "args": ["a", "b", "c"], "threshold": 2}),
        ("atleast1_4", {"op": "ATLEAST", "args": ["a", "b", "c", "d"], "threshold": 1}),
        ("atleast3_4", {"op": "ATLEAST", "args": ["a", "b", "c", "d"], "threshold": 3}),
        ("or_not", {"op": "OR", "args": ["a", {"op": "NOT", "args": ["b"]}, "c"]}),
    ]
    result = []
    for index in range(1, count + 1):
        label, template = patterns[(index - 1) % len(patterns)]
        # Changing arity and nesting on the second cycle prevents an AST hash
        # collision across families while retaining a compact evaluator.
        if index > len(patterns):
            template = {"op": "AND", "args": [template, f"e{index}"]}
            label = f"wrapped_{label}"
        names = [f"{prefix}{index:02d}_{letter}" for letter in "abcde"]
        mapping = {old: new for old, new in zip("abcde", names)}
        def rename(node: Any) -> Any:
            if isinstance(node, str):
                return mapping.get(node, node)
            if isinstance(node, dict):
                out = {"op": node["op"], "args": [rename(arg) for arg in node["args"]]}
                if "threshold" in node:
                    out["threshold"] = node["threshold"]
                return out
            return node
        ast = rename(template)
        result.append({"family_id": f"{prefix}{index:02d}", "mechanism_id": f"{label}_{index:02d}", "ast": ast,
                       "ast_hash": sha256_obj(canonical_ast(ast)), "atoms": atoms(ast)})
    return result


def _find_assignments(ast: Any) -> tuple[dict[str, bool], dict[str, bool]]:
    names = atoms(ast)
    values = list(itertools.product((False, True), repeat=len(names)))
    positives = []
    negatives = []
    for assignment in values:
        row = dict(zip(names, assignment))
        if evaluate(ast, row) is True:
            positives.append(row)
        else:
            negatives.append(row)
    if not positives or not negatives:
        raise AssertionError(f"AST has no two-valued contrast: {ast}")
    return positives[len(positives) // 2], negatives[len(negatives) // 2]


def _n1_records() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, family in enumerate(_templates("n1_", 24), 1):
        positive, negative = _find_assignments(family["ast"])
        for r_bit, a_bit in itertools.product((False, True), repeat=2):
            values = positive if a_bit else negative
            query = ("Which target recommendation should be applied?" if r_bit else
                     "Which unrelated administrative fact is recorded? The rule is outside this question.")
            gold = "TARGET" if r_bit and a_bit else "ALTERNATIVE" if r_bit else "IRRELEVANT"
            answer = {"TARGET": "A", "ALTERNATIVE": "B", "IRRELEVANT": "C"}[gold]
            facts = " ".join(f"{name} is {'present' if value else 'absent'}." for name, value in values.items())
            rule = f"Apply the target action when this rule is satisfied: {canonical_json(canonical_ast(family['ast']))}."
            prompt = ("This is a synthetic decision task. Use only the stated facts.\n"
                      f"Question: {query}\nRule: {rule}\nFacts: {facts}\n"
                      "Answer with A, B, or C.")
            root = f"{family['family_id']}_root"
            rows.append({"task": "N1", "case_id": f"{root}_R{int(r_bit)}A{int(a_bit)}", "family_id": family["family_id"],
                         "underlying_mechanism_id": family["mechanism_id"], "ast_hash": family["ast_hash"],
                         "root_case_id": root, "pair_id": root, "fact_group_id": root,
                         "split": _split(index, N1_SPLITS), "prompt_text": prompt,
                         "gold_answer": answer, "gold_action": gold, "gold_R": r_bit, "gold_A": a_bit,
                         "control_target": {"R": r_bit, "A": a_bit}, "values": values,
                         "rule_ast": family["ast"], "options": {"A": "TARGET", "B": "ALTERNATIVE", "C": "IRRELEVANT"},
                         "origin_dataset": "care_m2_n1_generated_v1"})
    return rows


def _n2_records() -> list[dict[str, Any]]:
    rows = []
    routes = [("Cedar", "Maple"), ("Quartz", "Willow"), ("Amber", "Slate"), ("Birch", "Coral"),
              ("Dawn", "Frost"), ("Elm", "Grove"), ("Harbor", "Ivory"), ("Jade", "Linen"),
              ("Mica", "Navy"), ("Olive", "Pine"), ("Rose", "Stone"), ("Teal", "Umber")]
    for index, (c_route, p_route) in enumerate(routes, 1):
        fact_group = f"n2_fact_{index:02d}"
        receiver = f"receiver_{index:02d}"
        donor_index = (index % len(routes)) + 1
        donor = f"donor_{donor_index:02d}"
        prompt = ("This is a synthetic source arbitration task.\n"
                  f"Context note: the current record states that the recommended route is {c_route}.\n"
                  f"Question: A separate closed-book record may contain a different route. Which route follows for this record?\n"
                  "Answer with A or B; do not explain.")
        split = "train" if index <= 6 else "dev" if index <= 8 else "test"
        rows.append({"task": "N2", "case_id": receiver, "receiver_id": receiver, "donor_id": donor,
                     "family_id": f"n2_family_{index:02d}", "underlying_mechanism_id": f"source_pair_{index:02d}",
                     "root_case_id": receiver, "pair_id": receiver, "fact_group_id": fact_group, "split": split,
                     "prompt_text": prompt, "gold_answer_C": "A", "gold_answer_P": "B",
                     "gold_source_answer_C": c_route, "gold_source_answer_P": p_route,
                     "gold_answer": "A", "control_target": {"source": "C"},
                     "source_context": c_route, "source_parametric": p_route,
                     "track": "synthetic_prior", "p_verified": False, "origin_dataset": "care_m2_n2_generated_v1",
                     "options": {"A": c_route, "B": p_route}})
    return rows


def _n3_records() -> list[dict[str, Any]]:
    rows = []
    for index, family in enumerate(_templates("n3_", 24), 1):
        names = atoms(family["ast"])
        positive, negative = _find_assignments(family["ast"])
        cases = []
        # D with an irrelevant unknown (the first atom is hidden only when the
        # formula is already decided by a short-circuiting false/true branch).
        for values in (dict(positive), dict(negative)):
            unknown = {names[0]: None, **{name: value for name, value in values.items() if name != names[0]}}
            label, possible, attrs = dq_label(family["ast"], unknown)
            if label == "D":
                cases.append(("irrelevant_unknown", unknown, label, possible, attrs))
        # A decisive unknown: retain a satisfying assignment for all other atoms.
        unknown = dict(positive)
        unknown[names[0]] = None
        label, possible, attrs = dq_label(family["ast"], unknown)
        if label != "Q":
            unknown = dict(negative)
            unknown[names[0]] = None
            label, possible, attrs = dq_label(family["ast"], unknown)
        if label != "Q":
            # The search is exhaustive; this family still gets a valid Q by
            # hiding all atoms, unless the AST is a constant (which it is not).
            unknown = {name: None for name in names}
            label, possible, attrs = dq_label(family["ast"], unknown)
        cases.append(("decisive_unknown", unknown, label, possible, attrs))
        # Include a fully observed D row so the unknown wording is shared by
        # D and Q only through the row's unrelated missing note.
        full = dict(positive)
        full_label, full_possible, full_attrs = dq_label(family["ast"], full)
        cases.append(("all_known", full, full_label, full_possible, full_attrs))
        for case_number, (kind, values, label, possible, attrs) in enumerate(cases, 1):
            facts = " ".join(f"{name} is {'present' if value is True else 'absent' if value is False else 'not recorded'}." for name, value in values.items())
            if label == "D":
                only = sorted(possible)[0]
                gold = only
                answer_text = "A means execute the rule; B means do not execute; Q means ask."
            else:
                gold = "Q"
                answer_text = "A means execute the rule; B means do not execute; Q means ask the missing decisive attribute."
            prompt = ("This is a synthetic decision sufficiency task. Missing or unresolved information is not false.\n"
                      f"Rule AST: {canonical_json(canonical_ast(family['ast']))}\nFacts: {facts}\n"
                      "There may be a missing observation. Answer with A, B, or Q.\n" + answer_text)
            root = f"{family['family_id']}_root_{case_number:02d}"
            rows.append({"task": "N3", "case_id": root, "family_id": family["family_id"],
                         "underlying_mechanism_id": family["mechanism_id"], "ast_hash": family["ast_hash"],
                         "root_case_id": root, "pair_id": root, "fact_group_id": root,
                         "split": _split(index, N3_SPLITS), "prompt_text": prompt, "gold_answer": gold,
                         "gold_DQ": label, "valid_question_attributes": attrs, "possible_actions": sorted(possible),
                         "values": values, "rule_ast": family["ast"], "control_target": {"DQ": label},
                         "origin_dataset": "care_m2_n3_generated_v1", "options": {"A": "execute", "B": "do_not_execute", "Q": "ask"}})
    return rows


def _n4_records() -> list[dict[str, Any]]:
    rows = []
    for index, family in enumerate(_templates("n4_", 8), 1):
        positive, negative = _find_assignments(family["ast"])
        for source, dq, applicability, order in itertools.product(("C", "P"), ("D", "Q"), ("T", "F"), ("original", "reversed")):
            values = positive if applicability == "T" else negative
            if dq == "Q":
                gold = "Q"
                attr = atoms(family["ast"])[0]
            elif applicability == "F":
                gold, attr = "SKIP", None
            else:
                gold, attr = source, None
            facts = " ".join(f"{name} is {'present' if value else 'absent'}." for name, value in values.items())
            prompt = ("This is a synthetic composability task. The context and a closed-book record may disagree.\n"
                      f"Rule order shown to the model: {'second rule first' if order == 'reversed' else 'first rule first'}.\n"
                      f"Context fact: source route C. Parametric fact: source route P.\nFacts: {facts}\n"
                      "If the decision is not determined, answer Q; otherwise answer C, P, or SKIP.")
            root = f"{family['family_id']}_block"
            rows.append({"task": "N4", "case_id": f"{root}_{source}{dq}{applicability}_{order}",
                         "family_id": family["family_id"], "underlying_mechanism_id": family["mechanism_id"],
                         "ast_hash": family["ast_hash"], "root_case_id": root, "pair_id": root,
                         "fact_group_id": root, "split": _split(index, N4_SPLITS), "prompt_text": prompt,
                         "gold_answer": gold, "gold_source": source, "gold_DQ": dq, "gold_A": applicability == "T",
                         "rule_order": order, "valid_question_attributes": [attr] if attr else [],
                         "values": values, "rule_ast": family["ast"], "control_target": {"source": source, "DQ": dq, "A": applicability, "order": order},
                         "origin_dataset": "care_m2_n4_generated_v1", "options": {"C": "context", "P": "parametric", "SKIP": "skip", "Q": "ask"}})
    return rows


def prepare(out: str | Path) -> dict[str, Any]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = {"N1": _n1_records(), "N2": _n2_records(), "N3": _n3_records(), "N4": _n4_records()}
    files = {}
    for task, rows in tasks.items():
        inputs = [{key: value for key, value in row.items() if not key.startswith("gold_") and key not in {"values", "rule_ast", "options", "possible_actions", "valid_question_attributes"}} for row in rows]
        labels = [{key: value for key, value in row.items() if key.startswith("gold_") or key in {"values", "rule_ast", "options", "possible_actions", "valid_question_attributes"}} for row in rows]
        splits = [{key: row[key] for key in ("case_id", "family_id", "underlying_mechanism_id", "root_case_id", "pair_id", "fact_group_id", "split")} for row in rows]
        for name, content in (("inputs.jsonl", inputs), ("labels.jsonl", labels), ("splits.jsonl", splits)):
            path = out / task / name
            write_jsonl(path, content)
            files[str(path.relative_to(out))] = None
    manifest = {"schema_version": "care.m2.incremental.data.v1", "generator": "care_m2_inc.generate", "seed": SEED,
                "tasks": {task: {"rows": len(rows), "families": len({r["family_id"] for r in rows}),
                                  "splits": {split: sum(r["split"] == split for r in rows) for split in ("train", "dev", "test")}}
                           for task, rows in tasks.items()},
                "prompt_policy": {"gold_fields_excluded": True, "control_target_scorer_only": True},
                "files": sorted(files)}
    write_json(out / "manifest.json", manifest)
    manifest["file_sha256"] = hash_manifest(out, files)
    write_json(out / "manifest.json", manifest)
    return manifest
