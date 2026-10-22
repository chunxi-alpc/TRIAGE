"""Construct paired, explicitly counterfactual MedGUIDE pathway-rule tasks.

No source text is redistributed: output must remain under a private/ directory.
The MCQA reference path is used ONLY to construct the experimental oracle,
not as a patient fact or a model input. Resulting B/C labels are synthetic
intervention labels, never verified NCCN recommendations.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import random
import re
from pathlib import Path


def norm(value):
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def path_tuple(path):
    if not isinstance(path, (list, tuple)) or len(path) < 3:
        return None
    if not all(isinstance(s, str) and s.strip() for s in path):
        return None
    return tuple(norm(s) for s in path)


def digest(data):
    return hashlib.sha256(json.dumps(data, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def label(n):
    if not 0 <= n < 26:
        raise ValueError("MCQA option count exceeds 26")
    return chr(ord("A") + n)


def render_rules(app_path, sib_path, app_action, sib_action, order):
    source = {"R_app": (app_path, app_action), "R_sib": (sib_path, sib_action)}
    parts, spans = [], {}
    cursor = 0
    for rid in order:
        path, action = source[rid]
        condition = " > ".join(path[:-1])
        prefix = f"{rid}: If the patient's clinical pathway follows {condition}, the next action is "
        line = prefix + action + ".\n"
        spans[rid] = {
            "rule": [cursor, cursor + len(line.rstrip("\n"))],
            "action": [cursor + len(prefix), cursor + len(prefix) + len(action)],
        }
        parts.append(line)
        cursor += len(line)
    result = "".join(parts)
    for rid in order:
        a, b = spans[rid]["action"]
        assert result[a:b] == source[rid][1]
    return result, spans


def make_case(row, paths, index, seed):
    """Return (case, None) or (None, explicit exclusion reason)."""
    if not isinstance(row.get("profile"), str) or not row["profile"].strip():
        return None, "missing_patient_profile"
    source_path = path_tuple(row.get("path"))
    disease = row.get("disease")
    if source_path is None:
        return None, "invalid_gold_path"
    original_paths = paths.get(disease, [])
    valid_paths = [p for p in original_paths if path_tuple(p) is not None]
    if not valid_paths:
        return None, "no_source_tree"
    matches = [p for p in valid_paths if path_tuple(p) == source_path]
    if len(matches) != 1:
        return None, "gold_path_not_unique_in_source"
    active = matches[0]
    # A sibling has the same parent and a different terminal *condition*.
    # Having a shared parent does NOT establish medical mutual exclusivity.
    siblings = [
        p for p in valid_paths
        if len(p) == len(active) and tuple(norm(s) for s in p[:-1]) != tuple(norm(s) for s in active[:-1])
        and tuple(norm(s) for s in p[:-2]) == tuple(norm(s) for s in active[:-2])
        and norm(p[-2]) != norm(active[-2]) and norm(p[-1]) != norm(active[-1])
    ]
    if not siblings:
        return None, "no_distinct_sibling_condition"
    opts = row.get("options")
    if not isinstance(opts, list) or not 3 <= len(opts) <= 26 or any(not isinstance(x, str) or not x.strip() for x in opts):
        return None, "invalid_options"
    gold_letter = str(row.get("answer", "")).strip().upper()
    gold_index = ord(gold_letter) - ord("A") if len(gold_letter) == 1 else -1
    if not 0 <= gold_index < len(opts):
        return None, "invalid_gold_letter"
    original_action = opts[gold_index]
    rng = random.Random(f"{seed}:{disease}:{index}")
    sibling_order = list(siblings)
    rng.shuffle(sibling_order)
    selected = None
    for sibling in sibling_order:
        sibling_action = sibling[-1]
        if norm(sibling_action) == norm(original_action):
            continue
        replacement_candidates = [
            p[-1] for p in valid_paths
            if norm(p[-1]) not in {norm(original_action), norm(sibling_action)}
        ]
        if replacement_candidates:
            replacement_candidates.sort(key=lambda x: norm(x))
            selected = sibling, rng.choice(replacement_candidates)
            break
    if selected is None:
        return None, "no_third_published_action"
    sibling, replacement_action = selected
    sibling_action = sibling[-1]

    # Keep one fixed choice set across A/B/C. Real actions replace distractors.
    # The reference answer stays in its original position; the edited action
    # takes another option letter. This is a controlled new MCQA task.
    remaining = [opts[i] for i in range(len(opts)) if i != gold_index]
    additional = [
        option for option in remaining
        if norm(option) not in {norm(original_action), norm(sibling_action), norm(replacement_action)}
    ]
    pool = [sibling_action, replacement_action] + additional
    if len(pool) < len(opts) - 1:
        return None, "ambiguous_duplicate_options"
    fixed = [None] * len(opts)
    fixed[gold_index] = original_action
    for pos in range(len(fixed)):
        if pos != gold_index:
            fixed[pos] = pool.pop(0)
    if len({norm(x) for x in fixed}) != len(fixed):
        return None, "ambiguous_duplicate_options"
    sibling_letter = label(next(i for i, text in enumerate(fixed) if norm(text) == norm(sibling_action)))
    replacement_letter = label(next(i for i, text in enumerate(fixed) if norm(text) == norm(replacement_action)))

    order = ["R_app", "R_sib"]
    rng.shuffle(order)
    versions = {}
    for variant in "ABC":
        app_action = replacement_action if variant == "B" else original_action
        sib_action = replacement_action if variant == "C" else sibling_action
        guideline, spans = render_rules(active, sibling, app_action, sib_action, order)
        versions[variant] = {
            "guideline": guideline,
            "spans": spans,
            "gold": replacement_letter if variant == "B" else gold_letter,
            "modified_rule": "R_app" if variant == "B" else "R_sib" if variant == "C" else None,
        }
    return {
        "case_id": f"medguide_{index:05d}",
        "family": str(disease),
        "source_path_hash": digest(list(active)),
        "patient": row["profile"],
        "question": "According to the supplied rules, which action applies to this patient's clinical pathway?",
        "options": fixed,
        "original_answer": gold_letter,
        "sibling_answer": sibling_letter,
        "replacement_answer": replacement_letter,
        "rule_order": order,
        "oracle_applicable": "R_app",
        "oracle_inapplicable": "R_sib",
        "versions": versions,
    }, None


def demo_source(n):
    tree = {
        "demo": [
            ["Demo root", "Path alpha: laboratory criterion below the threshold", "Continue plan alpha"],
            ["Demo root", "Path beta: laboratory criterion at or above the threshold", "Continue plan beta"],
            ["Demo root", "Path gamma: explicit third branch", "Continue plan gamma"],
        ]
    }
    rows = [{
        "disease": "demo", "path": tree["demo"][0],
        "profile": "A simulated patient follows path alpha and has a low laboratory value.",
        "answer": "B",
        "options": ["Other plan", "Continue plan alpha", "Alternative plan", "Unspecified plan"],
    } for _ in range(n)]
    return rows, tree


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, help="Private MedGUIDE original parquet")
    parser.add_argument("--paths", type=Path, help="Private source all_paths.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("private") / "cases.jsonl")
    parser.add_argument("--max-cases", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demo", action="store_true", help="Use invented rules; not a MedGUIDE evaluation")
    args = parser.parse_args()
    if args.max_cases <= 0:
        parser.error("--max-cases must be positive")
    if args.demo:
        rows, paths = demo_source(args.max_cases)
    else:
        if args.parquet is None or args.paths is None:
            parser.error("--parquet and --paths are required without --demo")
        import pyarrow.parquet as pq
        rows = pq.read_table(args.parquet).to_pylist()
        paths = json.loads(args.paths.read_text(encoding="utf-8"))
    rng = random.Random(args.seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    counts = Counter()
    cases = []
    for i in indices:
        case, reason = make_case(rows[i], paths, i, args.seed)
        if reason:
            counts[reason] += 1
            continue
        cases.append(case)
        if len(cases) == args.max_cases:
            break
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for case in cases:
            out.write(json.dumps(case, ensure_ascii=False) + "\n")
    stats = {
        "status": "prepared", "synthetic_demo": args.demo, "seed": args.seed,
        "requested": args.max_cases, "constructed": len(cases),
        "scanned": sum(counts.values()) + len(cases),
        "excluded_by_reason": dict(counts),
        "source_dataset_rows": len(rows),
        "method": "strict same-parent sibling and three distinct published actions",
        "source_sha256": {
            "parquet": None if args.demo else hashlib.sha256(args.parquet.read_bytes()).hexdigest(),
            "paths": None if args.demo else hashlib.sha256(args.paths.read_bytes()).hexdigest(),
        },
    }
    write_json(args.output.with_suffix(".stats.json"), stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
