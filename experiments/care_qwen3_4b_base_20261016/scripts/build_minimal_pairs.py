#!/usr/bin/env python3
"""Create controlled *fictional* decision-protocol pairs, not clinical evidence.

Four primary 2x2 states per template plus two same-side numeric controls.
Absolutely not a clinical or parametric-prior benchmark by itself.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

STATES = ("RA", "RN", "IA", "IN", "RA2", "RN2")


def split_groups(n_groups: int, seed: int) -> dict[int, str]:
    groups = list(range(n_groups))
    random.Random(seed).shuffle(groups)
    tr = int(0.6 * n_groups)
    dv = int(0.2 * n_groups)
    return {i: "train" if p < tr else "dev" if p < tr + dv else "test"
            for p, i in enumerate(groups)}


def substring_span(full_text: str, sub: str, start: int = 0) -> list[int]:
    i = full_text.find(sub, start)
    if i < 0:
        raise ValueError(f"span text not present: {sub!r}")
    return [i, i + len(sub)]


def make_cases(n_groups: int, seed: int) -> list[dict]:
    if n_groups < 5:
        raise ValueError("Need at least 5 groups for meaningful grouped splitting")
    rng = random.Random(seed)
    splits = split_groups(n_groups, seed)
    labels = ["index", "score", "marker", "ratio", "measurement", "indicator"]
    rows: list[dict] = []
    for g in range(n_groups):
        label = labels[g % len(labels)]
        var = f"{label.upper()}-{g:03d}"
        threshold = (g % 13 + 4) * 5
        age = 40 + (g % 30)
        irrelevant_question = "What is the patient's age in years?"
        relevant_question = "Which route does the supplied decision protocol select?"
        route_names = ["Route Amber", "Route Birch", "Route Cedar", "Route Dune"]
        shuffled_routes = route_names.copy()
        rng.shuffle(shuffled_routes)
        option_map = dict(zip("ABCD", shuffled_routes))
        letter_for = {v: k for k, v in option_map.items()}
        option_str = "; ".join(f"{letter}={name}" for letter,name in option_map.items())
        guideline = (f"Synthetic protocol only: if {var} is below {threshold}, "
                     "select Route Cedar; otherwise select Route Dune.")
        original_guideline = (f"Synthetic original protocol: if {var} is below {threshold}, "
                              "select Route Amber; otherwise select Route Birch.")
        # Use the same apparent scale on both sides; all tokens are fictional labels.
        app_val, non_val = threshold - 2, threshold + 2
        same_app, same_non = threshold - 1, threshold + 1
        for state in STATES:
            applicable = state in ("RA", "IA", "RA2")
            relevant = state in ("RA", "RN", "RA2", "RN2")
            val = (same_app if state == "RA2" else same_non if state == "RN2"
                   else app_val if applicable else non_val)
            question = relevant_question if relevant else irrelevant_question
            patient = f"Patient: {var} = {val} units; age = {age} years."
            prefix = f"Question: {question}\n{patient}\nGuideline: "
            # Decision candidate text is intentionally *outside* the edited span.
            suffix = f"\nOptions: {option_str}.\nAnswer (A/B/C/D):"
            prompt = prefix + guideline + suffix
            original_prompt = prefix + original_guideline + suffix
            closedbook_prompt = (f"Question: {relevant_question}\n{patient}"
                                 f"\nOptions: {option_str}."
                                 "\nAnswer (A/B/C/D):")
            start = prompt.index(guideline)
            action = "select Route Cedar"
            evidence = f"{var} = {val} units"
            row = {
                "case_id": f"synthetic-{g:03d}-{state}",
                "group_id": f"synthetic-{g:03d}",
                "split": splits[g],
                "state": state,
                "source_type": "synthetic_logic_only",
                "validated_clinical_conflict": False,
                "prior_eligible": False,
                "variable": var,
                "threshold": threshold,
                "patient_value": val,
                "patient_age": age,
                "applicable": applicable,
                "question_relevant": relevant,
                "gold_answer": letter_for["Route Cedar"] if applicable else letter_for["Route Dune"],
                "original_action": letter_for["Route Amber"] if applicable else letter_for["Route Birch"],
                "inapplicable_action": letter_for["Route Cedar"] if not applicable else letter_for["Route Dune"],
                "option_map": option_map,
                "modified_rule": guideline,
                "original_rule": original_guideline,
                "query": question,
                "patient": patient,
                "prompt": prompt,
                "original_prompt": original_prompt,
                "closedbook_prompt": closedbook_prompt,
                "spans": {
                    "rule": [start, start + len(guideline)],
                    "action": substring_span(prompt, action, start),
                    "patient_evidence": substring_span(prompt, evidence),
                },
                "note": "FICTIONAL branch logic fixture, not a clinical guideline or prior-conflict claim."
            }
            assert prompt[slice(*row["spans"]["rule"])] == guideline
            assert prompt[slice(*row["spans"]["action"])] == action
            assert prompt[slice(*row["spans"]["patient_evidence"])] == evidence
            rows.append(row)
    rng.shuffle(rows)
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-groups", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("data/derived/pilot.jsonl"))
    args = p.parse_args()
    cases = make_cases(args.n_groups, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in cases:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts = {s: sum(r["split"] == s for r in cases) for s in ("train", "dev", "test")}
    manifest = {
        "dataset_type": "FICTIONAL_LOGIC_SANITY_ONLY",
        "assertion": "DO NOT REPORT AS CLINICAL OR PARAMETRIC-PRIOR EVIDENCE",
        "seed": args.seed,
        "groups": args.n_groups,
        "rows": len(cases),
        "rows_by_split": counts,
        "states": list(STATES),
    }
    args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(cases)} rows from {args.n_groups} distinct rule groups to {args.out}")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
