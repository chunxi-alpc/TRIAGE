#!/usr/bin/env python3
"""Build leakage-aware, label-blinded ConflictMedQA trials (stdlib only).

This is a data/protocol builder, NOT a parametric-knowledge probe or CARE intervention.
Historical reference snapshot: Feb 2025 (not a current clinical recommendation).
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path

SEED = 20261019
ADVICE_COUNT = 194
SCENARIO_COUNT = 4290
PAIR_OFFSET = 2145
TRACKS = ("advice_choice", "scenario_judgment")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def stable_unit(key, seed=SEED):
    return int(digest(f"{seed}:{key}".encode())[:12], 16) / 16**12


def split_for(group, seed=SEED):
    p = stable_unit(group, seed)
    return "train" if p < .70 else "dev" if p < .85 else "test"


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number}: object expected")
                yield row


def make_id(*parts):
    return digest("|".join(map(str, parts)).encode())[:20]


def validate_advice(rows, strict=True):
    if strict and len(rows) != ADVICE_COUNT:
        raise ValueError(f"Expected {ADVICE_COUNT} advice pairs; got {len(rows)}")
    seen = set()
    for row in rows:
        if not {"id", "correct", "wrong", "disease"}.issubset(row):
            raise ValueError("Advice row missing id/correct/wrong/disease")
        rid = str(row["id"])
        if rid in seen:
            raise ValueError(f"Duplicate advice id {rid}")
        seen.add(rid)
        if not all(isinstance(row[k], str) and row[k].strip() for k in ("correct", "wrong", "disease")):
            raise ValueError(f"Advice row {rid}: empty text or disease")
        if row["correct"].strip() == row["wrong"].strip():
            raise ValueError(f"Advice row {rid}: variants identical")
    return {"total_pairs": len(rows), "disease_counts": dict(collections.Counter(r["disease"] for r in rows))}


def validate_scenarios(rows, strict=True):
    if strict and len(rows) != SCENARIO_COUNT:
        raise ValueError(f"Expected {SCENARIO_COUNT} scenarios; got {len(rows)}")
    by_id = {}
    for row in rows:
        required = ("id", "text", "label", "disease")
        if not all(k in row for k in required):
            raise ValueError("Scenario missing id/text/label/disease")
        rid = int(row["id"])
        if rid in by_id or rid < 0:
            raise ValueError(f"Duplicate/bad scenario id {rid}")
        if not isinstance(row["text"], str) or not row["text"].strip():
            raise ValueError(f"Empty scenario {rid}")
        if row["label"] not in {"correct", "wrong"}:
            raise ValueError(f"Unexpected label for scenario {rid}")
        by_id[rid] = row
    if strict:
        if set(by_id) != set(range(SCENARIO_COUNT)):
            raise ValueError("Official scenario IDs must be 0..4289")
        for i in range(PAIR_OFFSET):
            a, b = by_id[i], by_id[i + PAIR_OFFSET]
            if a["label"] != "correct" or b["label"] != "wrong":
                raise ValueError(f"Invalid label pair {i}")
            if a["disease"].casefold() != b["disease"].casefold():
                raise ValueError(f"Disease differs within pair {i}")
    return {"total_rows": len(rows), "label_counts": dict(collections.Counter(r["label"] for r in rows)),
            "official_pairing_verified": strict}


def construct(advice, scenarios=(), seed=SEED):
    inputs, labels = [], []

    def put(group, track, variant, source_kind, options, context, correct_choice, meta=None):
        item_id = make_id(group, track, variant, source_kind)
        split = split_for(group, seed)
        text = (
            "Research-only historical benchmark: judge agreement with the February 2025 clinical-"
            "guideline reference snapshot used by the dataset, NOT with today's guidelines. "
            "An optional retrieved note is of unknown reliability; never assume it is correct.\n\n"
        )
        if track == "advice_choice":
            text += "Question: Which of these mutually conflicting recommendations was marked consistent "
            text += "with that historical snapshot?\n"
        else:
            text += "Question: Is the advice in the following scenario consistent with that historical snapshot?\n"
        text += f"Disease area: {meta['disease']}\n"
        if track == "scenario_judgment":
            text += f"Scenario: {meta['scenario']}\n"
        if context is not None:
            text += f"Retrieved excerpt (source and date unverified): {context}\n"
        text += "\nChoices:\n" + "\n".join(f"{k}. {v}" for k, v in options.items())
        text += "\nReply with exactly one choice letter, A or B.\nAnswer:"
        inp = {"id": item_id, "group": group, "split": split, "track": track,
               "variant": variant, "source_kind": source_kind,
               "retrieved_option": source_kind[-1] if source_kind in {"excerpt_A", "excerpt_B"} else None,
               "prompt": text, "options": options,
               "provenance": {"dataset": "RDBH/DriftMed", "snapshot": "2025-02", "source_id": meta["source_id"]}}
        inputs.append(inp)
        labels.append({"id": item_id, "group": group, "split": split, "track": track,
                       "correct_choice": correct_choice, "source_kind": source_kind,
                       "disease": meta["disease"], "source_id": meta["source_id"]})

    for row in advice:
        group = f"adv:{row['id']}"
        for reversed_order in (False, True):
            options = {"A": row["wrong"] if reversed_order else row["correct"],
                       "B": row["correct"] if reversed_order else row["wrong"]}
            gold = "B" if reversed_order else "A"
            variant = "BA" if reversed_order else "AB"
            for source in ("none", "excerpt_A", "excerpt_B"):
                context = None if source == "none" else options[source[-1]]
                put(group, "advice_choice", variant, source, options, context, gold,
                    {"disease": row["disease"], "source_id": row["id"]})

    for row in scenarios:
        rid = int(row["id"])
        group = f"sc:{rid % PAIR_OFFSET}"
        reversed_order = stable_unit(group + ":position", seed) >= .5
        options = {"A": "Inconsistent", "B": "Consistent"} if reversed_order else {"A": "Consistent", "B": "Inconsistent"}
        correct_choice = ("B" if reversed_order else "A") if row["label"] == "correct" else ("A" if reversed_order else "B")
        put(group, "scenario_judgment", str(rid), "none", options, None, correct_choice,
            {"disease": row["disease"], "source_id": rid, "scenario": row["text"]})
    if len({r["id"] for r in inputs}) != len(inputs):
        raise AssertionError("Case id collision")
    lookup = {r["id"]: r for r in labels}
    for r in inputs:
        if lookup[r["id"]]["correct_choice"] not in r["options"]:
            raise AssertionError("Invalid gold choice")
    groups = collections.defaultdict(set)
    for r in inputs:
        groups[r["group"]].add(r["split"])
    assert all(len(splits) == 1 for splits in groups.values())
    return inputs, labels


def jsonl_bytes(rows):
    return ("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)).encode("utf-8")


def prepare(advice_file, scenario_file, out, seed=SEED, allow_partial=False):
    advice = list(read_jsonl(advice_file))
    scenarios = list(read_jsonl(scenario_file)) if scenario_file else []
    audit = {"advice": validate_advice(advice, strict=not allow_partial)}
    if scenarios:
        audit["scenarios"] = validate_scenarios(scenarios, strict=not allow_partial)
    elif not allow_partial:
        raise ValueError("Official run requires --scenarios for clinical vignette track")
    inputs, labels = construct(advice, scenarios, seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    ib, lb = jsonl_bytes(inputs), jsonl_bytes(labels)
    (out / "inputs.jsonl").write_bytes(ib)
    (out / "labels.jsonl").write_bytes(lb)
    manifest = {"status": "DATA_PREPARED_NOT_MODEL_EVALUATED", "source_advice": str(advice_file),
                "source_scenarios": str(scenario_file) if scenario_file else None,
                "source_sha256": {"advice": digest(Path(advice_file).read_bytes()),
                                  "scenarios": digest(Path(scenario_file).read_bytes()) if scenario_file else None},
                "seed": seed, "protocol": "historical_2025_02", "allow_partial_fixture": allow_partial,
                "input_sha256": digest(ib), "labels_sha256": digest(lb),
                "case_counts": dict(collections.Counter(r["track"] for r in inputs)),
                "split_counts": dict(collections.Counter(r["split"] for r in inputs)),
                "group_counts": dict(collections.Counter(split_for(g, seed) for g in {r["group"] for r in inputs})),
                "checks": audit,
                "limitations": ["Neither a correct label nor baseline success alone proves stored parametric knowledge.",
                                "Advised choice track evaluates conflicting recommendations, not patient-specific applicability.",
                                "Guideline overlap may exist between distinct scenario IDs and across advice/scenario tracks.",
                                "Historical 2025-02 labels are not automatically valid for 2026 medical decisions.",
                                "Conditional action labels require separate verified rule/patient annotations."]}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--advice", type=Path, required=True, help="advice_pairs.jsonl from RDBH/DriftMed")
    p.add_argument("--scenarios", type=Path, help="ConflictMed.jsonl from RDBH/DriftMed")
    p.add_argument("--out", type=Path, required=True, help="New, nonexistent output folder")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--allow-partial", action="store_true", help="For synthetic infrastructure tests only")
    a = p.parse_args()
    print(json.dumps(prepare(a.advice, a.scenarios, a.out, a.seed, a.allow_partial), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()