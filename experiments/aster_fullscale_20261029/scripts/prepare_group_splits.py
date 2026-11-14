#!/usr/bin/env python3
"""Deterministic guideline-level splits. Clinical text is never copied to manifests."""
from __future__ import annotations
import argparse, json, random, math
from collections import Counter
from pathlib import Path


def field(row, key):
    value = row
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"missing group key {key!r}")
        value = value[part]
    if value is None or not str(value).strip():
        raise ValueError(f"empty group key {key!r}")
    return str(value)


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as stream:
        for n, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{n}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{n}: expected JSON object")
            rows.append(row)
    return rows



def allocate(groups, fractions, seed):
    if len(set(groups)) != len(groups):
        raise ValueError("groups must be unique")
    if len(groups) < len(fractions):
        raise ValueError("not enough groups for requested splits")
    if not fractions or any(not math.isfinite(v) or v <= 0 for v in fractions.values()) or abs(sum(fractions.values()) - 1) > 1e-8:
        raise ValueError("split fractions must be positive and sum to 1")
    names = list(fractions)
    shuffled = sorted(groups)
    random.Random(seed).shuffle(shuffled)
    raw = {n: fractions[n] * len(groups) for n in names}
    counts = {n: int(raw[n]) for n in names}
    left = len(groups) - sum(counts.values())
    for n in sorted(names, key=lambda x: (raw[x] - counts[x], -names.index(x)), reverse=True)[:left]:
        counts[n] += 1
    for empty in [n for n in names if counts[n] == 0]:
        donor = max(names, key=lambda n: counts[n])
        if counts[donor] < 2:
            raise ValueError("cannot ensure every split has a group")
        counts[donor] -= 1
        counts[empty] += 1
    result, pos = {}, 0
    for name in names:
        for group in shuffled[pos:pos + counts[name]]:
            result[group] = name
        pos += counts[name]
    return result


def parse_fractions(text):
    result = {}
    for item in text.split(","):
        name, sep, val = item.partition("=")
        if not sep or not name.strip() or name.strip() in result:
            raise argparse.ArgumentTypeError("use unique names like fit=0.7,select=0.15,confirm=0.15")
        try:
            result[name.strip()] = float(val)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("fractions must be numeric") from exc
    if any(not math.isfinite(v) or v <= 0 or v >= 1 for v in result.values()) or abs(sum(result.values()) - 1) > 1e-8:
        raise argparse.ArgumentTypeError("fractions must be between 0 and 1 and sum to 1")
    return result


def create(args):
    rows = read_jsonl(args.input)
    if not rows:
        raise ValueError("input has no records")
    groups_per_row = [field(row, args.group_key) for row in rows]
    groups = sorted(set(groups_per_row))
    mapping = allocate(groups, args.fractions, args.seed)
    payload = {
        "schema_version": "aster_group_split.v1",
        "dataset": args.dataset,
        "source_file": Path(args.input).name,
        "group_key": args.group_key,
        "seed": args.seed,
        "fractions": args.fractions,
        "source_records": len(rows),
        "source_groups": len(groups),
        "records_by_split": dict(Counter(mapping[g] for g in groups_per_row)),
        "groups_by_split": dict(Counter(mapping.values())),
        "group_to_split": mapping,
        "note": "Split by source guideline group, not by case. Review cross-dataset source overlap before testing."
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: payload[k] for k in ("dataset", "source_records", "source_groups", "records_by_split", "groups_by_split")}, ensure_ascii=False))


def apply(args):
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    rows = read_jsonl(args.input)
    mapping = manifest["group_to_split"]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    counts, unknown, handles = Counter(), [], {}
    try:
        for row in rows:
            group = field(row, args.group_key)
            split = mapping.get(group)
            if split is None:
                unknown.append(group)
                continue
            if split not in handles:
                handles[split] = (output / f"{split}.jsonl").open("w", encoding="utf-8")
            handles[split].write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[split] += 1
    finally:
        for handle in handles.values():
            handle.close()
    report = {
        "schema_version": "aster_group_split_apply.v1",
        "input_file": Path(args.input).name,
        "group_key": args.group_key,
        "input_records": len(rows),
        "assigned_records": sum(counts.values()),
        "records_by_split": dict(counts),
        "unknown_group_records": len(unknown),
        "unknown_groups": sorted(set(unknown)),
        "status": "complete" if not unknown else "blocked_unknown_groups"
    }
    (output / "apply_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if unknown:
        raise SystemExit(f"blocked: {len(unknown)} records have groups absent from manifest; see apply_report.json")
    print(json.dumps(report, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("create", help="create a group split manifest from a JSONL file")
    p.add_argument("--input", required=True)
    p.add_argument("--group-key", required=True, help="verified field name, supports dotted nested keys")
    p.add_argument("--dataset", required=True)
    p.add_argument("--fractions", type=parse_fractions, required=True)
    p.add_argument("--seed", type=int, default=20261029)
    p.add_argument("--output", required=True)
    p.set_defaults(func=create)
    p = sub.add_parser("apply", help="apply an existing manifest to a related JSONL file")
    p.add_argument("--input", required=True)
    p.add_argument("--group-key", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=apply)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
