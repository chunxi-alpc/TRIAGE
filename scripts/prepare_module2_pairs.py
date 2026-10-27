#!/usr/bin/env python3
"""Prepare offline module-two triplets without loading a language model."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.module2.minimal_pairs import build_minimal_pairs, load_development_trees


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trees", required=True, type=Path, help="Original train_val/decision_trees.jsonl")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-split", default="train_val")
    parser.add_argument("--seed", type=int, default=261027)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--max-pairs-per-tree", type=int, default=1)
    parser.add_argument("--max-trees-per-group", type=int, default=1)
    parser.add_argument("--max-depth", type=int, default=6)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output must be absent or empty; completed preparations are immutable")
    records = load_development_trees(args.trees, args.source_split)
    rows, audit = build_minimal_pairs(records, source_split=args.source_split, seed=args.seed,
        max_groups=args.max_groups, max_pairs_per_tree=args.max_pairs_per_tree,
        max_trees_per_group=args.max_trees_per_group, max_depth=args.max_depth)
    audit["source_file"] = str(args.trees.resolve())
    audit["source_file_sha256"] = hashlib.sha256(args.trees.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    pairs_path = args.output / "triplets.jsonl"
    with pairs_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    audit["triplets_sha256"] = hashlib.sha256(pairs_path.read_bytes()).hexdigest()
    (args.output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: audit[key] for key in ("status", "triplets", "source_guidelines", "sibling_groups", "partitions")}, indent=2))


if __name__ == "__main__":
    main()
