#!/usr/bin/env python3
"""Run the mainline CARE A1 signal source on one JSON input (V6 CLI remains legacy)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.module1 import create_module1_a1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True,
                    help="JSON with patient, question and rules (or guideline.rules)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--base-model", default="/home/gcx/models/care_module1_v5/phi-4")
    ap.add_argument("--adapter-model", default="/home/gcx/models/care_module1_v5/trialmatchai-phi4-reasoning-lora")
    ap.add_argument("--cache-dir")
    args = ap.parse_args(argv)
    item = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(item, dict):
        ap.error("Expected JSON object")
    if args.output.exists():
        ap.error("Output exists; avoid overwriting an earlier result")
    try:
        module1 = create_module1_a1(device=args.device, base_model=args.base_model,
                                   adapter_model=args.adapter_model, cache_dir=args.cache_dir)
        result = module1.run(
            guideline=item.get("rules", item.get("guideline")),
            patient=item["patient"], question=item["question"])
    except (KeyError, ValueError) as err:
        ap.error(str(err))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(f"Saved A1 signal: {args.output}", flush=True)


if __name__ == "__main__":
    main()
