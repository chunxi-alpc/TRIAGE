"""python -m src.module1 --input input.json --output results/module1.json"""
import argparse
import json
from pathlib import Path
from . import Module1

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--top-k", type=int, default=12)
args = parser.parse_args()
sample = json.loads(args.input.read_text())
result = Module1(top_k=args.top_k).run(guideline=sample["guideline"],
    patient=sample["patient"], question=sample["question"])
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
