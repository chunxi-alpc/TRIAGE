"""Shared data and exact source-span operations for the automatic pilot."""
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "config.json").read_text())
DATA = ROOT / "data"
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"
for folder in (DATA, RESULTS, LOGS):
    folder.mkdir(exist_ok=True)

def read_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)

def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            append_row(f, row)

def append_row(handle, row):
    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    handle.flush()

def blocks_for(text):
    blocks = []
    for match in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", text, re.S):
        a, b = match.span()
        blocks.append(dict(block_id=len(blocks), start=a, end=b,
                           line_start=text.count("\n", 0, a) + 1,
                           line_end=text.count("\n", 0, b) + 1,
                           text=text[a:b]))
    return blocks

def compact(text):
    return re.sub(r"[\s$\\{}_*]", "", text).lower()

def locate_quote(block, quote):
    """Require the literal source sentence; no fabricated restoration."""
    if not quote:
        raise ValueError("Empty source quote")
    start = block["text"].find(quote)
    if start < 0:
        raise ValueError("Quote is not verbatim in its source block")
    return start, start + len(quote)

def render(blocks, deleted=(), cuts=(), insertions=None, replacements=()):
    """Retain source order and every character outside specified edits."""
    by_id = {b["block_id"]: b for b in blocks}
    changes = {}
    for item in cuts:
        a, b = locate_quote(by_id[item["block_id"]], item["quote"])
        changes.setdefault(item["block_id"], []).append((a, b, ""))
    for item in replacements:
        a, b = locate_quote(by_id[item["block_id"]], item["quote"])
        changes.setdefault(item["block_id"], []).append((a, b, item["replacement"]))
    output = []
    for block in blocks:
        bid = block["block_id"]
        if bid in deleted:
            value = (insertions or {}).get(bid, "")
        else:
            value = block["text"]
            edits = sorted(changes.get(bid, []))
            if any(left[1] > right[0] for left, right in zip(edits, edits[1:])):
                raise ValueError("Overlapping source edits")
            for a, b, replacement in reversed(edits):
                value = value[:a] + replacement + value[b:]
        # The same block slots and separators are present in all paired views.
        output.append(value)
    return "\n\n".join(output)

def threshold_truth(value, comparator, threshold):
    operations = {"<": lambda a,b:a<b, "<=":lambda a,b:a<=b,
                  ">":lambda a,b:a>b, ">=":lambda a,b:a>=b}
    if comparator not in operations:
        raise ValueError("Unsupported numerical comparator")
    return int(operations[comparator](float(value), float(threshold)))

def sigmoid(x):
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    e = math.exp(x)
    return e / (1 + e)

def fraction(numerator, denominator):
    return dict(numerator=int(numerator), denominator=int(denominator),
                rate=numerator / denominator if denominator else None)
