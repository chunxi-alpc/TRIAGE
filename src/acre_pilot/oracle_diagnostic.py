"""Frozen-source helpers for the explicitly non-deployable Oracle diagnostic."""
import csv, gzip, hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/acre_pilot/round2"
OLD = BASE / "revision3"
OUT = ROOT / "results/acre_pilot/oracle_switch_diagnostic"
REMOTE_BASE = "c25da4420d467e281dfe5243a15ed8315b8a8c90"
SEED = 20261015

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def jsonl(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(data)
    tmp.replace(path)

def csvwrite(path, rows):
    with Path(path).open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k,v in row.items()})

def archive(path, rows):
    # Fixed gzip metadata makes the published record reproducible.
    with Path(path).open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as z:
            for row in rows:
                z.write((json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n").encode())

def normal_inputs():
    """No Oracle/reference reads in this function."""
    m = read(BASE / "development_manifest.json")
    items = m["samples"]
    cases = jsonl(OLD / "case_results.jsonl.gz")
    annotations = jsonl(OLD / "annotations.jsonl.gz")
    assert len(items) == len(cases) == 80
    assert len({a["cache_key"] for a in annotations}) == len(annotations)
    assert {c["annotation_key"] for c in cases} <= {a["cache_key"] for a in annotations}
    assert len({c["case_id"] for c in cases}) == 80
    assert {i["case_id"] for i in items} == {c["case_id"] for c in cases}
    return items, {c["case_id"]:c for c in cases}, {a["cache_key"]:a for a in annotations}

def oracle_inputs():
    """Explicit diagnostic/evaluation boundary; never called by normal inference."""
    return read(BASE / "development_oracle.json"), read(BASE / "development_references.json")

def conflict_from_predictions(mapped, options):
    applicable = [r for r in mapped if r["relevant"] and r["applicability"] == "applicable"]
    actions = {s["text"] for r in applicable for s in r["mapped"]["action_spans"]
               if any(s["text"] == o["action"] for o in options[:-1])}
    return len(applicable)>1 and (len(actions)!=1 or any(not r["mapped"]["action_spans"] for r in applicable))

def mask_kind(a,b,conflict=False,errors=False):
    if conflict: return "conflict_disabled"
    if not a and not b: return "mapping_evidence_or_other_no_enhancement"
    if a == b: return "identical"
    if not a or not b: return "one_empty"
    return "both_nonempty_different"

def overlap(a,b):
    a,b=set(a),set(b)
    return {"fixed_count":len(a),"switch_count":len(b),"intersection_count":len(a&b),
            "union_count":len(a|b),"symmetric_difference_count":len(a^b),
            "jaccard":len(a&b)/len(a|b) if a|b else None,
            "removed":sorted(a-b),"added":sorted(b-a)}

def span_roles(mapped, positions):
    positions=set(positions); out=[]
    for row in mapped:
        for field,spans in row["mapped"].items():
            for span in spans:
                hits=positions & set(span["tokens"])
                if hits:
                    out.append({"rule_id":row["rule_id"],"state":row["applicability"],
                                "relevant":row["relevant"],"role":field,"text":span["text"],
                                "source_start":span["source_start"],"source_end":span["source_end"],
                                "tokens":sorted(hits)})
    return out

def code_fingerprints():
    paths = list((ROOT/"src/acre_pilot").glob("*.py"))
    paths += list((ROOT/"scripts").glob("*acre*factorial*.py"))
    paths += [ROOT/"scripts/audit_acre_switch_masks.py", ROOT/"configs/acre_oracle_switch.json"]
    return {str(p.relative_to(ROOT)):sha(p) for p in paths if p.exists()}
