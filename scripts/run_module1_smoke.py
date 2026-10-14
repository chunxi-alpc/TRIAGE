#!/usr/bin/env python3
"""Real MEGA-CDP train_val smoke inputs; no reference labels or patient paths."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REQUIRED = {"candidate_rules", "prior_decision", "conflict_scores", "relevance_scores",
    "condition_evidence_relations", "applicability", "critical_spans",
    "relation_weights", "binding_gate", "action_gate"}
SAMPLES = [
    (3, "For this patient, what antiplatelet management is appropriate when clopidogrel is being considered?", "official_structured_tree"),
    (9, "For this patient, should management begin with surgical excision or neoadjuvant imatinib, and under which conditions?", "text_clauses_from_official_tree"),
]

def expand_tree(tree, guideline_id):
    """Expand ALL original guideline branches independently of patient or labels."""
    rules = []
    def walk(node, conditions):
        if node.get("leaf") or "conclusion" in node:
            action = node["conclusion"]
            text = "Apply this recommendation only when ALL requirements hold: " + " AND ".join(
                ("The answer must be NO: " if c["negated"] else "The answer must be YES: ") + c["text"]
                for c in conditions) + ". Recommendation: " + action
            rules.append({"rule_id": f"mega_{guideline_id}_leaf_{node['index']}", "text": text,
                "conditions": conditions, "logic": "AND", "exceptions": [],
                "action": action, "source_node_index": node["index"]})
            return
        for branch in ("yes", "no"):
            if branch in node:
                condition = {"condition_id": f"mega_{guideline_id}_node_{node['index']}_{branch}",
                    "text": node["description"], "negated": branch == "no"}
                walk(node[branch], conditions + [condition])
    walk(tree, [])
    return rules

def load_inputs():
    config = json.loads((ROOT / "configs/datasets.json").read_text())
    dataset_root = Path(config["MEGA-CDP"]["path"])
    case_file = dataset_root / "train_val/vignette_cases.jsonl"
    tree_file = dataset_root / "train_val/decision_trees.jsonl"
    wanted = {s[0] for s in SAMPLES}
    cases, trees = {}, {}
    with case_file.open() as handle:
        for line_number, line in enumerate(handle, 1):
            record = json.loads(line)
            gid = record["guideline_id"]
            if gid in wanted and gid not in cases:
                cases[gid] = (record["text"], record["filename"], line_number)
            if len(cases) == len(wanted):
                break
    with tree_file.open() as handle:
        for line_number, line in enumerate(handle, 1):
            record = json.loads(line)
            gid = record["guideline_id"]
            if gid in wanted:
                trees[gid] = (json.loads(record["text"])["decision_tree"], line_number)
            if len(trees) == len(wanted):
                break
    inputs = []
    for gid, question, kind in SAMPLES:
        patient, filename, case_line = cases[gid]
        tree, tree_line = trees[gid]
        rules = expand_tree(tree, gid)
        guideline = {"rules": rules} if kind == "official_structured_tree" else [r["text"] for r in rules]
        inputs.append({"guideline": guideline, "patient": patient, "question": question,
            "input_source": {"dataset": "MEGA-CDP", "split": "train_val", "guideline_id": gid, "filename": filename,
                "patient": {"path": str(case_file), "line": case_line, "field": "text"},
                "guideline": {"path": str(tree_file), "line": tree_line, "field": "text.decision_tree",
                    "representation": kind, "all_tree_paths_used": True, "rule_count": len(rules)},
                "original_guideline_markdown": str(dataset_root / "train_val/guideline_md" / filename / (filename + ".md")),
                "question": {"origin": "generic management question written for this smoke run; not a reference answer"}}})
    return inputs

def gpu_memory():
    import torch
    rows = {str(i): {"name": torch.cuda.get_device_name(i),
        "allocated_gib": torch.cuda.memory_allocated(i) / 1024**3,
        "reserved_gib": torch.cuda.memory_reserved(i) / 1024**3,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(i) / 1024**3}
        for i in range(torch.cuda.device_count())}
    try:
        snapshot = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
            "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout.strip().splitlines()
    except (OSError, subprocess.CalledProcessError):
        snapshot = []
    return {"torch_devices": rows, "nvidia_smi_memory_mib": snapshot}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/module1_smoke.json")
    parser.add_argument("--prepare-only", action="store_true", help="Inspect input metadata without loading models")
    args = parser.parse_args()
    inputs = load_inputs()
    if args.prepare_only:
        print(json.dumps({"samples": [{"input_source": x["input_source"],
            "patient": x["patient"], "question": x["question"]} for x in inputs]}, ensure_ascii=False, indent=2))
        return
    from src.module1 import Module1
    import torch
    torch.cuda.init()
    module = Module1(top_k=3)
    started = time.time()
    output = {"run_kind": "module1_minimal_real_data_smoke", "dataset": "MEGA-CDP", "split": "train_val",
        "reference_answers_used": False, "single_module_instance": True, "samples": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for item in inputs:
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)
        sample_start = time.time()
        result = module.run(guideline=item["guideline"], patient=item["patient"], question=item["question"])
        missing = REQUIRED - set(result)
        if missing:
            raise AssertionError(f"Module1 output missing keys: {sorted(missing)}")
        json.dumps(result, ensure_ascii=False, allow_nan=False)
        output["samples"].append({"input_source": item["input_source"], "guideline": item["guideline"],
            "patient": item["patient"], "question": item["question"], "result": result,
            "elapsed_seconds": round(time.time() - sample_start, 3),
            "gpu_memory": gpu_memory(), "structured_json_verified": True})
        output["elapsed_seconds"] = round(time.time() - started, 3)
        args.output.write_text(json.dumps(output, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
        print(f"Completed guideline {item['input_source']['guideline_id']} in {output['samples'][-1]['elapsed_seconds']} seconds", flush=True)
    print(f"Saved {args.output}", flush=True)

if __name__ == "__main__":
    main()

