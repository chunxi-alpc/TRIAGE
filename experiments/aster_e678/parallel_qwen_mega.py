"""Resume only unfinished Qwen Base official MEGA cases on five independent GPUs."""
import argparse
import fcntl
import json
import os
import time
from pathlib import Path
from .prepare import OUT
from . import official_mega as official

ROOT = OUT / "models/qwen_base"
OUTPUT = ROOT / "evaluation/mega_official"
PLAN_DIR = OUTPUT / "parallel_resume_20261114"
PLAN_FILE = PLAN_DIR / "plan.json"
PARTS = 5

def partition(cases, predictions, count=PARTS):
    completed = {}
    for row in predictions:
        completed.setdefault(row["case_id"], set()).add(row["method"])
    remaining = [case for case in cases
                 if not set(official.METHODS) <= completed.get(case["case_id"], set())]
    return remaining, [remaining[index::count] for index in range(count)]

def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

def prepare():
    if PLAN_FILE.exists():
        return official.read(PLAN_FILE)
    if (ROOT / "completion.json").exists():
        raise RuntimeError("Qwen Base already completed; no parallel job is needed")
    original = OUTPUT / "shard0"
    cases = official.rows(official.OFFICIAL_CASES)
    predictions = official.rows(original / "path_predictions.jsonl")
    remaining, parts = partition(cases, predictions)
    if not remaining:
        raise RuntimeError("All cases already have results; summarize existing outputs only")
    nodes = official.rows(original / "nodes.jsonl")
    naturals = official.rows(original / "natural_nodes.jsonl")
    PLAN_DIR.mkdir(exist_ok=True)
    remaining_path = PLAN_DIR / "remaining_cases.jsonl"
    write_rows(remaining_path, remaining)
    assignments = []
    for index, items in enumerate(parts):
        work = OUTPUT / ("shard_resume" + str(index))
        work.mkdir(exist_ok=True)
        (work / "features").mkdir(exist_ok=True)
        ids = {case["case_id"] for case in items}
        reused = {}
        for name, records in (("nodes.jsonl", nodes), ("natural_nodes.jsonl", naturals),
                              ("path_predictions.jsonl", predictions)):
            selected = [row for row in records if row["case_id"] in ids]
            write_rows(work / name, selected)
            reused[name] = len(selected)
        assignments.append(dict(part=index, work=str(work), cases=len(items),
                                case_ids=[case["case_id"] for case in items],
                                copied_partial_records=reused))
    result = dict(model="qwen_base", original_cases_file=str(official.OFFICIAL_CASES),
                  remaining_cases_file=str(remaining_path), original_work=str(original),
                  total_cases=len(cases), completed_cases_reused=len(cases)-len(remaining),
                  remaining_cases=len(remaining), parts=assignments,
                  allowed_gpus=[0,1,4,5,6], parameter_directory=str(ROOT / "models/E1/full"),
                  protocol="E1-balanced-2540; existing frozen parameters",
                  training=False, changes="partition only; same prompts, scoring and decoding",
                  estimated_peak_gib=[20,28], estimated_remaining_minutes=[60,90],
                  prepared_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    official.dump(PLAN_FILE, result)
    return result

def run_part(index):
    plan = official.read(PLAN_FILE)
    entry = plan["parts"][index]
    work = Path(entry["work"])
    marker = work / "done.json"
    if marker.exists() and official.read(marker).get("state") == "complete":
        return
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if gpu not in ("0","1","4","5","6"):
        raise RuntimeError("One authorized physical GPU must be assigned")
    import torch
    torch.set_num_threads(4)
    from .engine import Engine
    with (work / ".execution.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        official.dump(work / "process.json", dict(pid=os.getpid(), physical_gpu=int(gpu),
                      part=index, started_at=time.strftime("%Y-%m-%d %H:%M:%S")))
        engine = Engine("qwen_base", metadata_path=work / "model_metadata.json")
        # The original scorer writes the shard outputs. Full-population aggregation
        # runs once on the CPU after every part finishes.
        old_summary = official.summarize
        official.summarize = lambda *args, **kwargs: {"state":"aggregation_pending"}
        try:
            official._run("qwen_base", ROOT, work, index, PARTS, engine,
                          Path(plan["remaining_cases_file"]))
        finally:
            official.summarize = old_summary

def finalize():
    plan = official.read(PLAN_FILE)
    for part in plan["parts"]:
        done = Path(part["work"]) / "done.json"
        if not done.exists() or official.read(done).get("state") != "complete":
            raise RuntimeError("A parallel MEGA part is unfinished")
    with (ROOT / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = official.summarize(ROOT, official.OFFICIAL_CASES)
        if result["state"] != "complete":
            raise RuntimeError("Full official population is incomplete after combining existing and new outputs")
        expected = {(case["case_id"], method) for case in official.rows(official.OFFICIAL_CASES)
                    for method in official.METHODS}
        actual = {(row["case_id"], row["method"]) for work in OUTPUT.glob("shard*")
                  for row in official.rows(work / "path_predictions.jsonl")}
        if actual != expected:
            raise RuntimeError("Merged case identifiers do not match the official population")
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        completion = dict(complete=True, model_key="qwen_base", scopes=["test","mega_official"],
                          time=now, official_cases=result["expected"]["paths"],
                          execution="existing single-GPU results plus five disjoint remaining parts",
                          parallel_plan=str(PLAN_FILE))
        official.dump(ROOT / "completion.json", completion)
        official.dump(ROOT / "run_status.json", dict(state="complete", stage="complete",
                      complete=True, physical_gpus=plan["allowed_gpus"], time=now))
        official.dump(PLAN_DIR / "completion.json", dict(completion, metrics=str(OUTPUT / "metrics.json")))
    from .missing_only_report import generate
    with (OUT / ".report.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        generate()
    return completion

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare","run","finalize"))
    parser.add_argument("--part", type=int, choices=range(PARTS))
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare()
        print(json.dumps({key:result[key] for key in ("total_cases","completed_cases_reused","remaining_cases")}))
    elif args.action == "run":
        if args.part is None: parser.error("--part is required")
        run_part(args.part)
    else:
        print(json.dumps(finalize()))
if __name__ == "__main__":
    main()
