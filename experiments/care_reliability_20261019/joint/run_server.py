#!/usr/bin/env python3
"""Run Phi-4 structure shards then Qwen inference without polling GPUs."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PHI_PYTHON = "/home/gcx/n/.venv/bin/python"
QWEN_PYTHON = "/home/gcx/gcx261014/bin/python"


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phi-gpus", type=int, nargs="+", default=[0, 1, 2, 3, 6, 7])
    parser.add_argument("--qwen-gpu", type=int, default=0)
    parser.add_argument("--training-preflight", action="store_true")
    args = parser.parse_args()
    devices = args.phi_gpus + [args.qwen_gpu]
    if len(set(args.phi_gpus)) != len(args.phi_gpus) or not set(devices) <= {0, 1, 2, 3, 6, 7}:
        parser.error("Distinct authorized structure GPUs and an authorized Qwen GPU are required")
    args.output.mkdir(parents=True, exist_ok=False)
    split = "train" if args.training_preflight else "confirmation"
    limit = ["--case-limit", "2"] if args.training_preflight else []
    state = {"status": "starting", "split": split, "started_utc": datetime.now(timezone.utc).isoformat(),
             "physical_gpus": {"structures": args.phi_gpus, "qwen": args.qwen_gpu},
             "stages": [], "monitoring": "wait for own subprocess exits; no GPU polling"}
    path = args.output / "controller_status.json"
    save(path, state)
    environment = os.environ.copy()
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    workers, streams, directories = [], [], []
    started = time.perf_counter()
    try:
        for rank, gpu in enumerate(args.phi_gpus):
            directory = args.output / "structures" / f"shard{rank}"
            directories.append(directory)
            command = [PHI_PYTHON, str(HERE / "run_structures.py"), "--split", split, *limit,
                       "--shard-count", str(len(args.phi_gpus)), "--shard-index", str(rank), "--output", str(directory),
                       "--device", "cuda:0", "--base-model", "/home/gcx/models/care_module1_v5/phi-4",
                       "--adapter-model", "/home/gcx/models/care_module1_v5/trialmatchai-phi4-reasoning-lora",
                       "--evidence-model", "/home/gcx/models/care_module1_v5/biolinkbert-mednli"]
            stream = (args.output / f"structures_shard{rank}.log").open("x")
            streams.append(stream)
            process = subprocess.Popen(command, cwd=ROOT, env=environment | {"CUDA_VISIBLE_DEVICES": str(gpu)},
                                       stdout=stream, stderr=subprocess.STDOUT)
            workers.append(process)
            state["stages"].append({"name": f"structures_shard{rank}", "pid": process.pid,
                                    "physical_gpu": gpu, "command": command})
        state["status"] = "running_structures"
        save(path, state)
        codes = [process.wait() for process in workers]
        for entry, code in zip(state["stages"], codes):
            entry["return_code"] = code
        if any(codes):
            raise RuntimeError("At least one structure shard failed; Qwen was not started")
        command = [QWEN_PYTHON, str(HERE / "run_joint.py"),
                   "--model", "/home/gcx/care_20261016/CARE_Codex_Pilot/models/Qwen3-4B-Base",
                   "--structure-runs", *map(str, directories),
                   "--device", "cuda:0", "--split", split, *limit,
                   "--sources", "reference", "reference_positive", "v6", "v7",
                   "--include-reference-diagnostic", "--output", str(args.output / "joint")]
        state["status"] = "running_joint"
        state["stages"].append({"name": "joint", "physical_gpu": args.qwen_gpu, "command": command})
        save(path, state)
        with (args.output / "joint.log").open("x") as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=environment | {"CUDA_VISIBLE_DEVICES": str(args.qwen_gpu)},
                                       stdout=stream, stderr=subprocess.STDOUT)
            state["stages"][-1]["pid"] = process.pid
            save(path, state)
            code = process.wait()
        state["stages"][-1]["return_code"] = code
        if code:
            raise RuntimeError("Joint inference failed or is incomplete; see the preserved stage status")
        state["status"] = "completed"
    except Exception as error:
        state.update(status="failed", error=type(error).__name__ + ": " + str(error))
    finally:
        for stream in streams:
            stream.close()
        state.update(ended_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter()-started)
        save(path, state)
    return 0 if state["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())

