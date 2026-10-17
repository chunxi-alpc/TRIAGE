#!/usr/bin/env python3
"""Server launcher: one O2 job + three independent clinical folds on four GPUs.

No model parallelism is needed: each GPU holds one complete 4B model.
All computation and logs stay on the server. No continuous GPU polling.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {0,1,2,3,6,7}

def save(path, obj):
    path.write_text(json.dumps(obj, indent=2))

def available_gpus(preferred):
    query = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True)
    active = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True)
    busy = {line.split(",")[0].strip() for line in active.splitlines() if "," in line}
    free = {}
    for line in query.splitlines():
        index, uuid, memory, utilization = [x.strip() for x in line.split(",")]
        if int(index) in ALLOWED and uuid not in busy and int(memory) >= 18000 and int(utilization) <= 10:
            free[int(index)] = dict(uuid=uuid, free_MiB=int(memory), utilization=int(utilization))
    selected = [i for i in preferred if i in free][:4]
    if len(selected) < 4:
        raise RuntimeError(f"Need four free authorized GPUs with >=18000 MiB free; available={list(free)}")
    return selected, free

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts",type=Path,default=Path("/home/gcx/care_20261016/CARE_Codex_Pilot"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--gpus",type=int,nargs="+",default=[0,1,2,3,6,7],
                   help="preference order; first four available authorized GPUs are used")
    args=p.parse_args()
    assert set(args.gpus) <= ALLOWED and len(set(args.gpus)) == len(args.gpus)
    assert len(args.gpus) >= 4
    args.output.mkdir(parents=True,exist_ok=True)
    logs=args.output/"logs"
    logs.mkdir(exist_ok=True)
    # Cooperative locks protect launches from this tool; nvidia-smi checks other users' jobs.
    launch_lock=(args.output/"launcher.lock").open("a")
    fcntl.flock(launch_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpus,free=available_gpus(args.gpus)
    locks=[]
    for gpu in gpus:
        lock=Path(f"/tmp/care-module2-gpu{gpu}.lock").open("a")
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        locks.append(lock)
    plan=dict(status="preflight",gpus=gpus,gpu_snapshot=free,
              assignment={str(gpus[0]):"o2",**{str(gpus[i+1]):f"clinical_fold{i}" for i in range(3)}},
              estimated_peak_GiB_per_gpu=[10,14],estimated_hours=[1,2],
              estimated_total_forwards=65000,no_continuous_gpu_monitoring=True)
    save(args.output/"launch.json",plan)
    # Tests and archive reproduction are server CPU tasks, before any GPU experiment.
    with (logs/"preflight.log").open("w") as log:
        subprocess.run([sys.executable,"-m","pytest","-q","tests/test_module2_followup.py"],
                       cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run([sys.executable,"scripts/analyze_module2_archive.py","--verify"],
                       cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    # Run a real smoke on O2 and clinical inputs before dispatching the long grid.
    for dataset,gpu in [("o2",gpus[0]),("clinical",gpus[1])]:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="4",MKL_NUM_THREADS="4")
        with (logs/f"{dataset}_smoke.log").open("w") as log:
            subprocess.run([sys.executable,"-u","scripts/run_module2_followup.py",
                            "--dataset",dataset,"--artifacts",str(args.artifacts),
                            "--output",str(args.output/(dataset+"_smoke")),"--smoke"],
                           cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    jobs=[]
    specs=[("o2","o2",None,gpus[0])]+[(f"clinical_fold{i}","clinical",i,gpus[i+1]) for i in range(3)]
    streams=[]
    for name,dataset,fold,gpu in specs:
        command=[sys.executable,"-u","scripts/run_module2_followup.py","--dataset",dataset,
                 "--artifacts",str(args.artifacts),"--output",str(args.output/name)]
        if fold is not None:
            command += ["--folds",str(fold)]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="4",MKL_NUM_THREADS="4")
        log=(logs/(name+".log")).open("a")
        streams.append(log)
        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        jobs.append((name,proc))
    plan.update(status="launched",jobs=[dict(name=n,pid=proc.pid) for n,proc in jobs])
    save(args.output/"launch.json",plan)
    # One bounded startup check; then only wait for process completion, without GPU monitoring.
    time.sleep(30)
    plan["startup_alive"]={name:proc.poll() is None for name,proc in jobs}
    save(args.output/"launch.json",plan)
    exits={name:proc.wait() for name,proc in jobs}
    for stream in streams:
        stream.close()
    plan.update(status="inference_complete" if all(code==0 for code in exits.values()) else "failed_or_partial",
                exit_codes=exits)
    save(args.output/"launch.json",plan)
    if not all(code==0 for code in exits.values()):
        raise SystemExit("Some jobs failed; successful journals preserved; inspect server log files.")
    analyzer=[sys.executable,"scripts/analyze_module2_followup.py","--artifacts",str(args.artifacts)]
    with (logs/"analysis.log").open("w") as log:
        subprocess.run(analyzer+["--run",str(args.output/"o2")],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run(analyzer+["--run"]+[str(args.output/f"clinical_fold{i}") for i in range(3)]
                       +["--output",str(args.output/"clinical_analysis")],
                       cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    plan["status"]="completed_real_inference_and_analysis"
    save(args.output/"launch.json",plan)

if __name__=="__main__":
    main()
