"""Use the existing scheduler interface and gate expansion on Qwen acceptance."""
import argparse
import json
from pathlib import Path
from experiments.aster_cr_retrain_paired_v2_20261111.common import read,write
from .prepare import REPO,OUT,KEYS

SCHEDULER=Path("/home/gcx/aster-gpu-queue-20261113/scheduler.py")
QUEUE=Path("/home/gcx/aster-gpu-queue-20261113/m1_revision_20261117")
PYTHON="/home/gcx/gcx261014/bin/python"

def manifest(resume_models=()):
    jobs=[]
    for key in KEYS:
        cfg=read(OUT/"models"/key/"configuration.json")
        jobs.append(dict(id="m1_revision_"+key,
           command=[PYTHON,"-m","experiments.aster_m1_revision_20261117.run","--model",key,"--publish"]+(["--resume"] if key in resume_models else []),
           cwd=str(REPO),gpu_count=1,allowed_gpus=[0,1,5,6],estimated_vram_gib=cfg["model"]["expected_peak_gib"][1],
           dependencies=[] if key=="qwen" else ["m1_revision_qwen"],
           completion_marker=str(OUT/"models"/key/"queue_done.json"),priority=10 if key=="qwen" else 20,
           env=dict(OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",
                    PYTHONUNBUFFERED="1",TOKENIZERS_PARALLELISM="false")))
    jobs.append(dict(id="m1_revision_finalize",command=[PYTHON,"-m","experiments.aster_m1_revision_20261117.report","--publish"],
                     cwd=str(REPO),gpu_count=0,dependencies=[j["id"] for j in jobs],run_on_failure=True,
                     completion_marker=str(OUT/"summary_complete.json")))
    return dict(owner="aster_m1_revision_20261117",jobs=jobs)

def prepare():
    QUEUE.mkdir(parents=True,exist_ok=True)
    target=QUEUE/"jobs.json"
    data=manifest()
    if target.exists() and read(target)!=data:raise ValueError("Existing queue manifest differs")
    write(target,data)
    write(OUT/"queue_manifest.json",data)
    return target

if __name__=="__main__":
    print(prepare())
