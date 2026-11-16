"""Publish this experiment's final reports to its existing dedicated GitHub branch."""
import subprocess
from pathlib import Path
from .prepare import OUT, REPO
from experiments.aster_cr_retrain_paired_v2_20261111.common import read,write

BRANCH="aster-adaptive-m1-20261116"

def publish():
    if subprocess.check_output(["git","branch","--show-current"],cwd=REPO,text=True).strip()!=BRANCH:
        raise RuntimeError("Publication requires the dedicated adaptive experiment branch")
    if not read(OUT/"finalization.json").get("complete"):
        raise RuntimeError("Final report generation has not finished")
    files=[]
    for p in OUT.rglob("*"):
        if not p.is_file() or any(x in {"cache","data","patches","__pycache__"} for x in p.relative_to(OUT).parts):
            continue
        if p.name in {"stage_ids.json","gradient_ids.json","publication.json"}:
            continue
        if p.suffix not in {".json",".jsonl",".csv",".md",".png",".pdf",".npz"}:
            continue
        files.append(str(p.relative_to(REPO)))
    with (OUT/"publication.log").open("a") as log:
        subprocess.run(["git","add","-f","--",*files],cwd=REPO,stdout=log,stderr=log,check=True)
        changed=subprocess.run(["git","diff","--cached","--quiet"],cwd=REPO).returncode
        if changed==1:
            subprocess.run(["git","commit","-m","results(aster): publish adaptive seven-backbone outcomes"],
                           cwd=REPO,stdout=log,stderr=log,check=True)
        elif changed!=0:
            raise RuntimeError("Unable to inspect staged results")
        subprocess.run(["git","push","origin",BRANCH],cwd=REPO,stdout=log,stderr=log,check=True)
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip()
    write(OUT/"publication.json",dict(complete=True,branch=BRANCH,commit=commit,
        url="https://github.com/chunxi-alpc/TRIAGE/tree/"+BRANCH+"/reports/aster_e678_adaptive_20261116"))
