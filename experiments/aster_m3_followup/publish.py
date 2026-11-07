"""Write reviewable artifacts and update this experiment branch without merging."""
import gzip,shutil,subprocess
from .common import *
BRANCH="codex/aster-m3-p0-p3-20261107"
def export():
    target=ROOT/"exports";target.mkdir(exist_ok=True)
    for source in sorted((ROOT/"predictions").glob("*/*.jsonl")):
        output=target/(source.parent.name+"__"+source.name+".gz")
        with source.open("rb") as a,gzip.open(output,"wb") as b:shutil.copyfileobj(a,b)
    for source in sorted((ROOT/"p0").glob("fixed_nodes_*.jsonl")):
        with source.open("rb") as a,gzip.open(target/(source.name+".gz"),"wb") as b:shutil.copyfileobj(a,b)
    selections={}
    for p in sorted((ROOT/"models").glob("*/complete.json")):
        selections[p.parent.name]=read(p)
    write(ROOT/"training_summary.json",selections)
def publish(final=False):
    actual=subprocess.check_output(["git","branch","--show-current"],cwd=REPO,text=True).strip()
    if actual!=BRANCH:raise RuntimeError("Publish is limited to the experiment branch")
    if final:export()
    paths=["experiments/aster_m3_followup","results/aster_m3_followup_20261107","README.md",
           "docs/ASTER/README.md","docs/ASTER/manuscript.zh.md","docs/ASTER/method-clarifications.zh.md"]
    subprocess.run(["git","add","--"]+paths,cwd=REPO,check=True)
    result=subprocess.run(["git","diff","--cached","--quiet"],cwd=REPO)
    if result.returncode==1:
        message="Complete M3 P0-P3 comparisons and source-paired reports" if final else "Add M3 P0-P3 experiment pipeline and synchronize completed M3 results"
        subprocess.run(["git","commit","-m",message],cwd=REPO,check=True)
    elif result.returncode!=0:raise RuntimeError("Unable to inspect staged changes")
    subprocess.run(["git","push","-u","origin",BRANCH],cwd=REPO,check=True)
    status("publish",state="complete",branch=BRANCH,final=final)
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--final",action="store_true");a=p.parse_args();publish(a.final)
