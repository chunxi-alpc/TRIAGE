"""Sequential jobs per assigned GPU; no GPU polling or persistent monitoring."""
import argparse, datetime, json, os, subprocess, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/"reports/aster_codex03_external_20261112"
def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    tmp.replace(path)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
    p=argparse.ArgumentParser();p.add_argument("--plan",type=Path,required=True);args=p.parse_args()
    plan=json.loads(args.plan.read_text())
    gpu=int(plan["gpu"]);folder=REPORT/"controllers"/f"gpu{gpu}";folder.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=str(gpu);env["OMP_NUM_THREADS"]="4"
    env["MKL_NUM_THREADS"]="4";env["TOKENIZERS_PARALLELISM"]="false"
    if plan.get("pythonpath"):env["PYTHONPATH"]=plan["pythonpath"]+":"+str(ROOT)+":"+env.get("PYTHONPATH","")
    state=dict(pid=os.getpid(),gpu=gpu,plan=str(args.plan),started_at=now(),stage="running",jobs=[])
    write(folder/"status.json",state)
    index=0
    while True:
        plan=json.loads(args.plan.read_text())
        if index>=len(plan["jobs"]):break
        job=plan["jobs"][index]
        index+=1
        entry=dict(name=job["name"],started_at=now(),status="running",command=job["args"])
        state["current"]=job["name"];state["jobs"].append(entry);write(folder/"status.json",state)
        started=time.perf_counter()
        with (folder/(job["name"]+".log")).open("a",encoding="utf-8") as log:
            proc=subprocess.Popen([sys.executable,*job["args"]],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            entry["pid"]=proc.pid;write(folder/"status.json",state)
            code=proc.wait()
        entry.update(status="complete" if code==0 else "failed",returncode=code,
                     wall_seconds=time.perf_counter()-started,finished_at=now())
        write(folder/"status.json",state)
        subprocess.run([sys.executable,"-m","experiments.aster_codex03_external_20261112.finalize"],
                       cwd=ROOT,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        if code:
            state["stage"]="failed";state["finished_at"]=now();write(folder/"status.json",state)
            return code
    state["stage"]="complete";state["current"]=None;state["finished_at"]=now()
    write(folder/"status.json",state);write(folder/"completion.json",state)
    subprocess.run([sys.executable,"-m","experiments.aster_codex03_external_20261112.finalize"],cwd=ROOT,env=env)
    return 0
if __name__=="__main__":raise SystemExit(main())
