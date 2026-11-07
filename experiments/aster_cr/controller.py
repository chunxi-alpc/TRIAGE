"""Detached server execution; only GPUs 0,1,6, persistent state and logs."""
import argparse,fcntl,subprocess,sys,time,os,traceback
from .common import *
def phase(name,jobs):
 status("pipeline",state="running",phase=name)
 procs=[];adopted=[]
 for gpu,args in jobs:
  marker=ROOT/"jobs"/(name+"_"+str(gpu)+".json")
  old=read(marker)
  if old.get("state")=="complete" and old.get("args")==args:continue
  existing=None
  if name=="annotations":
   for proc in Path("/proc").iterdir():
    if not proc.name.isdigit():continue
    try:
     cmd=(proc/"cmdline").read_bytes().decode().split("\0")
     if args[0] in cmd and cmd[cmd.index(args[0]):cmd.index(args[0])+len(args)]==args and (proc/"cwd").resolve()==REPO:
      existing=int(proc.name);break
    except (OSError,ValueError):pass
  if existing is not None:
   adopted.append((existing,gpu,args));continue
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",TOKENIZERS_PARALLELISM="false")
  required=70 if "medgemma" in args else 50 if "experiments.aster_cr.baselines" in args else 20 if "annotations"==name else 30
  while True:
   memory=subprocess.check_output(["nvidia-smi","-i",str(gpu),"--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True)
   if int(memory.strip())>=required*1024:break
   status("pipeline",state="waiting_for_memory",phase=name,gpu=gpu,required_free_gib=required)
   time.sleep(30)
  logfile=ROOT/"logs"/f"{name}_{gpu}.log";logfile.parent.mkdir(parents=True,exist_ok=True);f=logfile.open("a")
  p=subprocess.Popen([sys.executable,"-u","-m",*args],cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT)
  procs.append((p,f,gpu,args))
 write(ROOT/"processes.json",dict(controller=os.getpid(),phase=name,workers=[dict(pid=p.pid,gpu=g) for p,f,g,args in procs]+[dict(pid=pid,gpu=g,adopted=True) for pid,g,args in adopted]))
 codes=[]
 for p,f,g,args in procs:
  code=p.wait();codes.append(code);f.close()
  write(ROOT/"jobs"/(name+"_"+str(g)+".json"),dict(state="complete" if code==0 else "failed",args=args,exit_code=code))
 for pid,g,args in adopted:
  while True:
   try:
    stat=(Path("/proc")/str(pid)/"stat").read_text().split(") ",1)[1][0]
    if stat=="Z":break
   except FileNotFoundError:break
   time.sleep(5)
  shard=args[args.index("--shard")+1]
  if read(ROOT/"status"/("annotate_"+shard+".json")).get("state")!="complete":codes.append(1)
  else:write(ROOT/"jobs"/(name+"_"+str(g)+".json"),dict(state="complete",args=args,adopted=True))
 if any(codes):raise RuntimeError(f"{name} failed with exit codes {codes}; see logs")
def main(stage):
 ROOT.mkdir(parents=True,exist_ok=True)
 with (ROOT/"controller.lock").open("a") as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   if stage in ("prepare","all"):
    phase("annotations",[(gpu,["experiments.aster_cr.annotate","--shard",str(i)]) for i,gpu in enumerate(GPUS)])
    from .build import build
    build()
    for split in ("fit","dev"):
     phase("qwen_"+split,[(gpu,["experiments.aster_cr.score","--split",split,"--shard",str(i)]) for i,gpu in enumerate(GPUS)])
    status("preparation",state="complete")
   if stage in ("downstream","all"):
    from .pipeline import downstream
    downstream(phase)
   status("pipeline",state="stage_complete" if stage=="prepare" else "complete",executed_stage=stage)
  except BaseException as exc:
   status("pipeline",state="failed",error=str(exc),traceback=traceback.format_exc());raise
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--stage",choices=["prepare","downstream","all"],default="prepare")
 main(p.parse_args().stage)
