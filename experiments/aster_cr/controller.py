"""Detached server execution on configured GPUs with persistent state and logs."""
import argparse,fcntl,subprocess,sys,time,os,traceback
from .common import *
def find_existing(gpu,args):
 for proc in Path("/proc").iterdir():
  if not proc.name.isdigit():continue
  try:
   cmd=(proc/"cmdline").read_bytes().decode().split("\0")
   env=dict(part.split("=",1) for part in (proc/"environ").read_bytes().decode().split("\0") if "=" in part)
   if ("-m" in cmd and cmd[cmd.index("-m")+1:-1]==args
       and (proc/"cwd").resolve()==REPO
       and env.get("ASTER_CR_ROOT")==str(ROOT)
       and env.get("CUDA_VISIBLE_DEVICES")==str(gpu)
       and (proc/"stat").read_text().split(") ",1)[1][0]!="Z"):
    return int(proc.name)
  except (OSError,ValueError,UnicodeError):pass
 return None
def phase(name,jobs):
 status("pipeline",state="running",phase=name)
 procs=[];adopted=[]
 for gpu,args in jobs:
  marker=ROOT/"jobs"/(name+"_"+str(gpu)+".json")
  old=read(marker)
  if old.get("state")=="complete" and old.get("args")==args:continue
  existing=None
  if name=="annotations" or args[0]=="experiments.aster_cr.baselines":
   existing=find_existing(gpu,args)
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
  if args[0]=="experiments.aster_cr.baselines":
   model=args[args.index("--model")+1] if "--model" in args else "qwen"
   kind=args[args.index("--kind")+1];lr=float(args[args.index("--lr")+1])
   split=args[args.index("--split")+1] if "--evaluate" in args else "dev"
   complete=bool(read(ROOT/model/"baselines"/(f"{kind}_lr{lr}_{split}_metrics.json")))
  else:
   shard=args[args.index("--shard")+1]
   complete=read(ROOT/"status"/("annotate_"+shard+".json")).get("state")=="complete"
  write(ROOT/"jobs"/(name+"_"+str(g)+".json"),dict(state="complete" if complete else "failed",args=args,adopted=True,completion_from_output=True))
  if not complete:codes.append(1)
 if any(codes):raise RuntimeError(f"{name} failed with exit codes {codes}; see logs")
def baseline_result(args):
 model=args[args.index("--model")+1] if "--model" in args else "qwen"
 kind=args[args.index("--kind")+1];lr=float(args[args.index("--lr")+1])
 split=args[args.index("--split")+1] if "--evaluate" in args else "dev"
 return ROOT/model/"baselines"/f"{kind}_lr{lr}_{split}_metrics.json"
def queue_result(args):
 if args[0]=="experiments.aster_cr.comparators":
  model=args[args.index("--model")+1];split=args[args.index("--split")+1];shard=args[args.index("--shard")+1]
  if split!="dev":raise ValueError("Only development comparators may overlap training")
  return ROOT/model/"comparators"/f"{split}_{shard}_metrics.json"
 if args[0]!="experiments.aster_cr.baselines":raise ValueError("Unsupported training queue task")
 return baseline_result(args)
def process_alive(pid):
 try:return (Path("/proc")/str(pid)/"stat").read_text().split(") ",1)[1][0]!="Z"
 except FileNotFoundError:return False
def baseline_queue(name,jobs):
 """Fill freed GPUs with training or independent dev comparisons; join before freeze."""
 pending=[];active={};completed=[];failed=[]
 def marker(i):return ROOT/"jobs"/f"{name}_job{i}.json"
 def record(i,args,gpu,code,adopted=False):
  ok=code==0 and bool(read(queue_result(args)))
  write(marker(i),dict(state="complete" if ok else "failed",args=args,gpu=gpu,exit_code=code,adopted=adopted))
  (completed if ok else failed).append(i)
 for i,args in enumerate(jobs):
  existing=[(gpu,find_existing(gpu,args)) for gpu in GPUS]
  existing=[(gpu,pid) for gpu,pid in existing if pid is not None]
  if existing:
   if len(existing)!=1 or existing[0][0] in active:raise RuntimeError("Multiple baseline workers on one assigned GPU")
   gpu,pid=existing[0];active[gpu]=dict(i=i,args=args,pid=pid,proc=None,log=None)
  elif read(queue_result(args)):
   record(i,args,None,0)
  else:pending.append((i,args))
 while pending or active:
  for gpu,job in list(active.items()):
   p=job["proc"]
   code=p.poll() if p is not None else None if process_alive(job["pid"]) else 0
   if code is None:continue
   if job["log"] is not None:job["log"].close()
   record(job["i"],job["args"],gpu,code,adopted=p is None)
   del active[gpu]
  # Let already running jobs finish if one fails; do not launch more work.
  if not failed:
   for gpu in GPUS:
    if gpu in active or not pending:continue
    memory=subprocess.check_output(["nvidia-smi","-i",str(gpu),"--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True)
    required=30 if pending[0][1][0]=="experiments.aster_cr.comparators" else 50
    if int(memory.strip())<required*1024:continue
    i,args=pending.pop(0)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",TOKENIZERS_PARALLELISM="false")
    logfile=ROOT/"logs"/f"{name}_job{i}_{gpu}.log";logfile.parent.mkdir(parents=True,exist_ok=True)
    f=logfile.open("a")
    try:p=subprocess.Popen([sys.executable,"-u","-m",*args],cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT)
    except BaseException:f.close();raise
    active[gpu]=dict(i=i,args=args,pid=p.pid,proc=p,log=f)
  workers=[dict(pid=j["pid"],gpu=g,job=j["i"],args=j["args"],adopted=j["proc"] is None) for g,j in active.items()]
  write(ROOT/"processes.json",dict(controller=os.getpid(),phase=name,workers=workers,pending=[i for i,args in pending],completed=completed,failed=failed))
  status("pipeline",state="draining_after_failure" if failed else "running" if active else "waiting_for_memory" if pending else "running",phase=name,completed=len(completed),pending=len(pending),active=len(active),failed=failed,training_completed=sum(jobs[i][0]=="experiments.aster_cr.baselines" for i in completed),dev_comparator_completed=sum(jobs[i][0]=="experiments.aster_cr.comparators" for i in completed))
  if failed and not active:raise RuntimeError(f"{name} failed jobs {failed}; see logs")
  if pending or active:time.sleep(5)

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
