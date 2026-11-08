import json, os, time
from pathlib import Path
ROOT=Path(os.environ.get("ASTER_CR_ROOT","/home/gcx/aster_cr_20261108"))
REPO=Path(__file__).resolve().parents[2]
SEED=20261108
GPUS=[int(g) for g in os.environ.get("ASTER_CR_GPUS","0,1").split(",")]
TASK_KINDS={"original","A","P","C","F","F0","grid"}
MODELS={"qwen":"/home/gcx/TRIAGE/models/modelscope/Qwen3-8B","medgemma":"/home/gcx/TRIAGE/models/modelscope/medgemma-27b-text-it"}
def rows(p):
 p=Path(p)
 if not p.exists(): return []
 with p.open() as f: return [json.loads(s) for s in f if s.strip()]
def read(p):
 p=Path(p)
 return json.loads(p.read_text()) if p.exists() else {}
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_suffix(p.suffix+".tmp")
 tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+"\n");tmp.replace(p)
def lines(p,rr):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_suffix(p.suffix+".tmp")
 with tmp.open("w") as f:
  for r in rr:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+"\n")
 tmp.replace(p)
def append(p,r):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open("a") as f:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+"\n")
def status(stage,**kw):write(ROOT/"status"/(stage+".json"),dict(time=time.strftime("%Y-%m-%d %H:%M:%S"),pid=os.getpid(),**kw))
