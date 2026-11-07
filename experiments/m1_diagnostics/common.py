import json, os, time
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path("/home/gcx/m1_diagnostics_20261106")
REPO=Path("/home/gcx/TRIAGE")
QWEN=Path("/home/gcx/aster_existing_data_20261106")
MED=Path("/home/gcx/medgemma_m1_20261106")
REPAIR=Path("/home/gcx/medgemma_m1_repair_20261106")
MODELS={"qwen":REPO/"models/modelscope/Qwen3-8B","medgemma":REPO/"models/modelscope/medgemma-27b-text-it"}
LEGACY={"qwen":QWEN,"medgemma":MED}
SEED,BOOTSTRAPS,EPSILON,RHO=20261106,2000,.1,.8
PRECISION="bf16_backbone_fp32_label_projection"
SYSTEM="根据给定指南和病例，判断当前节点的完整条件是否成立。仅输出“是”或“否”。 For English inputs, 是 means Yes and 否 means No."
def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def rows(path):
    p=Path(path)
    return [json.loads(s) for s in p.read_text(encoding="utf-8").splitlines() if s.strip()] if p.exists() else []
def write(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8");tmp.replace(p)
def write_rows(path,values):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f".{os.getpid()}.tmp")
    tmp.write_text("".join(json.dumps(x,ensure_ascii=False,allow_nan=False)+"\n" for x in values),encoding="utf-8");tmp.replace(p)
def status(name,**value):write(ROOT/"status"/(name+".json"),dict(updated=time.time(),**value))
def balanced(items,count,seed=SEED,key="source_id"):
    rng=np.random.default_rng(seed);groups=defaultdict(list)
    for x in items:groups[str(x[key])].append(x)
    order=sorted(groups);rng.shuffle(order)
    for k in order:rng.shuffle(groups[k])
    selected=[]
    while len(selected)<count and any(groups.values()):
        for k in order:
            if groups[k] and len(selected)<count:selected.append(groups[k].pop())
    return selected
def feature(model,vid):
    import torch
    return torch.load(LEGACY[model]/"features"/(vid.replace("/","_")+".pt"),map_location="cpu",weights_only=False)
def frozen():return read(ROOT/"data/frozen_heads.json")
def effects(a,b,forward,reverse):
    t=b-a;f=forward-a;r=reverse-b;e=(f-r)/2
    return dict(left_ell=a,right_ell=b,te=t,forward_ell=forward,reverse_ell=reverse,ie_forward=f,ie_reverse=r,e=e,
      recovery=e/t if abs(t)>=EPSILON else None,raw_ratio=e/t if t!=0 else None,effective=abs(t)>=EPSILON)
def distribution(values):
    a=np.asarray(values,dtype=float)
    if not len(a):return {"n":0}
    return dict(n=len(a),**{k:float(v) for k,v in zip(["min","p05","p25","median","p75","p95","max"],
      np.quantile(a,[0,.05,.25,.5,.75,.95,1]))},mean=float(a.mean()),rms=float(np.sqrt((a*a).mean())))

def cache_path(model,name,pair_id):
    return ROOT/"patch"/model/PRECISION/name/(pair_id.replace("/","_")+".json")
