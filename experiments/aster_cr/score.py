import argparse,time
import numpy as np
from .common import *
def run(model,split,shard=0,shards=3,limit=None):
 if split=="test" and not read(ROOT/model/"frozen.json").get("frozen"):
  raise RuntimeError("Test scoring requires a frozen development configuration")
 import torch
 from .engine import Engine
 torch.set_num_threads(4)
 out=ROOT/model;target=out/"scores"/f"{split}_{shard}.jsonl"
 done={r["view_id"] for r in rows(target) if r.get("status") in ("ok","unavailable")}
 allviews=[v for v in rows(ROOT/"data/views.jsonl") if v["split"]==split]
 owned=set(list(dict.fromkeys(v["group_id"] for v in allviews))[shard::shards])
 views=[v for v in allviews if v["group_id"] in owned]
 byid={v["view_id"]:v for v in allviews}
 def input_key(v):return (v["prompt"],tuple(v.get("labels") or ("\u662f","\u5426")))
 cache={}
 for record in rows(target):
  if record.get("status")=="ok" and record["view_id"] in byid:cache[input_key(byid[record["view_id"]])]=record
 if limit is not None:views=views[:limit]
 todo=[v for v in views if v["view_id"] not in done]
 if not todo:return
 eng=Engine(model);started=time.time()
 for i,v in enumerate(todo):
  t=time.perf_counter();rec=dict(view_id=v["view_id"],source_id=v["source_id"],group_id=v["group_id"],split=split)
  try:
   key=input_key(v)
   if key in cache:
    previous=cache[key]
    rec.update({k:val for k,val in previous.items() if k not in ("view_id","source_id","group_id","split","seconds","reused_from")})
    rec.update(seconds=0.,reused_from=previous["view_id"])
    append(target,rec)
    continue
   result=eng.score(v["prompt"],labels=v.get("labels"),capture=True,grad=(split=="fit"))
   feature=out/"features"/(v["view_id"]+".npz");feature.parent.mkdir(parents=True,exist_ok=True)
   arrays=dict(heads=result.pop("heads").numpy().astype(np.float16),residual=result.pop("residual").numpy().astype(np.float16))
   if "grads" in result:arrays["grads"]=result.pop("grads").numpy().astype(np.float16)
   np.savez_compressed(feature,**arrays)
   rec.update(result,status="ok",seconds=time.perf_counter()-t,feature=str(feature));cache[key]=rec
  except ValueError as exc:
   rec.update(status="unavailable",error=str(exc),seconds=time.perf_counter()-t)
  append(target,rec)
  status(f"{model}_score_{split}_{shard}",state="running",done=len(done)+i+1,total=len(views),
    seconds=time.time()-started,peak_gib=torch.cuda.max_memory_allocated()/2**30)
 write(out/"cost"/f"score_{split}_{shard}.json",dict(seconds=time.time()-started,counts=eng.counts,
     peak_gib=torch.cuda.max_memory_allocated()/2**30,completed=len(todo)))
 status(f"{model}_score_{split}_{shard}",state="complete",done=len(views),seconds=time.time()-started)
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--model",default="qwen");p.add_argument("--split",required=True)
 p.add_argument("--shard",type=int,default=0);p.add_argument("--shards",type=int,default=3);p.add_argument("--limit",type=int)
 a=p.parse_args();run(**vars(a))
