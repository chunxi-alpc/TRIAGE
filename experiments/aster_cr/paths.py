"""Cache unedited features at all reachable tree nodes, then execute each readout autonomously."""
import argparse,pickle,time
import numpy as np
from collections import defaultdict
from .common import *
from .readout import predict
from experiments.aster_closed_form.data import canonical_guide,prompt
def collect(model,split,shard=0,shards=len(GPUS)):
 if split=="test" and not read(ROOT/model/"frozen.json").get("frozen"):raise RuntimeError("test is not frozen")
 if split=="test":
  from .resume_repair import release_test_cache
  release_test_cache(model)
 from .engine import Engine
 import torch
 torch.set_num_threads(4)
 root=ROOT/model;target=root/"paths"/f"features_{split}_{shard}.jsonl"
 done={r["id"] for r in rows(target) if r.get("status")=="ok"}
 cases=[c for c in rows(ROOT/"data/path_cases.jsonl") if c["split"]==split][shard::shards]
 from .features import Features
 existing=Features(model,(split,))
 cached={v["prompt"]:existing.scores[v["view_id"]] for v in existing.views if not v.get("labels")}
 if all(c["case_id"]+":"+n["id"] in done for c in cases for n in canonical_guide(c["guideline"])["nodes"]):
  status(f"{model}_paths_{split}_{shard}",state="complete",nodes=len(done),cache_only=True);return
 eng=Engine(model);start=time.time();count=0
 for c in cases:
  for node in canonical_guide(c["guideline"])["nodes"]:
   key=c["case_id"]+":"+node["id"]
   if key in done:continue
   rec=dict(id=key,case_id=c["case_id"],source_id=c["source_id"],gid=node["id"])
   try:
    text=prompt(c["guideline"],c["definitions"],c["patient"],node["id"])
    if text in cached:
     old=cached[text];rec.update(ell=old["ell"],prediction=old["prediction"],tokens=old["tokens"],feature=old["feature"],status="ok",reused_from=old["view_id"],inference_seconds=old.get("inference_seconds",0.))
     append(target,rec);count+=1
     continue
    result=eng.score(text,capture=True)
    path=root/"path_features"/(key+".npz");path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,heads=result.pop("heads").numpy().astype(np.float16),residual=result.pop("residual").numpy().astype(np.float16))
    rec.update(result,feature=str(path),status="ok")
   except ValueError as e:rec.update(status="unavailable",error=str(e))
   append(target,rec);count+=1
  status(f"{model}_paths_{split}_{shard}",state="running",nodes=count,seconds=time.time()-start)
 write(root/"cost"/f"paths_{split}_{shard}.json",dict(seconds=time.time()-start,counts=eng.counts,peak_gib=torch.cuda.max_memory_allocated()/2**30))
 status(f"{model}_paths_{split}_{shard}",state="complete",nodes=count)
def execute(case,node_scores):
 tree=canonical_guide(case["guideline"]);nodes={n["id"]:n for n in tree["nodes"]};gid=tree["nodes"][0]["id"]
 actual=[];first=None;gold=case["gold_path"];error=None
 while gid in nodes:
  if len(actual)>len(nodes):error="cycle";break
  score=node_scores.get(gid)
  if score is None:error="unavailable_score";break
  label=int(score>0);actual.append(dict(gid=gid,label=label))
  i=len(actual)-1
  if first is None and (i>=len(gold) or gold[i]!=actual[-1]):first=gid
  gid=nodes[gid]["yes" if label else "no"]
 return dict(correct=not error and actual==gold and gid==case["gold_leaf"],path=actual,leaf=gid,
   first_error=first,error=error,node_forward_count=len(actual))
def evaluate(model,split,names=None):
 root=ROOT/model;models={}
 for p in (root/"readouts").glob("*.pkl"):
  if names is None or p.stem in names:
   with p.open("rb") as f:models[p.stem]=pickle.load(f)
 records=[r for p in (root/"paths").glob(f"features_{split}_*.jsonl") for r in rows(p) if r.get("status")=="ok"]
 grouped=defaultdict(list)
 for r in records:grouped[r["case_id"]].append(r)
 results={name:[] for name in models}
 for c in rows(ROOT/"data/path_cases.jsonl"):
  if c["split"]!=split:continue
  rs=grouped[c["case_id"]];predictions={name:{} for name in models};latencies={name:{} for name in models}
  base_times={r["gid"]:r.get("inference_seconds",0.) for r in rs}
  for r in rs:
   with np.load(r["feature"]) as a:
    heads=a["heads"].astype(np.float32);heads=heads.reshape(-1,heads.shape[-1])
    for name,obj in models.items():
     raw=heads[obj["heads"]].reshape(1,-1) if obj.get("heads") else None
     started=time.perf_counter();value=float(predict(obj,np.array([r["ell"]]),raw)[0]);latencies[name][r["gid"]]=time.perf_counter()-started;predictions[name][r["gid"]]=value
  for name in models:
   result=execute(c,predictions[name]);visited=[n["gid"] for n in result["path"]]
   results[name].append(dict(case_id=c["case_id"],source_id=c["source_id"],**result,
    cached_forward_seconds=sum(base_times.get(g,0.) for g in visited),readout_seconds=sum(latencies[name].get(g,0.) for g in visited),
    latency_scope="sum of measured cached unedited forwards and CPU readout; not an online wall-clock benchmark"))
 summary={}
 full={r["case_id"]:r["correct"] for r in results.get("full",[])}
 for name,rs in results.items():
  lines(root/"path_results"/f"{name}_{split}.jsonl",rs)
  summary[name]=dict(n=len(rs),correct=sum(r["correct"] for r in rs),accuracy=sum(r["correct"] for r in rs)/len(rs) if rs else None,
    corrected=sum(r["correct"] and not full.get(r["case_id"],False) for r in rs),
    new_errors=sum(not r["correct"] and full.get(r["case_id"],False) for r in rs))
 write(root/"path_results"/(split+".json"),summary);return summary
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--model",default="qwen");p.add_argument("--split",required=True);p.add_argument("--shard",type=int,default=0)
 p.add_argument("--shards",type=int,default=len(GPUS));a=p.parse_args();collect(**vars(a))
