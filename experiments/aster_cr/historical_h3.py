"""Only missing features at unique historical original-error nodes; no rerun of full paths."""
import argparse,time
import numpy as np
from .common import *
from .engine import Engine
from experiments.aster_closed_form.data import prompt
def cases():
 old=Path("/home/gcx/TRIAGE/results/aster_learnable_m3_20261106")
 index={r["case_id"]:r for r in rows(old/"data/test.jsonl")};pred={}
 for p in (old/"test").glob("predictions*.jsonl"):
  for r in rows(p):
   if r.get("method")=="full" and r.get("status")=="complete":pred[r["case_id"]]=r
 mapping=read(Path("/home/gcx/aster_manuscript_20261030/clinical01/source_clusters.json"))["records"]
 source={r["filename"]:r["cluster"] for r in mapping.values()}
 result=[];seen=set()
 for cid,r in sorted(pred.items()):
  if r["correct"] or cid not in index:continue
  c=index[cid]
  for gold,got in zip(c["gold_path"],r["predicted_path"]):
   if gold!=got:
    key=c["filename"]+":"+gold["gid"]
    if key in seen:break
    seen.add(key)
    result.append(dict(view_id="historical:"+key,source_id=source.get(c["filename"],c["source_id"]),
      group_id=key,label=gold["label"],case_id=cid,original_historical_error=True,
      prompt=prompt(c["guideline"],[],c["patient"],gold["gid"])))
    break
 return result
def run(model):
 import torch
 torch.set_num_threads(4);root=ROOT/model/"historical_h3";target=root/"scores.jsonl"
 vs=cases();lines(ROOT/"data/historical_h3_nodes.jsonl",vs)
 fit_sources={r["source_id"] for r in rows(ROOT/"data/sources.jsonl")}
 if any(v["source_id"] in fit_sources for v in vs):raise ValueError("Historical diagnostic source unexpectedly overlaps frozen new sources")
 done={r["view_id"] for r in rows(target)};todo=[v for v in vs if v["view_id"] not in done]
 if not todo:return
 eng=Engine(model);start=time.time()
 for v in todo:
  try:
   result=eng.score(v["prompt"],capture=True);path=root/"features"/(v["view_id"]+".npz");path.parent.mkdir(parents=True,exist_ok=True)
   np.savez_compressed(path,heads=result.pop("heads").numpy().astype(np.float16),residual=result.pop("residual").numpy().astype(np.float16))
   append(target,dict(v,**result,feature=str(path),status="ok",current_error=(result["ell"]>0)!=bool(v["label"])))
  except ValueError as e:append(target,dict(v,status="unavailable",error=str(e)))
 status(model+"_historical_h3",state="complete",unique_nodes=len(vs),seconds=time.time()-start,
  note="Historical error selection only; new canonical scores and features never mixed with legacy BF16 scores")
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--model",default="qwen");run(p.parse_args().model)
