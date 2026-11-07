import argparse,pickle
import numpy as np
from .common import *
from .features import Features
from .readout import predict
from .statistics import pair_metrics,interval
def evaluate(model):
 frozen=read(ROOT/model/"frozen.json")
 if not frozen.get("frozen"):raise RuntimeError("Not frozen")
 f=Features(model,("test",));views=[v for v in rows(ROOT/"data/views.jsonl") if v["split"]=="test" and v["kind"] in {"original","A","P","C","grid"}]
 pairs=[p for p in rows(ROOT/"data/pairs.jsonl") if p["split"]=="test"];results={};all_scores={}
 for name in frozen["readouts"]:
  with (ROOT/model/"readouts"/(name+".pkl")).open("rb") as file:obj=pickle.load(file)
  available=[v for v in views if v["view_id"] in f.scores]
  raw=f.matrix(available,obj["heads"]) if obj.get("heads") else None
  values=predict(obj,np.array([f.ell(v["view_id"]) for v in available]),raw)
  scores={v["view_id"]:float(s) for v,s in zip(available,values)};all_scores[name]=scores
  results[name]=pair_metrics(views,pairs,scores,ci=True)
  lines(ROOT/model/"readout_scores"/(name+"_test.jsonl"),[dict(view_id=k,ell=s) for k,s in scores.items()])
 # Paired source resampling directly on method-minus-Full differences.
 for name,scores in all_scores.items():
  differences={}
  for kind in ("A","P","C","F","F0"):
   ps=[p for p in pairs if p["kind"]==kind];ds=[]
   for p in ps:
    def correct(ss):
     a,b=ss.get(p["left"]),ss.get(p["right"])
     return int(a is not None and b is not None and (a>0)==bool(p["left_label"]) and (b>0)==bool(p["right_label"]))
    ds.append(correct(scores)-correct(all_scores["full"]))
   differences[kind]=interval(ds,[p["source_id"] for p in ps])
  results[name]["difference_from_full"]=differences
 write(ROOT/model/"test_pairs.json",results)
 return results
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--model",default="qwen");evaluate(p.parse_args().model)
