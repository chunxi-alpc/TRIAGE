"""Natural-case errors, output forms and numerical response curves stay separate."""
import argparse
from collections import defaultdict
import numpy as np
from .common import *
from .features import Features
from .statistics import interval
def report(model,splits=("fit","dev")):
 f=Features(model,splits);nodes=defaultdict(dict)
 for v in f.views:nodes[v["group_id"]][v["variant"]]=v
 e4={x["group_id"] for x in rows(ROOT/"data/E4_candidates.jsonl")}
 natural=[];changes=[];curves=[];forms=[]
 for nid,vs in nodes.items():
  if "full" not in vs:continue
  v=vs["full"];ell=f.ell(v["view_id"]);wrong=(ell>0)!=bool(v["label"])
  r=dict(group_id=nid,source_id=v["source_id"],split=v["split"],gold=v["label"],original_error=wrong)
  r["category"]="E4_candidate" if nid in e4 else "correct" if not wrong else "unresolved"
  for name in ("remove_action","neutral_background","both_clean","condition_ancestors"):
   if name in vs:
    ve=vs[name];correct=(f.ell(ve["view_id"])>0)==bool(v["label"])
    changes.append(dict(group_id=nid,source_id=v["source_id"],split=v["split"],operation=name,
      original_error=wrong,clean_correct=correct,corrected=wrong and correct,new_error=not wrong and not correct))
    if name=="both_clean" and wrong and correct and nid not in e4:r["category"]="E1"
  directions=[]
  for p in f.pairs:
   if p["group_id"]==nid and p["kind"]=="F":
    directions.append((f.ell(p["right"])-f.ell(p["left"]))*(p["right_label"]-p["left_label"])>0)
  if r["category"]=="unresolved" and directions:
   r["category"]="E2_candidate" if np.mean(directions)>.5 else "E3_candidate"
  r["direction_correct_rate"]=float(np.mean(directions)) if directions else None
  natural.append(r)
  fs={name:f.ell(ve["view_id"]) for name,ve in vs.items() if name in ("full","order_no_yes","map_ab","map_ba","equivalent")}
  forms.append(dict(group_id=nid,source_id=v["source_id"],split=v["split"],gold=v["label"],scores=fs,
     semantic_flips={k:int((s>0)!=(ell>0)) for k,s in fs.items()},
     symmetric_yes_no=(fs.get("full",0)+fs.get("order_no_yes",0))/2 if all(k in fs for k in ("full","order_no_yes")) else None,
     symmetric_AB=(fs.get("map_ab",0)+fs.get("map_ba",0))/2 if all(k in fs for k in ("map_ab","map_ba")) else None,
     forward_cost=dict(single_prompt=1,symmetric_yes_no=2,symmetric_AB=2)))
  grid=defaultdict(list)
  for ve in vs.values():
   if ve.get("threshold") is not None:grid[(ve["field"],ve["threshold"])].append(ve)
  for (field,t),vv in grid.items():
   vv=sorted(vv,key=lambda q:q["value"]);x=np.array([q["value"] for q in vv]);y=np.array([f.ell(q["view_id"]) for q in vv])
   labels=np.array([q["label"] for q in vv]);step=vv[0]["step"]
   slope=float(np.polyfit(x,y,1)[0]) if len(x)>1 and np.ptp(x)>0 else 0.
   crossings=[]
   if abs(slope)>1e-8:
    for i in range(len(x)-1):
     if (y[i]>0)!=(y[i+1]>0) and abs(y[i+1]-y[i])>1e-8:crossings.append(float(x[i]-y[i]*(x[i+1]-x[i])/(y[i+1]-y[i])))
   boundary=min(crossings,key=lambda b:abs(b-t)) if crossings else None
   sign=int(np.sign(labels[-1]-labels[0])) if len(labels)>1 else 0
   curves.append(dict(group_id=nid,source_id=v["source_id"],split=v["split"],field=field,threshold=t,step=step,
    values=x.tolist(),scores=y.tolist(),labels=labels.tolist(),sensitivity=sign*slope if sign else None,
    boundary=boundary,crossings=crossings,threshold_tracked=abs(boundary-t)<=step if boundary is not None else None,
    boundary_status="estimated" if boundary is not None else "not_estimable"))
 out=ROOT/model/"diagnostics"/("test" if splits==("test",) else "development")
 lines(out/"p0_nodes.jsonl",natural);lines(out/"cleaning.jsonl",changes);lines(out/"formats.jsonl",forms);lines(out/"thresholds.jsonl",curves)
 summary={}
 for split in splits:
  rs=[r for r in natural if r["split"]==split];chg=[r for r in changes if r["split"]==split]
  summary[split]=dict(original_nodes=len(rs),original_errors=sum(r["original_error"] for r in rs),
   error_rate=interval([r["original_error"] for r in rs],[r["source_id"] for r in rs]),
   categories=dict(__import__("collections").Counter(r["category"] for r in rs)),
   cleaning={name:dict(available=sum(r["operation"]==name for r in chg),
    corrected=sum(r["corrected"] for r in chg if r["operation"]==name),new_errors=sum(r["new_error"] for r in chg if r["operation"]==name),
    original_error_denominator=sum(r["original_error"] for r in rs)) for name in ("remove_action","neutral_background","both_clean","condition_ancestors")})
 summary["interpretation"]="E2/E3 are candidates; E2 needs readout confirmation. Original benchmark cases and constructed boundary cases have separate denominators."
 write(out/"p0.json",summary)
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--model",default="qwen");p.add_argument("--test",action="store_true");a=p.parse_args()
 report(a.model,("test",) if a.test else ("fit","dev"))
