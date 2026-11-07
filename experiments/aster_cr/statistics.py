import numpy as np
from collections import defaultdict
from .common import SEED
def interval(values,sources,draws=2000):
 groups=defaultdict(list)
 for v,s in zip(values,sources):groups[s].append(float(v))
 if not groups:return dict(numerator=0,denominator=0,estimate=None,ci95=None,sources=0)
 sums=np.array([sum(v) for v in groups.values()]);counts=np.array([len(v) for v in groups.values()])
 rng=np.random.default_rng(SEED);boot=[]
 for _ in range(draws):
  idx=rng.integers(len(sums),size=len(sums));boot.append(sums[idx].sum()/counts[idx].sum())
 return dict(numerator=float(sums.sum()),denominator=int(counts.sum()),estimate=float(sums.sum()/counts.sum()),
   ci95=np.quantile(boot,[.025,.975]).tolist(),sources=len(groups))
def pair_metrics(views,pairs,scores,ci=False):
 index={v["view_id"]:v for v in views};result={}
 for kind in ("A","P","C","F","F0"):
  ps=[p for p in pairs if p["kind"]==kind];correct=[];drift=[];direction=[];flip=[];sources=[]
  for p in ps:
   a,b=scores.get(p["left"]),scores.get(p["right"]);sources.append(p["source_id"])
   available=a is not None and b is not None
   correct.append(float(available and (a>0)==bool(p["left_label"]) and (b>0)==bool(p["right_label"])))
   if available:
    drift.append(abs(b-a));flip.append(int((a>0)!=(b>0)))
    direction.append(int((b-a)*(p["right_label"]-p["left_label"])>0) if kind in ("C","F") else int((a>0)==(b>0)))
  r=dict(numerator=int(sum(correct)),denominator=len(ps),rate=float(np.mean(correct)) if ps else None,
     nodes=len({p["group_id"] for p in ps}),sources=len(set(sources)),available=len(drift),
     mean_abs_score_drift=float(np.mean(drift)) if drift else None,
     flips=sum(flip),direction_or_stability_numerator=sum(direction),direction_or_stability_denominator=len(direction))
  if ci:r["interval"]=interval(correct,sources)
  result[kind]=r
 result["macro"]=float(np.mean([result[k]["rate"] for k in ("A","P","C","F")])) if all(result[k]["rate"] is not None for k in ("A","P","C","F")) else None
 return result
