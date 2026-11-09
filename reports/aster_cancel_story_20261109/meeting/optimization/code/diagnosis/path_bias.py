
"""Fit/dev-only scalar threshold search for fixed correction models.

Inputs are scores before bias. A complete correct gold path is equivalent to
all of its gold decisions being correct; off-gold scores cannot rescue a wrong
decision. This matches autonomous traversal when all reachable nodes are cached.
"""
import numpy as np
def select_path_bias(cases,node_scores,natural_scores=None,natural_labels=None,fit_scores=None,fit_labels=None):
 lower=[];upper=[];available=[];missing=[]
 for c in cases:
  ns=[(node_scores.get(c['case_id']+':'+n['gid']),int(n['label'])) for n in c['gold_path']]
  if any(s is None or not np.isfinite(s) for s,y in ns):
   missing.append(c['case_id']);continue
  lower.append(max([-float(s) for s,y in ns if y==1],default=-np.inf))
  upper.append(min([-float(s) for s,y in ns if y==0],default=np.inf))
  available.append(c['case_id'])
 lower=np.array(lower);upper=np.array(upper)
 bounds=np.unique(np.concatenate([lower,upper]));bounds=bounds[np.isfinite(bounds)]
 if not len(bounds):return dict(b=0.,correct=0,n=0,missing=missing)
 candidates=np.unique(np.concatenate([[0.],bounds,(bounds[1:]+bounds[:-1])/2,[bounds[0]-1,bounds[-1]+1]]))
 def node_correct(s,y,b):
  return int(np.sum((np.asarray(s)+b>0)==np.asarray(y))) if s is not None and y is not None else 0
 def scores(b):
  return (int(np.sum((b>lower)&(b<=upper))),node_correct(natural_scores,natural_labels,b),node_correct(fit_scores,fit_labels,b),-abs(b))
 b=float(max(candidates,key=scores))
 value=scores(b)
 return dict(b=b,correct=value[0],n=len(available),natural_correct=value[1],fit_natural_correct=value[2],
  missing=missing,selection='maximum dev paths, then dev natural nodes, then fit natural nodes, then smaller absolute b',
  impossible_threshold_cases=int(np.sum(lower>=upper)))

