
"""Fast development-path bias selection for an already fixed correction."""
import numpy as np
def select_dev_bias(node_scores,cases,natural_scores,natural_labels,bref=0.,min_natural_correct=195):
 def get(cid,gid):
  value=node_scores.get(cid+':'+gid)
  if value is None:value=node_scores.get((cid,gid))
  return value
 lower=[];upper=[];missing=[]
 for c in cases:
  ns=[(get(c['case_id'],n['gid']),int(n['label'])) for n in c['gold_path']]
  if any(s is None or not np.isfinite(s) for s,y in ns):
   missing.append(c['case_id']);continue
  lower.append(max([-float(s) for s,y in ns if y==1],default=-np.inf))
  upper.append(min([-float(s) for s,y in ns if y==0],default=np.inf))
 lower=np.asarray(lower);upper=np.asarray(upper)
 valid=lower<upper
 lo=np.sort(lower[valid]);hi=np.sort(upper[valid])
 natural_scores=np.asarray(natural_scores,dtype=float)
 natural_labels=np.asarray(natural_labels,dtype=int)
 if natural_scores.shape!=natural_labels.shape:raise ValueError('natural scores and labels must align')
 finite=np.isfinite(natural_scores)
 natural_scores=natural_scores[finite];natural_labels=natural_labels[finite]
 pos=np.sort(-natural_scores[natural_labels==1])
 neg=np.sort(-natural_scores[natural_labels==0])
 def path_counts(bs):
  bs=np.asarray(bs)
  return np.searchsorted(lo,bs,side='left')-np.searchsorted(hi,bs,side='left')
 def natural_counts(bs):
  bs=np.asarray(bs)
  return np.searchsorted(pos,bs,side='left')+len(neg)-np.searchsorted(neg,bs,side='left')
 bounds=np.unique(np.concatenate([lower,upper,-natural_scores]))
 bounds=bounds[np.isfinite(bounds)]
 # Use interval interiors: nextafter endpoints can change decisions when
 # downstream code regroups ell + delta + b in floating point.
 if len(bounds):
  midpoints=(bounds[:-1]+bounds[1:])/2
  stable=(bounds[1:]-bounds[:-1])>1e-8
  candidates=np.concatenate([midpoints[stable],[bounds[0]-1.,bounds[-1]+1.]])
  if np.min(np.abs(bounds-float(bref)))>1e-8:
   candidates=np.append(candidates,float(bref))
  candidates=np.unique(candidates)
 else:candidates=np.array([float(bref)])

 pc=path_counts(candidates);nc=natural_counts(candidates)
 feasible=nc>=min_natural_correct
 result=dict(path_count=len(cases),available_path_count=len(lower),missing_cases=missing,
  bref=float(bref),bref_correct=int(path_counts([bref])[0]),bref_natural_correct=int(natural_counts([bref])[0]),
  min_natural_correct=int(min_natural_correct),natural_count=len(natural_scores),
  threshold_impossible_cases=int(np.sum(~valid)),candidate_biases=len(candidates))
 if not feasible.any():
  result.update(best_b=None,correct=None,natural_correct=None,eligible=False)
  return result
 eligible=np.flatnonzero(feasible)
 bestpc=np.max(pc[eligible]);eligible=eligible[pc[eligible]==bestpc]
 bestnc=np.max(nc[eligible]);eligible=eligible[nc[eligible]==bestnc]
 closest=np.argmin(np.abs(candidates[eligible]-bref));j=eligible[closest]
 result.update(best_b=float(candidates[j]),correct=int(pc[j]),natural_correct=int(nc[j]),eligible=True)
 return result
# Alias for callers using the descriptive selection name.
select_path_bias=select_dev_bias
if __name__=='__main__':
 import json,sys
 from pathlib import Path
 root=Path('/home/gcx/aster_cr_repair_v2_run_20261108')
 old=Path('/home/gcx/aster_cancel_story_20261109/meeting/real_completion')
 def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
 cases=[c for c in rows(root/'data/path_cases.jsonl') if c['split']=='dev']
 records={r['id']:r for p in (root/'qwen/paths').glob('features_dev_*.jsonl') for r in rows(p) if r.get('status')=='ok'}
 scores={key:r['ell'] for key,r in records.items()}
 meta=json.loads((old/'metadata.json').read_text());a=np.load(old/'coordinates.npz');ix={s:i for i,s in enumerate(a['ids'].tolist())}
 vv=[v for v in meta['views'] if v['split']=='dev' and v['variant']=='full' and v['original']]
 natural_scores=[a['ell'][ix[v['view_id']]] for v in vv];natural_labels=[v['label'] for v in vv]
 out=select_dev_bias(scores,cases,natural_scores,natural_labels,bref=3.0993948961894406)
 sys.path.insert(0,'/home/gcx/TRIAGE-worktrees/aster-cr-20261108')
 from experiments.aster_cr.paths import execute
 b=out['best_b']
 real=sum(execute(c,{r['gid']:r['ell']+b for r in records.values() if r['case_id']==c['case_id']})['correct'] for c in cases)
 assert real==out['correct'],(real,out)
 out['official_execute_correct']=real;out['official_execute_matches']=True
 Path(__file__).with_name('path_tools_verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(out))

