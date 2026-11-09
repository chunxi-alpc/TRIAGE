
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS","2")
os.environ.setdefault("OMP_NUM_THREADS","2")
import json,sys,csv,pickle,time,argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.optimize import minimize,LinearConstraint,brentq
from scipy.special import expit
OUT=Path(__file__).resolve().parent
ROOT=Path('/home/gcx/aster_cr_repair_v2_run_20261108')
OLD=OUT.parent/'evidence_preserving'
sys.path.insert(0,'/home/gcx/TRIAGE-worktrees/aster-cr-20261108')
from experiments.aster_cr.paths import execute
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()] if Path(p).exists() else []
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def rms(x):return float(np.sqrt(np.mean(x*x))) if len(x) else None
def main(linear_only=False):
 start=time.time();a=np.load(OUT/'coordinates.npz');ids=a['ids'].tolist();ix={v:i for i,v in enumerate(ids)}
 ell=a['ell'];meta=json.loads((OUT/'metadata.json').read_text());pairs=meta['pairs'];views=meta['views']
 test=[v for v in views if v['split']=='test'];ps=[p for p in pairs if p['split']=='test']
 fitap=[p for p in pairs if p['split']=='fit' and p['kind'] in ('A','P')]
 fitcf=[p for p in pairs if p['split']=='fit' and p['kind'] in ('C','F')]
 al=np.array([ix[p['left']] for p in fitap]);ar=np.array([ix[p['right']] for p in fitap])
 cl=np.array([ix[p['left']] for p in fitcf]);cr=np.array([ix[p['right']] for p in fitcf])
 y=ell[ar]-ell[al];fit_scale=rms(y);budget=.1*fit_scale
 nv=[v for v in views if v['split']=='fit' and v['variant']=='full' and v['original']]
 ni=np.array([ix[v['view_id']] for v in nv]);ny=np.array([v['label'] for v in nv])
 def calibrate(c):
  f=lambda b:float(np.mean(expit(ell[ni]+c[ni]+b)-ny))
  lo,hi=-1.,1.
  while f(lo)>0:lo*=2
  while f(hi)<0:hi*=2
  return float(brentq(f,lo,hi))
 scores={};info={};linearweights={}
 def add(name,s,**kw):scores[name]=np.asarray(s,dtype='float64');info[name]=kw
 old=np.load(OLD/'weights.npz')
 def saved(name,key,basis='r16',file=old,remove_b=False,tau=None):
  w=file[key];b=0. if remove_b else float(file[key+'_b'])
  correction=a[basis]@w
  add(name,ell+correction+b,basis=basis,weights_source='existing',b=b,tau=tau,parameters=len(w)+int(not remove_b),
   fit_max_evidence_correction=float(np.max(np.abs(correction[cr]-correction[cl]))))
 saved('raw','raw');saved('no_w','bias_only')
 saved('linear_output','bounded_0.1',tau=.1)
 saved('no_b','bounded_0.1',remove_b=True,tau=.1)
 saved('no_CF','unconstrained_AP')
 for tau in (0.,.05,.25,.5,1.,2.):saved('budget_'+str(tau),'bounded_'+str(tau),tau=tau)
 aug=np.load(OLD/'augmented_weights.npz')
 for tau in (0.,.1,.25):saved('r32_budget_'+str(tau),'bounded_'+str(tau),basis='r32',file=aug,tau=tau)
 saved('r32_no_CF','unconstrained_AP',basis='r32',file=aug)
 for name,basis in [('r8','r8'),('unorthogonal_U','U'),('signal_only_S','S'),('random_subspace','random')]:
  t=a[basis];X=t[ar]-t[al];E=t[cr]-t[cl]
  Q=X.T@X+.01*np.eye(t.shape[1]);q=X.T@y
  n=len(y)*max(np.mean(y*y),1)
  result=minimize(lambda w:float((.5*w@Q@w+q@w)/n),np.zeros(t.shape[1]),
   jac=lambda w:(Q@w+q)/n,method='SLSQP',constraints=[LinearConstraint(E,-budget,budget)],
   options=dict(ftol=1e-13,maxiter=3000))
  w=result.x;c=t@w;b=calibrate(c);maximum=float(np.max(np.abs(E@w)))
  if not result.success or maximum>budget+1e-7:raise RuntimeError(name+': '+result.message)
  add(name,ell+c+b,basis=basis,b=b,tau=.1,parameters=len(w)+1,solver=result.message,fit_max_evidence_correction=maximum)
  linearweights[name]=w;linearweights[name+'_b']=np.array(b)
 np.savez_compressed(OUT/'new_linear_weights.npz',**linearweights)
 # Frozen legacy settings: scores and paths are consumed directly.
 frozen=json.loads((ROOT/'qwen/frozen.json').read_text())
 for name in ('leace','iti','closed_form'):
  s=np.full(len(ids),np.nan)
  for p in (ROOT/'qwen/comparators').glob('test_*.jsonl'):
   for r in rows(p):
    if r['view_id'] in ix and name in r['scores']:s[ix[r['view_id']]]=r['scores'][name]
  add(name,s,source='existing per-input scores',configuration='qwen/comparators/settings.json')
 for name in ('ordinary_r8','ordinary_matched'):
  s=np.full(len(ids),np.nan)
  for r in rows(ROOT/('qwen/readout_scores/'+name+'_test.jsonl')):
   if r['view_id'] in ix:s[ix[r['view_id']]]=r['ell']
  add(name,s,source='existing frozen readout scores',configuration=name)
 for family,name in frozen['baseline_families'].items():
  s=np.full(len(ids),np.nan)
  for r in rows(ROOT/('qwen/baselines/'+name+'_test.jsonl')):
   if r['view_id'] in ix:s[ix[r['view_id']]]=r['ell']
  add(family,s,source='existing selected training baseline',configuration=name)
 if not linear_only:
  for kind in ('mlp_output','linear_representation','mlp_representation'):
   for seed in (11,22,33):
    name=kind+'_seed'+str(seed);d=np.load(OUT/(name+'.npz'))
    add(name,ell+d['delta']+float(d['b']),**json.loads((OUT/(name+'.json')).read_text()))
 sources=sorted({v['source_id'] for v in test});sx={s:i for i,s in enumerate(sources)}
 rng=np.random.default_rng(11)
 boot=rng.multinomial(len(sources),np.full(len(sources),1/len(sources)),size=2000).astype('float64')
 def sums(values,ss):
  values=np.asarray(values,dtype='float64')
  if values.ndim==1:values=values[:,None]
  out=np.zeros((len(sources),values.shape[1]))
  for i,s in enumerate(ss):out[sx[s]]+=values[i]
  return out
 def ci(values):
  values=np.asarray(values);values=values[np.isfinite(values)]
  return np.quantile(values,[.025,.975]).tolist() if len(values) else None
 def boot_mean(v,ss):
  agg=sums(np.column_stack([np.ones(len(v)),v]),ss);b=boot@agg
  with np.errstate(divide='ignore',invalid='ignore'):z=b[:,1]/b[:,0]
  return ci(z)
 allmetrics={};flat=[];pairrows=[]
 testap=[p for p in ps if p['kind'] in ('A','P')]
 apraw=np.array([ell[ix[p['right']]]-ell[ix[p['left']]] for p in testap]);ap_scale=rms(apraw)
 groupdefs=[('A',('A',)),('P',('P',)),('AP',('A','P')),('F0',('F0',)),('C',('C',)),('F',('F',)),('CF',('C','F'))]
 for name,s in scores.items():
  groups={}
  for group,kinds in groupdefs:
   expected=[p for p in ps if p['kind'] in kinds]
   pp=[p for p in expected if np.isfinite(s[ix[p['left']]]) and np.isfinite(s[ix[p['right']]])]
   if not pp:groups[group]=dict(n=0,expected=len(expected),missing=len(expected));continue
   li=np.array([ix[p['left']] for p in pp]);ri=np.array([ix[p['right']] for p in pp])
   raw=ell[ri]-ell[li];dy=s[ri]-s[li];dc=dy-raw
   yl=np.array([p['left_label'] for p in pp]);yr=np.array([p['right_label'] for p in pp])
   cor=((s[li]>0)==yl)&((s[ri]>0)==yr);basecor=((ell[li]>0)==yl)&((ell[ri]>0)==yr)
   ss=[p['source_id'] for p in pp]
   vals=np.column_stack([np.ones(len(pp)),raw,raw**2,dy,dy**2,dc**2,cor.astype(float),(cor.astype(float)-basecor.astype(float))])
   agg=sums(vals,ss);bb=boot@agg
   with np.errstate(divide='ignore',invalid='ignore'):
    brms=np.sqrt(bb[:,4]/bb[:,0]);bvr=1-(bb[:,4]/bb[:,0]-(bb[:,3]/bb[:,0])**2)/(bb[:,2]/bb[:,0]-(bb[:,1]/bb[:,0])**2)
    bpa=bb[:,6]/bb[:,0];bdiff=bb[:,7]/bb[:,0];bcrms=np.sqrt(bb[:,5]/bb[:,0])
   mm=dict(n=len(pp),expected=len(expected),missing=len(expected)-len(pp),sources=len(set(ss)),raw_difference_rms=rms(raw),
    difference_rms=rms(dy),correction_rms=rms(dc),correction_rms_over_ap_rms=rms(dc)/ap_scale,
    correction_abs_max=float(np.max(np.abs(dc))),paired_correct=int(cor.sum()),paired_accuracy=float(cor.mean()),
    paired_accuracy_ci=ci(bpa),paired_accuracy_difference_vs_raw=float(np.mean(cor.astype(float)-basecor.astype(float))),
    paired_accuracy_difference_vs_raw_ci=ci(bdiff),difference_rms_ci=ci(brms),correction_rms_ci=ci(bcrms),
    corrected_pairs_vs_raw=int(np.sum(cor&~basecor)),new_error_pairs_vs_raw=int(np.sum(~cor&basecor)))
   if group in ('A','P','AP','F0'):
    mm.update(variance_reduction=1-float(np.var(dy))/float(np.var(raw)),variance_reduction_ci=ci(bvr))
   else:
    direction=yr-yl;rs=direction*raw;ds=direction*dy
    aa=sums(np.column_stack([rs,ds,((rs>0)&(ds<=0)).astype(float)]),ss);bb2=boot@aa
    with np.errstate(divide='ignore',invalid='ignore'):er=bb2[:,1]/bb2[:,0]
    mm.update(mean_evidence_retention=float(np.mean(ds)/np.mean(rs)),mean_evidence_retention_ci=ci(er),
     response_direction_degraded=int(np.sum((rs>0)&(ds<=0))),response_direction_improved=int(np.sum((rs<=0)&(ds>0))),
     exceeds_fit_budget_count=int(np.sum(np.abs(dc)>budget+1e-7)),
     exceeds_fit_budget_fraction=float(np.mean(np.abs(dc)>budget+1e-7)))
   groups[group]=mm;flat.append(dict(method=name,group=group,**mm))
   # Per-pair rows are saved once, without duplicating aggregate groups.
   if group in ('A','P','C','F','F0'):
    for j,p in enumerate(pp):pairrows.append(dict(method=name,pair_id=p['pair_id'],source_id=p['source_id'],kind=p['kind'],
      left_score=float(s[li[j]]),right_score=float(s[ri[j]]),original_difference=float(raw[j]),
      correction_difference=float(dc[j]),corrected_difference=float(dy[j]),paired_correct=bool(cor[j])))
  vv=[v for v in test if v['variant']=='full' and v['original']]
  valid=[v for v in vv if np.isfinite(s[ix[v['view_id']]])]
  inds=np.array([ix[v['view_id']] for v in valid]);lab=np.array([v['label'] for v in valid])
  cor=(s[inds]>0)==lab;bc=(ell[inds]>0)==lab;ss=[v['source_id'] for v in valid]
  groups['natural']=dict(n=len(valid),expected=len(vv),missing=len(vv)-len(valid),correct=int(cor.sum()),
   accuracy=float(cor.mean()),accuracy_ci=boot_mean(cor,ss),difference_vs_raw=float(np.mean(cor.astype(float)-bc.astype(float))),
   difference_vs_raw_ci=boot_mean(cor.astype(float)-bc.astype(float),ss))
  allmetrics[name]=dict(settings=info[name],groups=groups)
 cases=[c for c in rows(ROOT/'data/path_cases.jsonl') if c['split']=='test']
 pathrecords=meta['paths'];records=defaultdict(dict)
 for r in pathrecords:records[r['case_id']][r['gid']]=ix['path|'+r['id']]
 pathresults={name:[] for name in scores}
 newnames=[name for name in scores if info[name].get('source') is None]
 for name in newnames:
  s=scores[name]
  for c in cases:
   ns={gid:float(s[i]) for gid,i in records[c['case_id']].items() if np.isfinite(s[i])}
   pathresults[name].append(dict(case_id=c['case_id'],source_id=c['source_id'],**execute(c,ns)))
 for name in ('leace','iti','closed_form'):
  bycase={}
  for p in (ROOT/'qwen/comparators').glob('paths_test_*.jsonl'):
   for r in rows(p):
    if name in r['methods']:bycase[r['case_id']]=dict(case_id=r['case_id'],source_id=r['source_id'],**r['methods'][name])
  pathresults[name]=list(bycase.values())
 for name in ('ordinary_r8','ordinary_matched'):
  pathresults[name]=rows(ROOT/('qwen/path_results/'+name+'_test.jsonl'))
 for family,name in frozen['baseline_families'].items():pathresults[family]=rows(ROOT/('qwen/baselines/'+name+'_paths_test.jsonl'))
 baseres={r['case_id']:r for r in pathresults['raw']}
 for name,rr in pathresults.items():
  bycase={r['case_id']:r for r in rr};valid=[c for c in cases if c['case_id'] in bycase]
  cor=np.array([bycase[c['case_id']]['correct'] for c in valid]);bc=np.array([baseres[c['case_id']]['correct'] for c in valid])
  ss=[c['source_id'] for c in valid]
  allmetrics[name]['path']=dict(n=len(valid),expected=len(cases),missing=len(cases)-len(valid),correct=int(cor.sum()),accuracy=float(cor.mean()) if len(cor) else None,
   accuracy_ci=boot_mean(cor,ss),difference_vs_raw=float(np.mean(cor.astype(float)-bc.astype(float))) if len(cor) else None,
   difference_vs_raw_ci=boot_mean(cor.astype(float)-bc.astype(float),ss),
   unavailable_scores=sum(r.get('error')=='unavailable_score' for r in rr))
 with (OUT/'path_results.jsonl').open('w') as f:
  for name,rr in pathresults.items():
   for r in rr:f.write(json.dumps(dict(method=name,**r))+'\n')
 oldraw={r['case_id']:r for r in rows(ROOT/'qwen/path_results/full_test.jsonl')}
 path_mismatch=sum(r['case_id'] in oldraw and (r['correct']!=oldraw[r['case_id']]['correct'] or r['path']!=oldraw[r['case_id']]['path'] or r['leaf']!=oldraw[r['case_id']]['leaf']) for r in pathresults['raw'])
 result=dict(protocol='held-out test previously used in legacy evaluations; no current test fitting or model selection',
  fit_AP=len(fitap),fit_CF=len(fitcf),fit_budget=budget,test_AP_raw_rms=ap_scale,test_sources=len(sources),
  coverage=meta['coverage'],view_coverage=meta['view_coverage'],path_cases=len(cases),
  uncertainty='2000 source-cluster bootstrap replicates, seed 11, percentile 95% CIs; sources sampled jointly across methods; undefined sparse-group replicates omitted',
  path_score_protocol='Each method traverses its own decisions from root; exact gold decisions and gold terminal leaf required; all reachable unedited nodes cached',
  raw_path_comparison=dict(existing_cases=len(oldraw),new_cases=len(pathresults['raw']),mismatches=path_mismatch),
  methods=allmetrics,seconds=time.time()-start)
 dump(OUT/'metrics.json',result)
 for filename,rr in [('summary.csv',flat),('test_pair_results.csv',pairrows)]:
  fields=list(dict.fromkeys(k for r in rr for k in r))
  with (OUT/filename).open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rr)
 # Save all scores so metrics and paths can be recomputed without refitting.
 np.savez_compressed(OUT/'evaluated_scores.npz',ids=np.array(ids),**scores)
 dump(OUT/'verification.json',dict(pair_coverage=meta['coverage'],view_coverage=meta['view_coverage'],
  raw_paths=result['raw_path_comparison'],fit_budget=budget,
  fit_constraints={name:dict(maximum=i.get('fit_max_evidence_correction'),tau=i.get('tau')) for name,i in info.items() if i.get('fit_max_evidence_correction') is not None},
  nonlinear_zero_edits={name:i.get('zero_correction_max_abs_error') for name,i in info.items() if 'zero_correction_max_abs_error' in i},
  missing_results={name:dict(groups={g:m.get('missing',0) for g,m in v['groups'].items()},paths=v['path']['missing']) for name,v in allmetrics.items()},
  existing_weights_reused=True,backbone_forwards=0,test_fitting=False))
 dump(OUT/'evaluation_done.json',dict(state='complete',methods=len(scores),nonlinear_included=not linear_only))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--linear-only',action='store_true');main(p.parse_args().linear_only)

