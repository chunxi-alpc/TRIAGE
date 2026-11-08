"""Constrained paired correction on frozen cached r16 features; CPU only."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from pathlib import Path
import json, csv, time
from collections import Counter
import numpy as np
from scipy.optimize import minimize, LinearConstraint, brentq
from scipy.special import expit
from scipy.linalg import null_space

ROOT=Path('/home/gcx/aster_cr_repair_v2_run_20261108')
OUT=Path('/home/gcx/aster_cancel_story_20261109/meeting/evidence_preserving')
OUT.mkdir(parents=True,exist_ok=True)
start=time.time()
def rows(p):
 return [json.loads(s) for s in p.read_text().splitlines() if s.strip()] if p.exists() else []
def dump(p,obj):
 p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def rms(a): return float(np.sqrt(np.mean(np.square(a)))) if len(a) else None
cache=OUT/'cached_coordinates.npz'
metadata=OUT/'cached_metadata.json'
if cache.exists() and metadata.exists():
 with np.load(cache,allow_pickle=False) as f:
  t=f['t'];ell=f['ell'];labels=f['labels'];ids=f['ids'].tolist()
 meta=json.loads(metadata.read_text())
 views=meta['views'];pairs=meta['pairs'];coverage=meta['coverage']
 print('loaded coordinate cache',len(ids),flush=True)
else:
 scores={}
 for f in sorted((ROOT/'qwen/scores').glob('*.jsonl')):
  for r in rows(f):
   if r.get('status')=='ok' and r.get('split') in ('fit','dev'): scores[r['view_id']]=r
 full_views=[v for v in rows(ROOT/'data/views.jsonl') if v['split'] in ('fit','dev') and v['view_id'] in scores]
 byid={v['view_id']:v for v in full_views}
 all_pairs=[p for p in rows(ROOT/'data/pairs.jsonl') if p['split'] in ('fit','dev')]
 pairs=[p for p in all_pairs if p['left'] in byid and p['right'] in byid]
 needed={p[k] for p in pairs for k in ('left','right')}
 needed.update(v['view_id'] for v in full_views if v['variant']=='full')
 ids=sorted(needed);ix={v:i for i,v in enumerate(ids)}
 with np.load(ROOT/'qwen/supports_r16.npz') as f:
  support={k:f[k] for k in f.files}
 raw=np.empty((len(ids),len(support['mu'])),np.float32)
 byfile={}
 for vid in ids:byfile.setdefault(scores[vid]['feature'],[]).append(ix[vid])
 for i,(file,positions) in enumerate(byfile.items()):
  with np.load(file) as a:
   h=a['heads'];value=h.reshape(-1,h.shape[-1])[support['heads']].reshape(-1).astype(np.float32)
  raw[positions]=value
  if (i+1)%2000==0: print('loaded',i+1,'of',len(byfile),flush=True)
 t=(((raw-support['mu'])/support['scale'])@support['N']).astype(np.float64)
 del raw
 ell=np.array([scores[vid]['ell'] for vid in ids],np.float64)
 labels=np.array([byid[vid]['label'] for vid in ids],np.int8)
 keys=('view_id','source_id','group_id','variant','original','label','split','case_id','kind')
 views=[{k:v.get(k) for k in keys} for v in full_views if v['view_id'] in ix]
 coverage={split:{k:dict(expected=sum(p['split']==split and p['kind']==k for p in all_pairs),
  available=sum(p['split']==split and p['kind']==k for p in pairs)) for k in ('A','P','F0','C','F')} for split in ('fit','dev')}
 np.savez_compressed(cache,t=t,ell=ell,labels=labels,ids=np.array(ids))
 dump(metadata,dict(views=views,pairs=pairs,coverage=coverage))
 print('saved coordinate cache',len(ids),flush=True)
ix={v:i for i,v in enumerate(ids)}
byid={v['view_id']:v for v in views}
def pair_arrays(split,kinds):
 pp=[p for p in pairs if p['split']==split and p['kind'] in kinds]
 li=np.array([ix[p['left']] for p in pp],int);ri=np.array([ix[p['right']] for p in pp],int)
 return pp,li,ri,t[ri]-t[li],ell[ri]-ell[li]
fit_ap,fl,fr,X,y=pair_arrays('fit',('A','P'))
fit_cf,cl,cr,E,cy=pair_arrays('fit',('C','F'))
scale=rms(y)
alpha=.01
Q=X.T@X+alpha*np.eye(t.shape[1])
q=X.T@y
w_un=-np.linalg.solve(Q,q)
fit_orig=np.array([ix[v['view_id']] for v in views if v['split']=='fit' and v['variant']=='full' and v.get('original',True)],int)
def calibrate(w):
 base=ell[fit_orig]+t[fit_orig]@w
 yy=labels[fit_orig]
 def deriv(b):return float(np.mean(expit(base+b)-yy))
 lo,hi=-1.,1.
 while deriv(lo)>0:lo*=2
 while deriv(hi)<0:hi*=2
 return float(brentq(deriv,lo,hi,xtol=1e-12))
bias=calibrate(np.zeros(t.shape[1]))
sv=np.linalg.svd(E,compute_uv=False)
Z=null_space(E)
models={'raw':dict(w=np.zeros(t.shape[1]),b=0.,tau=None,solver='baseline'),
        'bias_only':dict(w=np.zeros(t.shape[1]),b=bias,tau=None,solver='baseline'),
        'unconstrained_AP':dict(w=w_un,b=calibrate(w_un),tau=None,solver='closed_form')}
for tau in [0.,.05,.1,.25,.5,1.,2.]:
 delta=tau*scale
 if tau==0:
  w=-Z@np.linalg.solve(Z.T@Q@Z,Z.T@q) if Z.shape[1] else np.zeros(t.shape[1])
  info=dict(success=True,message='exact null-space solution',nit=0,solver='null_space')
 else:
  objscale=max(float(np.mean(y*y)),1.)
  Qn=Q/len(y)/objscale;qn=q/len(y)/objscale
  fun=lambda w:float(.5*w@Qn@w+qn@w)
  jac=lambda w:Qn@w+qn
  w0=w_un*min(1.,delta/max(float(np.max(np.abs(E@w_un))),1e-15))
  res=minimize(fun,w0,jac=jac,method='SLSQP',constraints=[LinearConstraint(E,-delta,delta)],
   options=dict(ftol=1e-13,maxiter=3000,disp=False))
  w=res.x
  info=dict(success=bool(res.success),message=str(res.message),nit=int(res.nit),solver='SLSQP')
 violation=max(0.,float(np.max(np.abs(E@w)))-delta)
 info.update(w=w,b=calibrate(w),tau=tau,delta=delta,constraint_violation=violation,
   fit_max_evidence_correction=float(np.max(np.abs(E@w))))
 models['bounded_'+str(tau)]=info
 print('fit',tau,'success',info['success'],'violation',violation,flush=True)

def evaluate(method,m,split):
 ss=ell+t@m['w']+m['b'];out={}
 ap_pairs,ap_li,ap_ri,apx,apdy=pair_arrays(split,('A','P'))
 ap_scale=rms(apdy)
 for group,kinds in [('A',('A',)),('P',('P',)),('AP',('A','P')),('F0',('F0',)),('C',('C',)),('F',('F',)),('CF',('C','F'))]:
  pp,li,ri,dx,rawdy=pair_arrays(split,kinds)
  dc=dx@m['w'];dy=rawdy+dc
  truthl=np.array([p['left_label'] for p in pp]);truthr=np.array([p['right_label'] for p in pp])
  correct=((ss[li]>0)==truthl)&((ss[ri]>0)==truthr)
  basecorrect=((ell[li]>0)==truthl)&((ell[ri]>0)==truthr)
  biascorrect=((ell[li]+bias>0)==truthl)&((ell[ri]+bias>0)==truthr)
  mm=dict(n=len(pp),sources=len({p['source_id'] for p in pp}),raw_difference_rms=rms(rawdy),
   difference_rms=rms(dy),difference_mean=float(np.mean(dy)),
   correction_rms=rms(dc),correction_abs_median=float(np.median(np.abs(dc))),
   correction_abs_p95=float(np.quantile(np.abs(dc),.95)),correction_abs_max=float(np.max(np.abs(dc))),
   correction_rms_over_ap_rms=rms(dc)/ap_scale,
   correction_abs_median_over_ap_rms=float(np.median(np.abs(dc)))/ap_scale,
   correction_abs_p95_over_ap_rms=float(np.quantile(np.abs(dc),.95))/ap_scale,
   correction_abs_max_over_ap_rms=float(np.max(np.abs(dc)))/ap_scale,
   paired_correct=int(correct.sum()),paired_accuracy=float(correct.mean()),
   corrected_pairs_vs_raw=int(np.sum(correct&~basecorrect)),new_error_pairs_vs_raw=int(np.sum(~correct&basecorrect)),
   corrected_pairs_vs_bias=int(np.sum(correct&~biascorrect)),new_error_pairs_vs_bias=int(np.sum(~correct&biascorrect)))
  if group in ('A','P','AP','F0'):
   mm.update(variance_reduction=1-float(np.var(dy))/float(np.var(rawdy)),
    prediction_r2=1-float(dy@dy)/float(np.sum((rawdy-rawdy.mean())**2)))
  else:
   direction=truthr-truthl;original_signed=direction*rawdy;signed=direction*dy
   mm.update(mean_evidence_retention=float(np.mean(signed)/np.mean(original_signed)),
    raw_response_mean=float(np.mean(original_signed)),response_mean=float(np.mean(signed)),
    response_direction_degraded=int(np.sum((original_signed>0)&(signed<=0))),
    response_direction_improved=int(np.sum((original_signed<=0)&(signed>0))),
    raw_response_positive=int(np.sum(original_signed>0)),corrected_response_positive=int(np.sum(signed>0)),
    exceeds_fit_budget_count=(int(np.sum(np.abs(dc)>m['delta']+1e-9)) if m.get('delta') is not None else None),
    exceeds_fit_budget_fraction=(float(np.mean(np.abs(dc)>m['delta']+1e-9)) if m.get('delta') is not None else None),
    large_raw_response_count=(int(np.sum(original_signed>m['delta'])) if m.get('delta') is not None else None),
    large_raw_response_direction_degraded=(int(np.sum((original_signed>m['delta'])&(signed<=0))) if m.get('delta') is not None else None))
  out[group]=mm
 for group,fn in [('natural',lambda v:v.get('original',True)),('constructed',lambda v:not v.get('original',True)),('all_full',lambda v:True)]:
  inds=np.array([ix[v['view_id']] for v in views if v['split']==split and v['variant']=='full' and fn(v)],int)
  cor=(ss[inds]>0)==labels[inds];rawcor=(ell[inds]>0)==labels[inds];bcor=(ell[inds]+bias>0)==labels[inds]
  out[group]=dict(n=len(inds),correct=int(cor.sum()),accuracy=float(cor.mean()),
   corrected_vs_raw=int(np.sum(cor&~rawcor)),new_errors_vs_raw=int(np.sum(~cor&rawcor)),
   corrected_vs_bias=int(np.sum(cor&~bcor)),new_errors_vs_bias=int(np.sum(~cor&bcor)))
 return out
result=dict(data_version='repair-v2',model='Qwen3-8B',gpu_used=False,test_used=False,
 protocol='Existing frozen r16 support; objective sum AP(delta_logit+delta_trace@w)^2+.01||w||^2 subject to per-fit-CF |delta_trace@w|<=tau*fit-AP-raw-RMS. tau=.1 fixed before dev evaluation in this run. b fitted to natural fit nodes after fixing w.',
 primary_tau=.1,rank=int(t.shape[1]),alpha=alpha,fit_ap_n=len(fit_ap),fit_cf_n=len(fit_cf),
 fit_calibration_n=len(fit_orig),fit_ap_raw_rms=scale,
 constraint_rank=int(np.linalg.matrix_rank(E)),constraint_null_dimension=int(Z.shape[1]),
 constraint_singular_values=sv.tolist(),coverage=coverage,methods={})
source_fit={v['source_id'] for v in views if v['split']=='fit'}
source_dev={v['source_id'] for v in views if v['split']=='dev'}
result['source_overlap']=dict(fit=len(source_fit),dev=len(source_dev),shared=len(source_fit&source_dev),
 shared_ids=sorted(source_fit&source_dev))
summary=[]
pairrows=[]
weights={}
for name,m in models.items():
 mr={k:v for k,v in m.items() if k!='w'}
 mr['splits']={split:evaluate(name,m,split) for split in ('fit','dev')}
 result['methods'][name]=mr
 weights[name]=m['w'];weights[name+'_b']=m['b']
 for split,groups in mr['splits'].items():
  for group,vals in groups.items():
   summary.append(dict(method=name,tau=m['tau'],split=split,group=group,**vals))
 ss=ell+t@m['w']+m['b']
 for p in pairs:
  if p['split']!='dev':continue
  li,ri=ix[p['left']],ix[p['right']]
  pairrows.append(dict(method=name,pair_id=p['pair_id'],kind=p['kind'],source_id=p['source_id'],
   original_left=float(ell[li]),original_right=float(ell[ri]),corrected_left=float(ss[li]),
   corrected_right=float(ss[ri]),correction_difference=float((t[ri]-t[li])@m['w']),
   original_difference=float(ell[ri]-ell[li]),corrected_difference=float(ss[ri]-ss[li]),
   left_label=p['left_label'],right_label=p['right_label']))
result['runtime_seconds']=time.time()-start
dump(OUT/'metrics.json',result)
for name,rr in [('summary.csv',summary),('dev_pair_results.csv',pairrows)]:
 fields=list(dict.fromkeys(k for r in rr for k in r))
 with (OUT/name).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rr)
np.savez_compressed(OUT/'weights.npz',**weights)
brief=dict(runtime=result['runtime_seconds'],fit_ap_raw_rms=scale,fit_cf_rank=result['constraint_rank'],
 null_dim=result['constraint_null_dimension'],source_overlap=result['source_overlap'],
 rows=[])
for name,mr in result['methods'].items():
 d=mr['splits']['dev']
 brief['rows'].append(dict(method=name,AP_RMS=d['AP']['difference_rms'],VR=d['AP']['variance_reduction'],
  R2=d['AP']['prediction_r2'],CF_ER=d['CF']['mean_evidence_retention'],
  CF_correction_rms=d['CF']['correction_rms'],CF_correction_p95=d['CF']['correction_abs_p95'],
  CF_correction_max=d['CF']['correction_abs_max'],CF_degraded=d['CF']['response_direction_degraded'],
  CF_improved=d['CF']['response_direction_improved'],natural=d['natural']))
dump(OUT/'brief.json',brief)
print(json.dumps(brief,ensure_ascii=True),flush=True)
