
"""One CPU convex residual correction, fitted only on existing fit data."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
import argparse,json,time,sys
from collections import Counter
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize,LinearConstraint,brentq
OUT=Path('/home/gcx/aster_hard_optimization_20261109/linear')
BASE=Path('/home/gcx/aster_next_20261109')
OLD=Path('/home/gcx/aster_cancel_story_20261109/meeting/real_completion')
AUG=Path('/home/gcx/aster_hard_optimization_20261109/fit_augmentation')
sys.path.insert(0,'/home/gcx/TRIAGE-worktrees/aster-cr-20261108')
from experiments.aster_cr.paths import execute
def dump(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');q.replace(p)
def source_weights(sources,multiplier=None):
 sources=list(sources);counts=Counter(sources);weights=np.ones(len(sources)) if multiplier is None else np.asarray(multiplier,dtype=float)
 totals={s:float(weights[np.array([x==s for x in sources])].sum()) for s in counts}
 return np.array([weights[j]/totals[s]/len(counts) for j,s in enumerate(sources)])
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def main():
 started=time.time();OUT.mkdir(parents=True,exist_ok=True)
 manifest=json.loads((BASE/'frozen_methods.json').read_text())
 cfg=manifest['methods']['optimized_balanced_fit_calibration']
 cached=np.load(cfg['prediction_file']);ids=cached['ids'].tolist();ix={s:i for i,s in enumerate(ids)}
 shared=np.load('/home/gcx/aster_optimization_20261109/shared/features.npz')
 six={s:i for i,s in enumerate(shared['ids'].tolist())};positions=[six[s] for s in ids]
 ell=shared['ell'][positions].astype('float64');t=shared['r32'][positions].astype('float64')
 # The cached raw ensemble output is scaled once by frozen .436452 gain.
 base_delta=cached['delta'].astype('float64')*float(cfg['gain'])
 if 'final_delta' in cached.files and not np.allclose(base_delta,cached['final_delta'],atol=1e-7,rtol=1e-7):raise ValueError('Frozen base gain mismatch')
 meta=json.loads((OLD/'metadata.json').read_text())
 views=[v for v in meta['views'] if v['split'] in ('fit','dev')]
 if any(v['view_id'] not in ix for v in views):raise ValueError('fit/dev views missing from frozen predictions')
 pairs=[p for p in meta['pairs'] if p['split'] in ('fit','dev')]
 def indices(split,kinds):
  pp=[p for p in pairs if p['split']==split and p['kind'] in kinds]
  return pp,np.array([ix[p['left']] for p in pp]),np.array([ix[p['right']] for p in pp])
 ap,al,ar=indices('fit',('A','P'));cf,cl,cr=indices('fit',('C','F'))
 natural={sp:[v for v in views if v['split']==sp and v['variant']=='full' and v['original']] for sp in ('fit','dev')}
 ni={sp:np.array([ix[v['view_id']] for v in vv]) for sp,vv in natural.items()}
 ny={sp:np.array([v['label'] for v in vv],dtype=float) for sp,vv in natural.items()}
 aug=np.load(AUG/'features.npz');at=aug['r32'].astype('float64');ae=aug['ell'].astype('float64');ad=aug['base_balanced_delta'].astype('float64');ay=aug['labels'].astype(float)
 old_sources=[v['source_id'] for v in natural['fit']];new_sources=aug['source_ids'].tolist()
 if not set(new_sources)<=set(old_sources):raise ValueError('Augmentation contains non-fit source')
 if abs(float(aug['base_balanced_b'])-float(cfg['b']))>1e-7:raise ValueError('Augmentation base calibration mismatch')
 cw=np.concatenate([.5*source_weights(old_sources),.5*source_weights(new_sources)])
 nt=np.concatenate([t[ni['fit']],at]);ne=np.concatenate([ell[ni['fit']]+base_delta[ni['fit']],ae+ad]);labels=np.concatenate([ny['fit'],ay])
 mu=np.sum(nt*cw[:,None],axis=0);scale=np.sqrt(np.sum((nt-mu)**2*cw[:,None],axis=0));scale=np.maximum(scale,1e-5)
 z=(t-mu)/scale;az=(at-mu)/scale;nz=np.concatenate([z[ni['fit']],az])
 def calibrate(delta):
  v=ell[ni['fit']]+delta[ni['fit']];yy=ny['fit']
  fn=lambda b:float(np.mean(expit(v+b)-yy))
  lo,hi=-1.,1.
  while fn(lo)>0:lo*=2
  while fn(hi)<0:hi*=2
  return float(brentq(fn,lo,hi,xtol=1e-12))
 bref=calibrate(base_delta);base_score=ell+base_delta+bref
 yl=np.array([p['left_label'] for p in ap]);yr=np.array([p['right_label'] for p in ap])
 wrong=((base_score[al]>0)!=yl)|((base_score[ar]>0)!=yr)
 near=np.minimum(np.abs(base_score[al]),np.abs(base_score[ar]))<1.
 target=(ell[ar]+base_delta[ar])-(ell[al]+base_delta[al])
 tail_cut=float(np.quantile(np.abs(target),.9));tail=np.abs(target)>=tail_cut
 raw_weight=1.+wrong.astype(float)+near.astype(float)+tail.astype(float)
 pw=source_weights([p['source_id'] for p in ap],raw_weight)
 X=z[ar]-z[al];E=z[cr]-z[cl];base_cf=base_delta[cr]-base_delta[cl]
 fit_raw=ell[ar]-ell[al];apscale=float(np.sqrt(np.mean(fit_raw**2)));budget=.1*apscale
 if np.max(np.abs(base_cf))>budget+1e-7:raise ValueError('Zero residual not feasible under total fit CF budget')
 regularizer=.01/len(ap)/(apscale**2)
 def objective(w):
  residual=target+X@w;score=ne+nz@w+bref
  cls=np.logaddexp(0,score)-labels*score
  value=float(np.sum(pw*residual**2)/(apscale**2)+np.sum(cw*cls)+regularizer*np.sum((w/scale)**2))
  gradient=2*X.T@(pw*residual)/(apscale**2)+nz.T@(cw*(expit(score)-labels))+2*regularizer*w/(scale**2)
  return value,gradient
 result=minimize(lambda w:objective(w)[0],np.zeros(32),jac=lambda w:objective(w)[1],method='SLSQP',
  constraints=[LinearConstraint(E,-budget-base_cf,budget-base_cf)],options=dict(ftol=1e-11,maxiter=1500,disp=False))
 w=result.x;total_cf=base_cf+E@w
 if not result.success or np.max(np.abs(total_cf))>budget+1e-7:raise RuntimeError(str(result.message))
 train_info=dict(method='balanced_plus_hard_residual_r32',dimension=32,base_method='optimized_balanced_fit_calibration',
  base_frozen_manifest=str(BASE/'frozen_methods.json'),base_prediction_file=cfg['prediction_file'],base_gain=float(cfg['gain']),
  base_delta_convention='cached delta times frozen base_gain exactly once; no bias included',
  old_natural_n=len(ni['fit']),old_natural_sources=len(set(old_sources)),new_natural_n=len(at),new_natural_sources=len(set(new_sources)),
  natural_group_weight=dict(original=.5,augmentation=.5),within_group='equal source means',AP_pairs=len(ap),CF_pairs=len(cf),
  AP_weights='Within-source normalized (1 + fit-base wrong endpoint + abs margin<1 + top fit90% absolute difference)',
  fit_tail_cut=tail_cut,fit_wrong_pairs=int(wrong.sum()),fit_near_pairs=int(near.sum()),fit_tail_pairs=int(tail.sum()),
  objective='source-balanced weighted AP squared residual / fit raw AP RMS^2 + half-group source-balanced natural BCE + .01/nAP coordinate ridge / fit AP RMS^2',
  budget=budget,base_fit_cf_max=float(np.max(np.abs(base_cf))),final_fit_cf_max=float(np.max(np.abs(total_cf))),
  calibration='b fitted only on original 761 fit natural nodes after each fixed residual gain',solver=result.message,iterations=int(result.nit))
 np.savez_compressed(OUT/'model_unscaled.npz',w=w,mu=mu,scale=scale,bref=bref)
 dump(OUT/'training.json',train_info)
 sharedmeta=json.loads(Path('/home/gcx/aster_optimization_20261109/shared/metadata.json').read_text())
 pathrec=[r for r in sharedmeta['paths'] if r['split']=='dev'];pidx={r['id']:ix['path|'+r['id']] for r in pathrec}
 cases=[c for c in rows('/home/gcx/aster_cr_repair_v2_run_20261108/data/path_cases.jsonl') if c['split']=='dev']
 bycase={}
 for r in pathrec:bycase.setdefault(r['case_id'],{})[r['gid']]=pidx[r['id']]
 groups=[('A',('A',)),('P',('P',)),('AP',('A','P')),('C',('C',)),('F',('F',)),('CF',('C','F')),('F0',('F0',))]
 def evaluate(delta,b,split):
  score=ell+delta+b;out={}
  _,pl,pr=indices(split,('A','P'));rawscale=float(np.sqrt(np.mean((ell[pr]-ell[pl])**2)))
  for name,kinds in groups:
   pp,l,r=indices(split,kinds);raw=ell[r]-ell[l];new=score[r]-score[l];corr=new-raw
   labl=np.array([p['left_label'] for p in pp]);labr=np.array([p['right_label'] for p in pp]);cor=((score[l]>0)==labl)&((score[r]>0)==labr)
   m=dict(n=len(pp),difference_rms=float(np.sqrt(np.mean(new**2))),correction_rms=float(np.sqrt(np.mean(corr**2))),
    correction_rms_over_ap_rms=float(np.sqrt(np.mean(corr**2)))/rawscale,paired_correct=int(cor.sum()),paired_accuracy=float(cor.mean()))
   if name in ('A','P','AP','F0'):m['variance_reduction']=1-float(np.var(new)/np.var(raw))
   else:
    old=(labr-labl)*raw;fresh=(labr-labl)*new
    m.update(mean_evidence_retention=float(fresh.mean()/old.mean()),response_direction_degraded=int(np.sum((old>0)&(fresh<=0))))
   out[name]=m
  correct=((score[ni[split]]>0)==ny[split]);out['natural']=dict(n=len(correct),correct=int(correct.sum()),accuracy=float(correct.mean()))
  return out
 raw_residual=z@w;aug_residual=az@w;records=[]
 for amplitude in (0.,.25,.5,.75,1.):
  delta=base_delta+amplitude*raw_residual;b=calibrate(delta)
  fit=evaluate(delta,b,'fit');dev=evaluate(delta,b,'dev')
  augscore=ae+ad+amplitude*aug_residual+b
  augcorrect=(augscore>0)==ay
  pathresults=[]
  score=ell+delta+b
  for c in cases:pathresults.append(dict(case_id=c['case_id'],source_id=c['source_id'],**execute(c,{gid:float(score[j]) for gid,j in bycase[c['case_id']].items()})))
  count=sum(r['correct'] for r in pathresults);dev['path']=dict(n=len(pathresults),correct=count,accuracy=count/len(pathresults))
  entry=dict(amplitude=amplitude,b=b,fit=fit,fit_augmentation=dict(n=len(ay),sources=len(set(new_sources)),correct=int(augcorrect.sum()),accuracy=float(augcorrect.mean())),
   dev=dev,fit_cf_max=float(np.max(np.abs(base_cf+amplitude*E@w))),
   eligible=dev['CF']['correction_rms_over_ap_rms']<=.1+1e-7)
  records.append(entry)
  np.savez_compressed(OUT/('scores_a'+str(amplitude)+'.npz'),ids=np.array(ids),delta=delta,b=b)
  with (OUT/('dev_paths_a'+str(amplitude)+'.jsonl')).open('w') as f:
   for r in pathresults:f.write(json.dumps(r)+'\n')
 eligible=[r for r in records if r['eligible']]
 if not eligible:raise ValueError('Original zero-residual balanced point unexpectedly ineligible')
 best=max(eligible,key=lambda r:(r['dev']['path']['correct'],r['dev']['AP']['variance_reduction'],-r['amplitude']))
 np.savez_compressed(OUT/'frozen_model.npz',w=w,mu=mu,scale=scale,amplitude=best['amplitude'],b=best['b'],base_b=float(cfg['b']),base_gain=float(cfg['gain']))
 frozen=dict(train_info,kind='balanced_hard_linear_residual',feature='r32',base_b=float(cfg['b']),base_gain=float(cfg['gain']),selected_fit_cf_max=best['fit_cf_max'],nonzero_residual_selected=bool(best['amplitude']),amplitude=best['amplitude'],b=best['b'],selected_amplitude=best['amplitude'],selected_b=best['b'],model_file=str(OUT/'frozen_model.npz'),
  dev=best['dev'],fit=best['fit'],fit_augmentation=best['fit_augmentation'],scores_file=str(OUT/('scores_a'+str(best['amplitude'])+'.npz')),
  selection='dev CF correction RMS/raw AP RMS <= .1; maximize dev exact path count then AP VR then smaller amplitude; original-fit-only b calibration',
  test_used_for_training_or_selection=False,test_evaluated=False,seconds=time.time()-started)
 dump(OUT/'dev_amplitudes.json',records);dump(OUT/'frozen.json',frozen);dump(OUT/'selected.json',frozen)
 dump(OUT/'done.json',dict(state='complete',amplitudes=5,test_evaluated=False,seconds=time.time()-started))
 print(json.dumps(dict(state='complete',selected_amplitude=best['amplitude'],dev_path=best['dev']['path']['correct'],
  dev_AP_VR=best['dev']['AP']['variance_reduction'],dev_CF_ER=best['dev']['CF']['mean_evidence_retention'],
  dev_CF_correction_ratio=best['dev']['CF']['correction_rms_over_ap_rms'],fit_CF_max=frozen['final_fit_cf_max'],seconds=time.time()-started)))
if __name__=='__main__':main()

