"""Source-held-out CV from real cached activations; no backbone forwards."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','8')
os.environ.setdefault('OMP_NUM_THREADS','8')
import sys,json,csv,time,argparse
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
REPO=Path(__file__).resolve().parents[5]
# exp3/run_cv.py -> trace_validation -> meeting -> story -> reports -> repo
sys.path.insert(0,str(REPO))
from experiments.aster_cr.features import basis,orth_nuisance
ROOT=Path('/home/gcx/aster_cr_repair_v2_run_20261108')
CACHE=Path('/home/gcx/aster_cancel_story_20261109/meeting/real_completion')
OUT=Path(__file__).resolve().parent
SEED=20261109

def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def table(p,rr):
 if not rr:return
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def metrics(y,p):
 y=np.asarray(y);p=np.asarray(p)
 return dict(n=len(y),r2=float(r2_score(y,p)) if len(y)>1 and np.var(y)>1e-14 else None,rmse=float(np.sqrt(np.mean((y-p)**2))),target_rms=float(np.sqrt(np.mean(y**2))),target_sd=float(np.std(y)))
def source_folds(sources):
 s=np.array(sorted(sources));np.random.default_rng(SEED).shuffle(s)
 return [x.tolist() for x in np.array_split(s,5)]
def get_data():
 if not (CACHE/'coordinates.npz').exists() or not (CACHE/'raw_selected.npy').exists():
  sys.path.insert(0,str(OUT.parent))
  from shared import Data
  d=Data();heads=d.heads
  return d.ix,d.ell,d.raw(heads),d.views,d.pairs,heads
 a=np.load(CACHE/'coordinates.npz');ids=a['ids'].tolist();ix={x:i for i,x in enumerate(ids)}
 ell=a['ell'];raw=np.load(CACHE/'raw_selected.npy',mmap_mode='r')
 views=[v for v in rows(ROOT/'data/views.jsonl') if v['view_id'] in ix]
 pairs=[p for p in rows(ROOT/'data/pairs.jsonl') if p['left'] in ix and p['right'] in ix]
 with np.load(ROOT/'qwen/supports_r16.npz') as s:heads=s['heads'].tolist()
 return ix,ell,raw,views,pairs,heads

def support(raw,ix,views,pairs,train_sources,rank=16):
 vs=[v for v in views if v['split']=='fit' and v['variant']=='full' and v['source_id'] in train_sources]
 x=np.asarray(raw[[ix[v['view_id']] for v in vs]],dtype=np.float32)
 mu=x.mean(0);scale=np.maximum(x.std(0),1e-5)
 nd=[];sd=[]
 for p in pairs:
  if p['split']!='fit' or p['source_id'] not in train_sources or p['kind'] not in ['A','P','C','F']:continue
  d=(raw[ix[p['right']]]-raw[ix[p['left']]])/scale
  (nd if p['kind'] in ['A','P'] else sd).append(d)
 S=basis(np.asarray(sd),rank);N=orth_nuisance(S,basis(np.asarray(nd),rank))
 return dict(mu=mu,scale=scale,S=S,N=N,fit_full_views=len(vs),fit_ap=len(nd),fit_cf=len(sd))

def weighted_subset(pairs,ell,ix,train_sources,kind):
 ps=[]
 for p in pairs:
  if p['split']!='fit' or p['source_id'] not in train_sources or p['kind']!=kind:continue
  l,r=ell[ix[p['left']]],ell[ix[p['right']]]
  wrong=int((l>0)!=bool(p['left_label']) or (r>0)!=bool(p['right_label']))
  boundary=int(min(abs(l),abs(r))<1)
  ps.append((p,1+wrong+boundary))
 out=[];seen=set()
 for p,w in sorted(ps,key=lambda pw:(-pw[1],pw[0]['pair_id'])):
  if p['source_id'] in seen or abs(ell[ix[p['right']]]-ell[ix[p['left']]])<1e-8:continue
  out.append((p,w));seen.add(p['source_id'])
  if len(out)>=12:break
 return out

def audit():
 ix,ell,raw,views,pairs,heads=get_data()
 sources={v['source_id'] for v in views if v['split']=='fit'}
 exact=rows(ROOT/'qwen/mechanisms/fit_exact.jsonl');keys={(x['pair_id'],x['head']) for x in exact}
 ep={x['pair_id'] for x in exact}
 rr=[]
 for fold,valid in enumerate(source_folds(sources)):
  train=sources-set(valid)
  for kind in ['A','P','C','F']:
   ps=weighted_subset(pairs,ell,ix,train,kind)
   rr.append(dict(fold=fold,kind=kind,selected_pairs=len(ps),cached_pairs=sum(p['pair_id'] in ep for p,w in ps),missing_pairs=sum(p['pair_id'] not in ep for p,w in ps),required_exact_rows=len(ps)*64,cached_rows_for_selected_pairs=sum(x['pair_id'] in {p['pair_id'] for p,w in ps} for x in exact)))
 table(OUT/'full_cv_cache_coverage.csv',rr)
 dump(OUT/'full_cv_feasibility.json',dict(existing_exact_rows=len(exact),existing_exact_pairs=len(ep),fold_kind_coverage=rr,full_original_procedure='Fit weighted gradient attribution on each training source subset; top64 by kind; up to12 source-balanced train pairs per kind; bidirectional exact single-head interventions; alternate A/P and C/F rankings to 51 heads; train-only standardization/subspaces/ridge.',can_complete_original_five_fold_from_exact_cache_alone=not any(x['missing_pairs'] for x in rr),candidate_coverage='Even if pair cached, fold-specific top64 must be recomputed; an old cached64 list need not contain every new candidate.',approximate_upper_bound_final_forwards=5*4*12*64*2,estimated_full_cv_wall_minutes_two_a100='30-60 (rough estimate from cached original final-token timings; not measured)',new_sources='Original preparation inventory had1346 eligible sources and froze140. Remaining sources have no confirmed unused-source activation cache in this workflow. No new-source independent validation claimed.',test_history='Original repair-v2 method frozen2026-11-08 16:51:33; test scores and paths already evaluated. meeting/real_completion/metrics.json explicitly says held-out test previously used in legacy evaluations.',evidence_type='real_cache_metadata_recomputation'))
 return rr

def run():
 start=time.time();ix,ell,raw,views,pairs,heads=get_data()
 sources={v['source_id'] for v in views if v['split']=='fit'}
 all_predictions=[];fold_rows=[];cfg=[]
 for fold,valid in enumerate(source_folds(sources)):
  valid=set(valid);train=sources-valid
  s=support(raw,ix,views,pairs,train)
  coords=np.asarray((raw-s['mu'])/s['scale'])@s['N']
  tr=[p for p in pairs if p['split']=='fit' and p['source_id'] in train and p['kind'] in ['A','P']]
  va=[p for p in pairs if p['split']=='fit' and p['source_id'] in valid and p['kind'] in ['A','P']]
  def arrays(ps):
   li=np.array([ix[p['left']] for p in ps]);ri=np.array([ix[p['right']] for p in ps])
   return coords[ri]-coords[li],ell[ri]-ell[li]
  tx,ty=arrays(tr);vx,vy=arrays(va)
  model=Ridge(alpha=.01,fit_intercept=False).fit(tx,ty);pred=model.predict(vx)
  np.savez_compressed(OUT/f'frozen_heads_fold{fold}.npz',heads=heads,mu=s['mu'],scale=s['scale'],S=s['S'],N=s['N'],coef=model.coef_,intercept=model.intercept_)
  fold_rows.append(dict(protocol='frozen_heads_conditional_diagnostic',fold=fold,train_sources=len(train),validation_sources=len(valid),train_pairs=len(tr),effective_rank=s['N'].shape[1],**metrics(vy,pred)))
  cfg.append(dict(fold=fold,train_sources=sorted(train),validation_sources=sorted(valid),fit_full_views=s['fit_full_views'],fit_ap=s['fit_ap'],fit_cf=s['fit_cf'],intercept=float(model.intercept_)))
  for p,y,z in zip(va,vy,pred):all_predictions.append(dict(protocol='frozen_heads_conditional_diagnostic',fold=fold,pair_id=p['pair_id'],source_id=p['source_id'],group_id=p['group_id'],kind=p['kind'],target=float(y),prediction=float(z),error=float(z-y)))
  dump(OUT/'status.json',dict(stage='frozen_heads_cv',folds_completed=fold+1,seconds=time.time()-start))
 y=[p['target'] for p in all_predictions];p=[p['prediction'] for p in all_predictions]
 allmetrics=metrics(y,p)
 arows=[x for x in all_predictions if x['kind']=='A'];ametrics=metrics([x['target'] for x in arows],[x['prediction'] for x in arows])
 source_rows=[]
 for sid in sorted(sources):
  rr=[x for x in all_predictions if x['source_id']==sid]
  if rr:source_rows.append(dict(source_id=sid,fold=rr[0]['fold'],**metrics([x['target'] for x in rr],[x['prediction'] for x in rr])))
 table(OUT/'frozen_heads_fold_metrics.csv',fold_rows);table(OUT/'frozen_heads_source_metrics.csv',source_rows);table(OUT/'frozen_heads_pair_predictions.csv',all_predictions)
 finite=[x for x in source_rows if x['r2'] is not None]
 finite.sort(key=lambda x:x['r2'])
 cfgout=dict(seed=SEED,alpha=.01,fit_intercept=False,heads=heads,selection='Frozen51 heads were chosen using all80 fit sources, including each CV validation fold; this is explicitly a conditional diagnostic, not full-pipeline CV.',feature_fit='For each fold all standardization, C/F signal basis, A/P nuisance basis, orthogonalization and ridge are estimated only on64 training sources.',splits=cfg,data_root=str(ROOT),cache_root=str(CACHE),rank=16,evidence_type='real_cache_recomputation_with_new_ridge_training')
 dump(OUT/'frozen_heads_config.json',cfgout)
 dump(OUT/'frozen_heads_metrics.json',dict(protocol='frozen_heads_conditional_diagnostic',overall=allmetrics,A=ametrics,folds=fold_rows,source_r2_median=float(np.median([x['r2'] for x in finite])),positive_source_r2=sum(x['r2']>0 for x in finite),source_r2_min=finite[0] if finite else None,source_r2_max=finite[-1] if finite else None,worst_sources=finite[:5],seconds=time.time()-start,evidence_type='real_cache_recomputation_with_new_ridge_training'))
 audit();dump(OUT/'status.json',dict(stage='complete',seconds=time.time()-start))
 print(json.dumps(dict(overall=allmetrics,A=ametrics,source_r2_median=float(np.median([x['r2'] for x in finite])),positive_source_r2=sum(x['r2']>0 for x in finite),worst=finite[:3],seconds=time.time()-start)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--audit-only',action='store_true');a=p.parse_args();audit() if a.audit_only else run()
