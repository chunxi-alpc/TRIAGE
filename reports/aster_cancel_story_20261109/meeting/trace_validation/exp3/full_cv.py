"""Optional original-procedure full CV. Runs real Qwen single-head interventions.
CPU attribution: python full_cv.py --prepare
GPU0: CUDA_VISIBLE_DEVICES=0 python full_cv.py --folds 0 2 4
GPU1: CUDA_VISIBLE_DEVICES=1 python full_cv.py --folds 1 3
Aggregate: python full_cv.py --aggregate
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','8')
os.environ.setdefault('OMP_NUM_THREADS','8')
import sys,json,csv,time,argparse
from pathlib import Path
from collections import defaultdict
from functools import lru_cache
import numpy as np
from sklearn.linear_model import Ridge
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent));sys.path.insert(0,str(HERE.parents[4]))
from shared import Data,ROOT,rows,dump
from run_cv import source_folds,weighted_subset,metrics,table
OUT=HERE/'full'
KINDS=('A','P','C','F')

def prepare():
 OUT.mkdir(exist_ok=True);start=time.time();d=Data();raw=d.all_raw();sources=sorted({v['source_id'] for v in d.views if v['split']=='fit'})
 si={s:i for i,s in enumerate(sources)};attr=np.zeros((len(sources),4,1152));den=np.zeros((len(sources),4))
 grouped=defaultdict(list)
 for p in d.pairs:
  if p['split']=='fit' and p['kind'] in KINDS:grouped[p['left']].append(p)
 for j,(left,pp) in enumerate(grouped.items()):
  with np.load(d.scores[left]['feature']) as f:
   if 'grads' not in f:raise ValueError('Fit gradient missing '+left)
   grad=f['grads'].astype('float32').reshape(1152,128)
  for p in pp:
   l,r=d.ell[d.ix[p['left']]],d.ell[d.ix[p['right']]]
   w=1+int((l>0)!=bool(p['left_label']) or (r>0)!=bool(p['right_label']))+int(min(abs(l),abs(r))<1)
   a,b=si[p['source_id']],KINDS.index(p['kind'])
   diff=(raw[d.ix[p['right']]]-raw[d.ix[left]]).reshape(1152,128)
   attr[a,b]+=w*np.abs((diff*grad).sum(-1));den[a,b]+=w
  if (j+1)%200==0:dump(OUT/'attribution_status.json',dict(stage='preparing',left_views=j+1,total=len(grouped),seconds=time.time()-start))
 np.savez_compressed(OUT/'attribution_by_source.npz',sources=sources,attr=attr,den=den)
 dump(OUT/'attribution_status.json',dict(stage='complete',left_views=len(grouped),seconds=time.time()-start,evidence_type='real_cache_recomputation'))

def run(folds,replay=False):
 OUT.mkdir(exist_ok=True);d=Data();raw=d.all_raw();vs={v['view_id']:v for v in d.views}
 z=np.load(OUT/'attribution_by_source.npz');sources=z['sources'].tolist();source_set=set(sources);allfolds=source_folds(sources)
 existing={(x['pair_id'],x['head']):x for x in rows(ROOT/'qwen/mechanisms/fit_exact.jsonl')}
 engine=None
 for fold in folds:
  start=time.time();valid=set(allfolds[fold]);train=source_set-valid
  original_target=OUT/f'fold{fold}'
  target=(OUT/'replay'/f'fold{fold}') if replay else original_target
  target.mkdir(parents=True,exist_ok=True)
  if (target/'metrics.json').exists() and not replay:continue
  selected_si=[i for i,s in enumerate(sources) if s in train]
  attrs=z['attr'][selected_si].sum(0);dens=z['den'][selected_si].sum(0)
  ranks={};selected_pairs={};new_forward=0;reused_rows=0
  exact=dict(existing);exact.update({(x['pair_id'],x['head']):x for x in rows(original_target/'exact.jsonl')})
  for ki,kind in enumerate(KINDS):
   attr=attrs[ki]/max(dens[ki],1);candidate=np.argsort(-attr)[:64].tolist()
   pp=weighted_subset(d.pairs,d.ell,d.ix,train,kind);selected_pairs[kind]=[p['pair_id'] for p,w in pp]
   sums=defaultdict(float);weights=0
   for pair_index,(p,w) in enumerate(pp):
    needed=[h for h in candidate if (p['pair_id'],h) not in exact]
    if needed:
     if replay:raise RuntimeError('Exact cache incomplete for CPU replay: '+p['pair_id'])
     if engine is None:
      from experiments.aster_cr.engine import Engine
      engine=Engine('qwen',device='cuda:0')
     from experiments.aster_closed_form.m1 import _effects
     left=dict(heads=np.array(raw[d.ix[p['left']]],copy=True).reshape(36,32,128),ell=float(d.ell[d.ix[p['left']]]))
     right=dict(heads=np.array(raw[d.ix[p['right']]],copy=True).reshape(36,32,128),ell=float(d.ell[d.ix[p['right']]]))
     contexts=(engine.prepare(vs[p['left']]['prompt']),engine.prepare(vs[p['right']]['prompt']))
     for h in needed:
      e=_effects(engine,contexts,left,right,[h],right['ell']-left['ell'])
      rr=dict(pair_id=p['pair_id'],source_id=p['source_id'],kind=kind,head=h,weight=w,**e)
      with (target/'exact.jsonl').open('a') as f:f.write(json.dumps(rr)+'\n')
      exact[(p['pair_id'],h)]=rr;new_forward+=2
     del contexts
    for h in candidate:
     e=exact[(p['pair_id'],h)];sums[h]+=w*.5*(abs(e['ie_forward'])+abs(e['ie_reverse']))
     reused_rows+=int((p['pair_id'],h) in existing)
    weights+=w
    dump(target/'status.json',dict(stage='exact',fold=fold,kind=kind,pairs_done=pair_index+1,pairs_total=len(pp),new_final_forwards=new_forward,reused_original_rows=reused_rows,seconds=time.time()-start))
   ranks[kind]=sorted(candidate,key=lambda h:(-sums[h]/max(weights,1),h))
  nsort=sorted(set(ranks['A']+ranks['P']),key=lambda h:sum(ranks[t].index(h) if h in ranks[t] else 64 for t in ('A','P')))
  ssort=sorted(set(ranks['C']+ranks['F']),key=lambda h:sum(ranks[t].index(h) if h in ranks[t] else 64 for t in ('C','F')))
  heads=[]
  for i in range(max(len(nsort),len(ssort))):
   for ranking in (nsort,ssort):
    if i<len(ranking) and ranking[i] not in heads:heads.append(ranking[i])
   if len(heads)>=51:break
  heads=heads[:51]
  s=d.support(heads,16,train_sources=train);rr=d.raw(heads);coords=((rr-s['mu'])/s['scale'])@s['N']
  tr=[p for p in d.pairs if p['split']=='fit' and p['source_id'] in train and p['kind'] in ('A','P')]
  va=[p for p in d.pairs if p['split']=='fit' and p['source_id'] in valid and p['kind'] in ('A','P')]
  def arrays(pp):
   a=np.array([d.ix[p['left']] for p in pp]);b=np.array([d.ix[p['right']] for p in pp])
   return coords[b]-coords[a],d.ell[b]-d.ell[a]
  tx,ty=arrays(tr);vx,vy=arrays(va);model=Ridge(alpha=.01,fit_intercept=False).fit(tx,ty);pred=model.predict(vx)
  predictions=[dict(protocol='full_pipeline_cv',fold=fold,pair_id=p['pair_id'],source_id=p['source_id'],group_id=p['group_id'],kind=p['kind'],target=float(y),prediction=float(z),error=float(z-y)) for p,y,z in zip(va,vy,pred)]
  table(target/'pair_predictions.csv',predictions)
  np.savez_compressed(target/'support_and_ridge.npz',**s,coef=model.coef_,intercept=model.intercept_)
  dump(target/'configuration.json',dict(fold=fold,train_sources=sorted(train),validation_sources=sorted(valid),heads=heads,ranks=ranks,exact_fit_pairs=selected_pairs,rank=16,ridge_alpha=.01,fit_intercept=False,head_selection='Original weighted gradient top64 and bidirectional exact donor effect; source-balanced up to12 train pairs perkind; alternating interference and evidence rankings to51.',preprocessing='All scale, S, N and regression fitting use this fold training sources exclusively.',cache_reuse='Only identical pair/head original exact effects reused; their effect is independent of other training sources.',evidence_type='new_real_model_interventions_and_real_cache_training'))
  ar=[p for p in predictions if p['kind']=='A']
  dump(target/'metrics.json',dict(fold=fold,train_sources=len(train),validation_sources=len(valid),train_pairs=len(tr),effective_rank=s['N'].shape[1],AP=metrics(vy,pred),A=metrics([p['target'] for p in ar],[p['prediction'] for p in ar]),new_final_forwards=new_forward,reused_original_rows=reused_rows,seconds=time.time()-start,evidence_type='new_real_model_interventions_and_real_cache_training'))
  dump(target/'status.json',dict(stage='complete',fold=fold,seconds=time.time()-start))
 if engine is not None:
  import torch
  dump(OUT/('worker_'+''.join(map(str,folds))+'_cost.json'),dict(counts=engine.counts,peak_gib=torch.cuda.max_memory_allocated()/2**30))

def aggregate():
 pairs=[];folds=[]
 for fold in range(5):
  p=OUT/f'fold{fold}'
  if not (p/'metrics.json').exists():raise RuntimeError('Fold incomplete '+str(fold))
  folds.append(json.loads((p/'metrics.json').read_text()))
  with (p/'pair_predictions.csv').open() as f:
   for rr in csv.DictReader(f):
    for k in ('target','prediction','error'):rr[k]=float(rr[k])
    pairs.append(rr)
 table(OUT/'pair_predictions.csv',pairs)
 ss=[]
 for source in sorted({p['source_id'] for p in pairs}):
  pp=[p for p in pairs if p['source_id']==source];ss.append(dict(source_id=source,fold=pp[0]['fold'],**metrics([p['target'] for p in pp],[p['prediction'] for p in pp])))
 table(OUT/'source_metrics.csv',ss)
 table(OUT/'fold_metrics.csv',[dict(fold=x['fold'],train_sources=x['train_sources'],validation_sources=x['validation_sources'],new_final_forwards=x['new_final_forwards'],reused_original_rows=x['reused_original_rows'],seconds=x['seconds'],**x['AP']) for x in folds])
 ap=metrics([p['target'] for p in pairs],[p['prediction'] for p in pairs]);a=[p for p in pairs if p['kind']=='A']
 finite=sorted([s for s in ss if s['r2'] is not None],key=lambda x:x['r2'])
 dump(OUT/'metrics.json',dict(protocol='full_pipeline_source_cv',AP=ap,A=metrics([p['target'] for p in a],[p['prediction'] for p in a]),folds=folds,positive_source_r2=sum(x['r2']>0 for x in finite),source_r2_median=float(np.median([x['r2'] for x in finite])),worst_sources=finite[:5],best_sources=finite[-5:],evidence_type='new_real_model_interventions_and_real_cache_training'))
 print(json.dumps(dict(AP=ap,positive_source_r2=sum(x['r2']>0 for x in finite))))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--aggregate',action='store_true');p.add_argument('--replay',action='store_true');p.add_argument('--folds',nargs='+',type=int);a=p.parse_args()
 if a.prepare:prepare()
 elif a.aggregate:aggregate()
 elif a.folds:run(a.folds,replay=a.replay)
 else:p.error('Choose --prepare, --aggregate or --folds')
