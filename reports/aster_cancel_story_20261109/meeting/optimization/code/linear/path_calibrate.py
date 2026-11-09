
import os
os.environ["OPENBLAS_NUM_THREADS"]="4"
os.environ["OMP_NUM_THREADS"]="4"
from pathlib import Path
import json,sys,time
from collections import defaultdict
import numpy as np
OUT=Path(__file__).resolve().parent
ROOT=Path('/home/gcx/aster_cr_repair_v2_run_20261108')
OLD=Path('/home/gcx/aster_cancel_story_20261109/meeting/real_completion')
sys.path.insert(0,'/home/gcx/aster_optimization_20261109/diagnosis')
from path_tools import select_dev_bias
def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def stable_interval_bias(scores,natural_scores,natural_labels,bias,bref,gold_index,gold_label,gold_mask):
 if bias['best_b'] is None:return bias
 values=scores[gold_index]
 lower=np.max(np.where(gold_mask&(gold_label==1),-values,-np.inf),axis=1)
 upper=np.min(np.where(gold_mask&(gold_label==0),-values,np.inf),axis=1)
 valid=lower<upper
 lo=np.sort(lower[valid]);hi=np.sort(upper[valid])
 pos=np.sort(-natural_scores[natural_labels==1]);neg=np.sort(-natural_scores[natural_labels==0])
 bounds=np.unique(np.concatenate([lower,upper,-natural_scores]));bounds=bounds[np.isfinite(bounds)]
 j=np.searchsorted(bounds,bias['best_b'])
 mids=[]
 for k in range(max(0,j-3),min(len(bounds)-1,j+3)):
  gap=bounds[k+1]-bounds[k]
  if gap>1e-9*(1+abs(bounds[k])):
   mids.append(bounds[k]+.5*gap)
 if not mids:return bias
 mids=np.array(mids)
 pc=np.searchsorted(lo,mids,side='left')-np.searchsorted(hi,mids,side='left')
 nc=np.searchsorted(pos,mids,side='left')+len(neg)-np.searchsorted(neg,mids,side='left')
 ok=(pc>=bias['correct'])&(nc>=bias['natural_correct'])
 if not np.any(ok):return bias
 eligible=np.flatnonzero(ok)
 k=eligible[np.argmin(np.abs(mids[eligible]-bref))]
 out=dict(bias,boundary_best_b=bias['best_b'],best_b=float(mids[k]),bias_inside_optimal_interval=True,correct=int(pc[k]),natural_correct=int(nc[k]))
 return out

def main():
 start=time.time()
 candidates=json.loads((OUT/'scaled_candidates.json').read_text())
 originals=json.loads((OUT/'candidates.json').read_text())
 orig={r['config']['name']:r['config'] for r in originals}
 pf=dict(np.load(OUT/'dev_path_features.npz'));pathids=pf['ids'].tolist()
 a=np.load(OLD/'coordinates.npz');ids=a['ids'].tolist();ix={v:i for i,v in enumerate(ids)};ell=a['ell']
 meta=json.loads((OLD/'metadata.json').read_text())
 cases=[c for c in rows(ROOT/'data/path_cases.jsonl') if c['split']=='dev']
 pidx={v:i for i,v in enumerate(pathids)};maxgold=max(len(c['gold_path']) for c in cases)
 gold_index=np.zeros((len(cases),maxgold),dtype=int);gold_label=np.zeros_like(gold_index);gold_mask=np.zeros_like(gold_index,dtype=bool)
 for j,c in enumerate(cases):
  for k,n in enumerate(c['gold_path']):gold_index[j,k]=pidx[c['case_id']+':'+n['gid']];gold_label[j,k]=n['label'];gold_mask[j,k]=True
 vv=[v for v in meta['views'] if v['split']=='dev' and v['variant']=='full' and v['original']]
 ni=np.array([ix[v['view_id']] for v in vv]);ny=np.array([v['label'] for v in vv])
 pairsets={};natsets={}
 for sp in ('fit','dev'):
  for kind in ('A','P','AP','C','F','CF','F0'):
   kinds=(kind,) if kind not in ('AP','CF') else tuple(kind)
   ps=[p for p in meta['pairs'] if p['split']==sp and p['kind'] in kinds]
   pairsets[sp,kind]=np.array([ix[p['left']] for p in ps]),np.array([ix[p['right']] for p in ps]),np.array([p['left_label'] for p in ps]),np.array([p['right_label'] for p in ps])
  vs=[v for v in meta['views'] if v['split']==sp and v['variant']=='full' and v['original']]
  natsets[sp]=np.array([ix[v['view_id']] for v in vs]),np.array([v['label'] for v in vs])
 parents=sorted({r['config']['parent'] for r in candidates})
 byfamily=defaultdict(list)
 for parent in parents:byfamily[orig[parent]['family']].append(parent)
 pathdelta={};deltas={}
 for family,pp in byfamily.items():
  ms=[dict(np.load(orig[p]['model_file'])) for p in pp]
  weights=np.stack([x['w'] for x in ms],axis=1)
  z=(pf[family]-ms[0]['mu'])/ms[0]['scale']
  dd=z@weights
  for j,p in enumerate(pp):
   pathdelta[p]=dd[:,j]
   deltas[p]=np.load(orig[p]['score_file'])['delta'].astype('float64')
 result=[]
 for j,r in enumerate(candidates):
  cfg=r['config'];parent=cfg['parent'];gain=cfg['gain'];old_b=cfg['b']
  s=pf['ell']+pathdelta[parent]*gain
  delta=deltas[parent]*gain
  nt=ell[ni]+delta[ni]
  bias=select_dev_bias(dict(zip(pathids,s)),cases,nt,ny,bref=old_b,min_natural_correct=195)
  bias=stable_interval_bias(s,nt,ny,bias,old_b,gold_index,gold_label,gold_mask)
  if bias['best_b'] is None:continue
  new_b=bias['best_b']
  cfg=dict(cfg,name='pb_'+str(j).zfill(4),parent=parent,parent_score_file=orig[parent]['score_file'],
    old_b=old_b,new_b=new_b,b=new_b,bias_selection='dev exact gold-path intervals; natural dev correct >=195')
  cfg['score_file']=str(OUT/'candidates'/(cfg['name']+'.npz'))
  scores=ell+delta+new_b
  metrics={}
  for sp in ('fit','dev'):
   g={key:dict(vals) for key,vals in r['metrics'][sp].items()}
   for kind in ('A','P','AP','C','F','CF','F0'):
    li,ri,yl,yr=pairsets[sp,kind];cor=((scores[li]>0)==yl)&((scores[ri]>0)==yr)
    g[kind]['paired_accuracy']=float(np.mean(cor));g[kind]['paired_correct']=int(cor.sum())
   nn,lab=natsets[sp];cor=(scores[nn]>0)==lab
   g['natural']=dict(n=len(nn),correct=int(cor.sum()),accuracy=float(np.mean(cor)))
   metrics[sp]=g
  metrics['dev']['path']=dict(n=len(cases),correct=bias['correct'],accuracy=bias['correct']/len(cases))
  result.append(dict(config=cfg,metrics=metrics,bias_selection=bias))
  if j%100==0:dump(OUT/'path_bias_status.json',dict(completed=j,total=len(candidates),seconds=time.time()-start))
 dump(OUT/'path_calibrated_all.json',result)
 selected={};save=set();byname={r['config']['name']:r for r in result}
 for tier in (.05,.1,.2,.4):
  eligible=[r for r in result if abs(r['config']['dev_tier']-tier)<1e-8 and r['metrics']['dev']['CF']['mean_evidence_retention']>=.98-1e-7 and r['metrics']['dev']['natural']['correct']>=195]
  maintained=[r for r in eligible if r['metrics']['dev']['path']['correct']>=384]
  vrbest=sorted(maintained,key=lambda r:(r['metrics']['dev']['AP']['variance_reduction'],r['metrics']['dev']['path']['correct']),reverse=True)[:3]
  pathbest=sorted(eligible,key=lambda r:(r['metrics']['dev']['path']['correct'],r['metrics']['dev']['AP']['variance_reduction']),reverse=True)[:3]
  selected[str(tier)]=dict(highest_VR_preserve_path=vrbest,highest_path=pathbest)
  save.update(r['config']['name'] for r in vrbest+pathbest)
 for name in save:
  c=byname[name]['config'];delta=deltas[c['parent']]*c['gain']
  np.savez_compressed(c['score_file'],delta=delta,b=c['new_b'])
 dump(OUT/'path_calibrated_selection.json',dict(selection_split='dev',test_evaluated=False,total_candidates=len(result),min_natural_correct=195,raw_path_correct=384,tiers=selected))
 dump(OUT/'path_bias_done.json',dict(state='complete',candidates=len(result),selected_unique=len(save),seconds=time.time()-start))
 print(json.dumps({tier:{key:[dict(name=r['config']['name'],family=r['config']['family'],VR=r['metrics']['dev']['AP']['variance_reduction'],path=r['metrics']['dev']['path']['correct'],natural=r['metrics']['dev']['natural']['correct'],ER=r['metrics']['dev']['CF']['mean_evidence_retention'],b=r['config']['new_b']) for r in rr] for key,rr in val.items()} for tier,val in selected.items()}))
if __name__=='__main__':main()

