
"""Prepare allowed fit/dev cached 99-dimensional features for six fixed runs."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from pathlib import Path
import json,numpy as np
BASE=Path('/home/gcx/aster_hard_optimization_20261109')
SH=Path('/home/gcx/aster_optimization_20261109/shared')
ORIGINAL=Path('/home/gcx/aster_cr_repair_v2_run_20261108')
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 BASE.mkdir(exist_ok=True);a=dict(np.load(SH/'features.npz'));ix={v:i for i,v in enumerate(a['ids'].tolist())}
 meta=json.loads((SH/'metadata.json').read_text());views=[v for v in meta['views'] if v['split'] in ('fit','dev')]
 paths=[r for r in meta['paths'] if r['split']=='dev']
 ids=[v['view_id'] for v in views]+['path|'+r['id'] for r in paths]
 pos=np.array([ix[v] for v in ids]);ell=a['ell'][pos];score=ell.astype('float32')
 q=np.column_stack([score/10,np.abs(score)/10,np.tanh(score/4)]).astype('float32')
 x=np.concatenate([a['r32'][pos],a['residual64'][pos],q],axis=1).astype('float32')
 labels=np.array([v['label'] for v in views]+[-1]*len(paths),dtype='int8')
 source_ids=[v['source_id'] for v in views]+[r['source_id'] for r in paths]
 aug=dict(np.load(BASE/'fit_augmentation/features.npz'))
 fit_sources={v['source_id'] for v in views if v['split']=='fit'}
 assert set(aug['source_ids'].tolist())<=fit_sources
 assert not (set(ids)&set(aug['ids'].tolist()))
 offset=len(ids);ids+=aug['ids'].tolist();source_ids+=aug['source_ids'].tolist()
 x=np.concatenate([x,aug['combined_score'].astype('float32')]);ell=np.concatenate([ell,aug['ell']]);labels=np.concatenate([labels,aug['labels'].astype('int8')])
 outix={v:i for i,v in enumerate(ids)}
 natural=[v for v in views if v['split']=='fit' and v['variant']=='full' and v['original']]
 ni=np.array([outix[v['view_id']] for v in natural]);extra=np.arange(offset,len(ids))
 ce=np.concatenate([ni,extra]);weights=[]
 for inds in (ni,extra):
  ss=[source_ids[i] for i in inds];counts={s:ss.count(s) for s in set(ss)}
  weights.extend([.5/len(counts)/counts[s] for s in ss])
 np.savez_compressed(BASE/'data.npz',ids=np.array(ids),x=x,ell=ell,labels=labels,source_ids=np.array(source_ids),natural_fit=ni,extra_fit=extra,ce_indices=ce,ce_weights=np.array(weights,dtype='float32'))
 cases=[c for c in rows(ORIGINAL/'data/path_cases.jsonl') if c['split']=='dev']
 original_weights='/home/gcx/aster_cancel_story_20261109/meeting/evidence_preserving/weights.npz'
 frozen=json.loads(Path('/home/gcx/aster_next_20261109/frozen_methods.json').read_text())
 entry=frozen['methods']['optimized_balanced_fit_calibration']
 dump(BASE/'data.json',dict(views=views,pairs=[p for p in meta['pairs'] if p['split'] in ('fit','dev')],dev_paths=paths,dev_cases=cases,
   fit_sources=sorted(fit_sources),augmentation=str(BASE/'fit_augmentation/inventory.json'),original_weights=original_weights,initial_balanced=entry))
 dump(BASE/'plan.json',dict(schemes=['tail_pair','tail_robust'],seeds=[11,22,33],runs=6,steps=1200,lr=.0003,weight_decay=.001,width=128,
  input_dimension=99,ap_tail_fraction=.2,cf_tail_fraction=.2,cf_weight=1000.,cf_hinge_weight=1000.,cf_jitter=.1,
  anchor_weight=.01,ce_weight=1.,ce_pool_weights=dict(original_natural=.5,augmented_fit=.5),ce_source_balanced=True,
  calibration='After fixing correction and fit-budget scale, original 761 natural fit CE only',fit_scale='min(1, .1*fit_AP_raw_RMS / max_abs_fit_CF_correction)',
  initialization='Corresponding seed old balanced member with its output layer scaled by the frozen ensemble gain',selection='No checkpoint, width or hyperparameter search; report both fixed 3-seed ensembles',gpu=[0,1],
  fit_AP=4887,fit_CF=135,natural_fit=len(ni),extra_fit=len(extra),dev_views=sum(v['split']=='dev' for v in views),dev_cases=len(cases)))
 print('prepared allowed fit/dev data')
if __name__=='__main__':main()

