
"""Prepare supervised gold-node cache additions from original fit sources only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
import json,sys,datetime
from collections import Counter,defaultdict
import numpy as np
BASE=Path('/home/gcx/aster_next_20261109')
ROOT=Path('/home/gcx/aster_cr_repair_v2_run_20261108')
OLD=Path('/home/gcx/aster_cancel_story_20261109/meeting/real_completion')
OUT=Path('/home/gcx/aster_hard_optimization_20261109/fit_augmentation')
def rows(p):
 out=[]
 for line in Path(p).read_text().splitlines():
  if not line.strip():continue
  try:out.append(json.loads(line))
  except json.JSONDecodeError:continue
 return out
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 original=rows(ROOT/'data/path_cases.jsonl')
 fit={c['case_id']:c for c in original if c['split']=='fit'}
 fit_sources={c['source_id'] for c in fit.values()}
 excluded_sources={c['source_id'] for c in original if c['split'] in ('dev','test')}
 if fit_sources&excluded_sources:raise RuntimeError('source split overlap')
 meta=json.loads((OLD/'metadata.json').read_text())
 natural=[v for v in meta['views'] if v['split']=='fit' and v['variant']=='full' and v['original']]
 natural_keys={(v['case_id'],v['group_id'].rsplit('-',1)[-1]) for v in natural}
 cached={}
 node_files=sorted((BASE/'primary_results/mega_train_val').glob('shard*/nodes.jsonl'))
 for p in node_files:
  for r in rows(p):
   cid=r.get('case_id')
   if cid not in fit or r.get('source_id') not in fit_sources:continue
   if r.get('source_id') in excluded_sources:continue
   if r.get('input_variant','canonical')!='canonical' or r.get('status')!='ok':continue
   if not r.get('feature') or not Path(r['feature']).is_file():continue
   cached[cid,r['gid']]=dict(r,node_index_file=str(p))
 labels={(cid,g['gid']):int(g['label']) for cid,c in fit.items() for g in c['gold_path']}
 gold_keys=set(cached)&set(labels)
 additions=sorted(gold_keys-natural_keys)
 sys.path.insert(0,str(BASE/'inference'));from frozen import FrozenMethods
 adapter=FrozenMethods(BASE/'frozen_methods.json')
 balanced_name='optimized_balanced_fit_calibration'
 b=float(adapter.manifest['methods'][balanced_name]['b'])
 metadata=[];arrays=defaultdict(list)
 for key in additions:
  cid,gid=key;r=cached[key];label=labels[key]
  with np.load(r['feature']) as z:
   selected=adapter.raw_selected(z);residual=z['residual'].astype('float32')
  coordinates=adapter.coordinates(r['ell'],selected,residual)
  predictions=adapter.predict(r['ell'],selected,residual)
  identifier='fitaug|'+cid+':'+gid
  full_score=float(predictions[balanced_name]);delta=full_score-float(r['ell'])-b
  entry=dict(id=identifier,case_id=cid,gid=gid,source_id=r['source_id'],split='fit',label=label,ell=float(r['ell']),
   feature=r['feature'],node_index_file=r['node_index_file'],
   origin='original fit case gold-path node with existing canonical feature',
   original_fit_case=True,original_fit_source=True,old_natural_case_gid_overlap=False,
   old_linear_score=float(predictions['old_linear']),old_linear_correct=int(predictions['old_linear']>0)==label,
   base_balanced_score=full_score,base_balanced_delta=delta,base_balanced_b=b,
   raw_correct=int(r['ell']>0)==label,base_balanced_correct=int(full_score>0)==label)
  metadata.append(entry)
  arrays['ids'].append(identifier);arrays['source_ids'].append(r['source_id']);arrays['labels'].append(label);arrays['ell'].append(float(r['ell']))
  for name in ('r16','r32','residual64','q'):arrays[name].append(coordinates[name])
  arrays['combined_score'].append(np.concatenate([coordinates['r32'],coordinates['residual64'],coordinates['q']]))
  arrays['old_linear_score'].append(entry['old_linear_score']);arrays['base_balanced_score'].append(full_score)
  arrays['base_balanced_delta'].append(delta)
 values={}
 for k,v in arrays.items():
  if k in ('ids','source_ids'):values[k]=np.array(v)
  elif k=='labels':values[k]=np.array(v,dtype=np.int8)
  elif k in ('r16','r32','residual64','q','combined_score'):values[k]=np.stack(v).astype('float32')
  else:values[k]=np.array(v,dtype='float64')
 values['id']=values['ids'];values['source_id']=values['source_ids'];values['label']=values['labels'];values['base_balanced_b']=np.array(b)
 np.savez_compressed(OUT/'features.npz',**values)
 with (OUT/'metadata.jsonl').open('w',encoding='utf-8') as f:
  for r in metadata:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 original_fit_gold_scores=[]
 for key in sorted(gold_keys):
  r=cached[key]
  with np.load(r['feature']) as z:
   selected=adapter.raw_selected(z);residual=z['residual'].astype('float32')
  prediction=adapter.predict(r['ell'],selected,residual)
  original_fit_gold_scores.append(dict(raw_wrong=int(r['ell']>0)!=labels[key],
   old_linear_wrong=int(prediction['old_linear']>0)!=labels[key],
   base_balanced_wrong=int(prediction[balanced_name]>0)!=labels[key]))
 inventory=dict(saved_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  original_fit_sources=len(fit_sources),original_fit_cases=len(fit),
  cached_fit_gold_nodes=len(gold_keys),cached_fit_cases=len({k[0] for k in gold_keys}),cached_fit_sources=len({cached[k]['source_id'] for k in gold_keys}),
  original_natural_nodes=len(natural),overlap_case_gid=len(gold_keys&natural_keys),new_supervised_nodes=len(additions),
  addition_sources=sorted({r['source_id'] for r in metadata}),
  source_counts=dict(Counter(r['source_id'] for r in metadata)),label_counts=dict(Counter(str(r['label']) for r in metadata)),
  additional_wrong_counts=dict(raw=sum(not r['raw_correct'] for r in metadata),old_linear=sum(not r['old_linear_correct'] for r in metadata),base_balanced=sum(not r['base_balanced_correct'] for r in metadata)),
  all_cached_gold_wrong_counts={k:sum(r[k] for r in original_fit_gold_scores) for k in ('raw_wrong','old_linear_wrong','base_balanced_wrong')},
  coordinate_shapes={k:list(v.shape) for k,v in values.items()},
  coordinate_definition='Exact unnormalized inputs from old FrozenMethods.coordinates: r16, r32, residual64, q=(raw_ell/10,abs(raw_ell)/10,tanh(raw_ell/4)), combined_score=concat(r32,residual64,q).',
  base_delta_definition='base_balanced_delta is final frozen gain-weighted correction before its existing base_balanced_b; base_balanced_score=ell+base_balanced_delta+base_balanced_b.',
  deduplication='Remove existing natural-fit case_id/gid keys only. Rows are supervised case-node pairs; not asserted independent sources.',
  no_dev_test_or_unassigned_sources=True,no_new_model_forward=True,no_training=True,
  source_case_file=str(ROOT/'data/path_cases.jsonl'),source_node_files=[str(p) for p in node_files],frozen_manifest=str(BASE/'frozen_methods.json'))
 (OUT/'inventory.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(inventory,ensure_ascii=True))
if __name__=='__main__':main()

