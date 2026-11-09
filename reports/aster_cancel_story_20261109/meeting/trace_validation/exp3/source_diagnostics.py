"""Summarize true held-out sources and inspect source context; no model fitting."""
import csv,gzip,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from run_cv import metrics,table,dump,rows,ROOT

def run():
 with gzip.open(HERE.parent/'exp12/pair_predictions.csv.gz','rt') as f:
  pp=[p for p in csv.DictReader(f) if p['method']=='selected_N16' and p['split']=='dev']
 for p in pp:
  p['target']=float(p['target']);p['prediction']=float(p['prediction'])
 rr=[]
 for source in sorted({p['source_id'] for p in pp}):
  for group,kinds in [('A',('A',)),('AP',('A','P'))]:
   ps=[p for p in pp if p['source_id']==source and p['kind'] in kinds]
   if ps:rr.append(dict(split='dev',method='selected_N16',group=group,source_id=source,**metrics([p['target'] for p in ps],[p['prediction'] for p in ps])))
 table(HERE/'dev_source_metrics.csv',rr)
 ap=sorted([x for x in rr if x['group']=='AP' and x['r2'] is not None],key=lambda x:x['r2'])
 dump(HERE/'dev_source_summary.json',dict(sources=len(ap),positive_source_r2=sum(x['r2']>0 for x in ap),median_source_r2=float(np.median([x['r2'] for x in ap])),worst_sources=ap[:5],best_sources=ap[-5:],evidence_type='real_cache_prediction_recomputation'))
 views=rows(ROOT/'data/views.jsonl');sources=rows(ROOT/'data/sources.jsonl');nodes=rows(ROOT/'data/nodes.jsonl')
 fit_metrics=list(csv.DictReader((HERE/'frozen_heads_source_metrics.csv').open()))
 fit_metrics.sort(key=lambda x:float(x['r2']) if x['r2'] else np.inf)
 full_metrics=list(csv.DictReader((HERE/'full/source_metrics.csv').open())) if (HERE/'full/source_metrics.csv').exists() else []
 full_metrics.sort(key=lambda x:float(x['r2']) if x['r2'] else np.inf)
 selected=list(dict.fromkeys([x['source_id'] for x in ap[:3]]+[x['source_id'] for x in full_metrics[:3]]+[x['source_id'] for x in fit_metrics[:3]]))
 details=[]
 for source in selected:
  vv=[v for v in views if v['source_id']==source and v['variant']=='full']
  nn=[n for n in nodes if n['source_id']==source]
  summary=next(x for x in sources if x['source_id']==source)
  prompts=[v['prompt'] for v in vv]
  conditions=[]
  for v in vv:
   prompt=v['prompt'];part=prompt.split('\u5f53\u524d\u6761\u4ef6\uff1a')[-1].split('\u8bf7\u5224\u65ad\u5f53\u524d\u6761\u4ef6')[0].strip()
   if part not in conditions:conditions.append(part)
  entry=dict(source_id=source,split=vv[0]['split'],guideline_ids=summary['guideline_ids'],source_filename=nn[0].get('source_filename'),full_views=len(vv),label_counts={str(k):sum(v['label']==k for v in vv) for k in [0,1]},prompt_char_median=float(np.median([len(x) for x in prompts])),conditions=conditions[:12])
  m=next((x for x in ap if x['source_id']==source),None)
  if m:entry['prediction_metrics']=m
  else:
   entry['prediction_metrics']=next(x for x in (full_metrics or fit_metrics) if x['source_id']==source)
   entry['frozen_prediction_metrics']=next(x for x in fit_metrics if x['source_id']==source)
  sp=[p for p in pp if p['source_id']==source]
  if not sp:
   file=HERE/'full/pair_predictions.csv' if full_metrics else HERE/'frozen_heads_pair_predictions.csv'
   with file.open() as f:sp=[p for p in csv.DictReader(f) if p['source_id']==source]
  for p in sp:p['target']=float(p['target']);p['prediction']=float(p['prediction'])
  entry['target_mean']=float(np.mean([p['target'] for p in sp]));entry['prediction_mean']=float(np.mean([p['prediction'] for p in sp]));entry['prediction_sd']=float(np.std([p['prediction'] for p in sp]))
  entry['observation']='R2 compares squared error against this source target variance. Low target variance can give a negative R2 at modest RMSE; task wording and context are descriptive observations, not identified causes.'
  details.append(entry)
 dump(HERE/'source_context_diagnostics.json',details)
 if full_metrics:
  fi={x['source_id']:x for x in fit_metrics}
  comparison=[dict(source_id=x['source_id'],fold=x['fold'],n=x['n'],frozen_r2=fi[x['source_id']]['r2'],full_r2=x['r2'],r2_full_minus_frozen=float(x['r2'])-float(fi[x['source_id']]['r2']),frozen_rmse=fi[x['source_id']]['rmse'],full_rmse=x['rmse']) for x in full_metrics]
  table(HERE/'source_cv_comparison.csv',comparison)
 print(json.dumps(dict(dev_sources=len(ap),dev_positive=sum(x['r2']>0 for x in ap),dev_median=float(np.median([x['r2'] for x in ap])),dev_worst=ap[:3]),ensure_ascii=False))
if __name__=='__main__':run()
