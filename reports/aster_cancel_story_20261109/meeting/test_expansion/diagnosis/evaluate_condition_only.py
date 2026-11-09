
"""Summarize one frozen prompt on fit/dev only; no test data is read."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from pathlib import Path
import argparse,json,time,csv,datetime
from collections import Counter,defaultdict
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit
BASE=Path('/home/gcx/aster_next_20261109')
OUT=BASE/'diagnosis'
RESULTS=BASE/'condition_only_results'
OLD=Path('/home/gcx/aster_cancel_story_20261109/meeting/real_completion')
DATA=Path('/home/gcx/aster_cr_repair_v2_run_20261108/data')
POLICY=dict(variants=['raw','prompt_zero','prompt_fit_ce'],
 candidate_count=1,calibration_candidates=[0.0,'single scalar fit-natural CE optimum'],
 intercept_search_on_dev=False,
 recommendation='Among prompt_zero and prompt_fit_ce, require dev natural >= raw and dev exact paths >= raw; maximize exact paths, then natural nodes, then prefer prefit CE calibration. If neither qualifies, keep raw for official comparison and report candidate as not accepted.',
 bootstrap=dict(grouping='source_id',replicates=2000,seed=11))
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()] if Path(p).exists() else []
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def csvwrite(p,rr):
 fields=list(dict.fromkeys(k for r in rr for k in r))
 with Path(p).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in rr:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
def ci(x):
 x=np.asarray(x);x=x[np.isfinite(x)]
 return np.quantile(x,[.025,.975]).tolist() if len(x) else None
def result_index(rr):
 index={};duplicates=0
 for r in rr:
  if r['id'] in index:duplicates+=1
  index[r['id']]=r
 return index,duplicates
def complete(manifest,result):
 index,duplicate=result_index(result)
 missing=[r['id'] for r in manifest if r['id'] not in index or index[r['id']].get('status')!='ok']
 if missing:raise RuntimeError('Incomplete prompt results; missing/failed IDs: '+str(missing[:8])+' total='+str(len(missing)))
 return index,duplicate
def infer_aliases(manifest,index):
 result={}
 for item in manifest:
  rec=index[item['id']]
  for name in [item['id'],item['view_id']]+item.get('aliases',[]):result[name]=float(rec['ell'])
 return result
def aggregate(values,sources,sourceix):
 values=np.asarray(values,dtype=float)
 a=np.zeros((len(sourceix),2))
 np.add.at(a[:,0],[sourceix[s] for s in sources],1)
 np.add.at(a[:,1],[sourceix[s] for s in sources],values)
 return a
def evaluate(wait=False):
 OUT.mkdir(exist_ok=True)
 policy_file=OUT/'condition_only_evaluation_policy.json'
 if not policy_file.exists():dump(policy_file,dict(created_at=now(),**POLICY))
 if wait:
  while not (RESULTS/'done.json').exists():
   if (RESULTS/'not_advanced.json').exists():raise RuntimeError('Candidate stopped after quick development check')
   if (RESULTS/'failed.json').exists():raise RuntimeError('Prompt worker failed: '+str(RESULTS/'failed.json'))
   time.sleep(10)
 if not (RESULTS/'done.json').exists():raise RuntimeError('Prompt GPU scoring not complete yet; use --wait')
 manifest_dev=rows(OUT/'condition_only_v1.dev_full_gold.jsonl')
 manifest_fit=rows(OUT/'condition_only_v1.fit_natural.jsonl')
 manifest_natural=rows(OUT/'condition_only_v1.dev_natural.jsonl')
 manifest_focus=rows(OUT/'condition_only_v1.dev_focus.jsonl')
 dev_results=rows(RESULTS/'dev_full_gold.jsonl');fit_results=rows(RESULTS/'fit_natural.jsonl')
 dev_index,dupd=complete(manifest_dev,dev_results);fit_index,dupf=complete(manifest_fit,fit_results)
 assert len(manifest_dev)==2568 and len(manifest_fit)==761
 fit_labels=np.array([r['label'] for r in manifest_fit])
 fit_score=np.array([fit_index[r['id']]['ell'] for r in manifest_fit])
 fn=lambda b:float(np.mean(expit(fit_score+b)-fit_labels))
 lo,hi=-1.,1.
 while fn(lo)>0:lo*=2
 while fn(hi)<0:hi*=2
 bias=float(brentq(fn,lo,hi,xtol=1e-12))
 # Persist the fitted calibration before any selection/report; this is fitted
 # exclusively to the 761 fit labels, without dev threshold optimization.
 dump(OUT/'condition_only_fit_calibration.json',dict(saved_at=now(),method='condition_only_v1',fit_n=761,b=bias,
  objective='mean binary cross entropy of prompt_score+b on fit natural nodes',
  dev_or_test_labels_used=False,source_result_file=str(RESULTS/'fit_natural.jsonl')))
 candidate_scores=infer_aliases(manifest_dev,dev_index)
 raw_nodes={}
 for p in Path('/home/gcx/aster_cr_repair_v2_run_20261108/qwen/paths').glob('features_dev_*.jsonl'):
  for r in rows(p):
   if r.get('status')=='ok':raw_nodes['path|'+r['id']]=float(r['ell'])
 old=np.load(OLD/'coordinates.npz');oldix={v:i for i,v in enumerate(old['ids'].tolist())}
 raw_views={r['id']:float(old['ell'][oldix[r['view_id']]]) for r in manifest_fit+manifest_natural}
 cases=[c for c in rows(DATA/'path_cases.jsonl') if c['split']=='dev']
 natural_labels=np.array([r['label'] for r in manifest_natural])
 raw_natural=np.array([raw_views[r['id']] for r in manifest_natural])
 prompt_natural=np.array([candidate_scores[r['id']] for r in manifest_natural])
 methods={'raw':dict(b=0.),'prompt_zero':dict(b=0.),'prompt_fit_ce':dict(b=bias)}
 pathrows=[];node_rows=[];natural_rows=[];focus_rows=[]
 vectors={};summary={}
 for method,cfg in methods.items():
  b=cfg['b'];israw=method=='raw'
  nscore=raw_natural if israw else prompt_natural+b
  natural_correct=(nscore>0)==natural_labels
  node_correct=[];node_sources=[];path_correct=[];path_sources=[];first_errors=[]
  for case in cases:
   checks=[]
   for j,g in enumerate(case['gold_path']):
    key='path|'+case['case_id']+':'+g['gid']
    s=raw_nodes[key] if israw else candidate_scores[key]+b
    correct=int(s>0)==int(g['label'])
    checks.append(correct);node_correct.append(correct);node_sources.append(case['source_id'])
    node_rows.append(dict(method=method,case_id=case['case_id'],source_id=case['source_id'],gid=g['gid'],gold_order=j,label=g['label'],ell=float(s),correct=bool(correct)))
   correct=all(checks);path_correct.append(correct);path_sources.append(case['source_id'])
   first=next((case['gold_path'][j]['gid'] for j,z in enumerate(checks) if not z),None)
   pathrows.append(dict(method=method,case_id=case['case_id'],source_id=case['source_id'],correct=correct,first_error=first,
    gold_nodes=len(checks),correct_gold_nodes=sum(checks)))
  for j,r in enumerate(manifest_natural):
   natural_rows.append(dict(method=method,view_id=r['view_id'],case_id=r['case_id'],source_id=r['source_id'],gid=r['gid'],label=r['label'],ell=float(nscore[j]),correct=bool(natural_correct[j])))
  focus_correct=[]
  for r in manifest_focus:
   s=raw_nodes[r['id']] if israw else candidate_scores[r['id']]+b
   correct=int(s>0)==int(r['label']);focus_correct.append(correct)
   focus_rows.append(dict(method=method,id=r['id'],case_id=r['case_id'],source_id=r['source_id'],gid=r['gid'],label=r['label'],ell=float(s),correct=correct))
  actual_fit=raw_views if israw else {r['id']:float(fit_index[r['id']]['ell'])+b for r in manifest_fit}
  fit_cor=np.array([int(actual_fit[r['id']]>0)==r['label'] for r in manifest_fit])
  vectors[method]=dict(path=(np.array(path_correct,dtype=float),path_sources),
   gold_nodes=(np.array(node_correct,dtype=float),node_sources),
   natural=(natural_correct.astype(float),[r['source_id'] for r in manifest_natural]))
  summary[method]=dict(b=b,fit_natural=dict(n=761,correct=int(fit_cor.sum()),accuracy=float(fit_cor.mean())),
   dev_paths=dict(n=len(path_correct),correct=sum(path_correct),accuracy=float(np.mean(path_correct))),
   dev_gold_nodes=dict(n=len(node_correct),correct=sum(node_correct),accuracy=float(np.mean(node_correct))),
   dev_natural=dict(n=199,correct=int(natural_correct.sum()),accuracy=float(np.mean(natural_correct))),
   dev_PMA_diagnostic_family=dict(n=len(focus_correct),correct=sum(focus_correct),accuracy=float(np.mean(focus_correct)),sources=1,
    interpretation='Previously identified dev diagnostic family; descriptive only, not an independent test.'))
 sourceids=sorted({c['source_id'] for c in cases});sourceix={s:i for i,s in enumerate(sourceids)}
 draws=np.random.default_rng(11).multinomial(len(sourceids),np.full(len(sourceids),1/len(sourceids)),size=2000)
 bootstrap={}
 rep={}
 for method,groups in vectors.items():
  bootstrap[method]={}
  for group,(values,ss) in groups.items():
   z=draws@aggregate(values,ss,sourceix)
   with np.errstate(divide='ignore',invalid='ignore'):r=z[:,1]/z[:,0]
   rep[method,group]=r
   bootstrap[method][group]=dict(accuracy=float(np.mean(values)),ci=ci(r))
 for method in ('prompt_zero','prompt_fit_ce'):
  for group in vectors[method]:
   bootstrap[method][group]['difference_vs_raw']=summary[method]['dev_'+({'path':'paths','gold_nodes':'gold_nodes','natural':'natural'}[group])]['accuracy']-summary['raw']['dev_'+({'path':'paths','gold_nodes':'gold_nodes','natural':'natural'}[group])]['accuracy']
   bootstrap[method][group]['difference_vs_raw_ci']=ci(rep[method,group]-rep['raw',group])
 transitions={}
 by_path={m:{r['case_id']:r for r in pathrows if r['method']==m} for m in methods}
 by_node={m:{(r['case_id'],r['gid']):r for r in node_rows if r['method']==m} for m in methods}
 by_natural={m:{r['view_id']:r for r in natural_rows if r['method']==m} for m in methods}
 for method in ('prompt_zero','prompt_fit_ce'):
  groups={}
  for kind,lookup in [('paths',by_path),('gold_nodes',by_node),('natural',by_natural)]:
   fixed=[key for key,r in lookup['raw'].items() if not r['correct'] and lookup[method][key]['correct']]
   broken=[key for key,r in lookup['raw'].items() if r['correct'] and not lookup[method][key]['correct']]
   groups[kind]=dict(fixed=fixed,broken=broken,fixed_count=len(fixed),broken_count=len(broken))
  transitions[method]=groups
 raw_p=summary['raw']['dev_paths']['correct'];raw_n=summary['raw']['dev_natural']['correct']
 eligible=[m for m in ('prompt_zero','prompt_fit_ce') if summary[m]['dev_paths']['correct']>=raw_p and summary[m]['dev_natural']['correct']>=raw_n]
 chosen=max(eligible,key=lambda m:(summary[m]['dev_paths']['correct'],summary[m]['dev_natural']['correct'],m=='prompt_fit_ce')) if eligible else 'raw'
 suffix='candidate_accepted' if chosen!='raw' else 'candidate_not_accepted'
 result=dict(method='condition_only_v1',scope='fit/dev only',completed_at=now(),fit_ce_b=bias,policy=POLICY,
  summary=summary,source_bootstrap=bootstrap,source_count=len(sourceids),paired_changes_vs_raw=transitions,
  recommendation=dict(variant=chosen,b=methods[chosen]['b'],status=suffix,eligible_variants=eligible),
  coverage=dict(fit_results=len(fit_index),dev_results=len(dev_index),expected_fit=761,expected_dev_gold=2568,
   dev_natural=199,PMA_focus=19,duplicate_fit_log_rows=dupf,duplicate_dev_log_rows=dupd),
  path_metric_note='For this fixed node prompt, every node score is independent of the traversed decision prefix. The autonomous path is exactly gold iff all gold-node decisions are correct. Only gold nodes were scored; wrong off-gold trajectories are not reconstructed.',
  grouping_note='420 case paths have20 source clusters. PMA19 cases come from one diagnosed development source; report separately.',
  no_official_or_old_test_data_used=True)
 csvwrite(OUT/'condition_only_path_results.csv',pathrows);csvwrite(OUT/'condition_only_gold_node_results.csv',node_rows)
 csvwrite(OUT/'condition_only_natural_results.csv',natural_rows);csvwrite(OUT/'condition_only_PMA_results.csv',focus_rows)
 dump(OUT/'condition_only_summary.json',result)
 frozen=dict(frozen=True,saved_at=now(),method='condition_only_v1',
  prompt_metadata=str(OUT/'condition_only_v1.json'),prompt_construction_module=str(OUT/'condition_only.py'),
  input_control='condition_only',candidate_count=1,
  selection_split='dev',selected_variant=chosen,recommended_b=methods[chosen]['b'],fit_ce_b=bias,
  calibration=dict(fit_n=761,fit_labels_only=True,dev_threshold_search=False,
   fit_record=str(OUT/'condition_only_fit_calibration.json')),
  acceptance=result['recommendation'],fit_dev_metrics=summary,source_bootstrap=bootstrap,
  official_or_old_test_used=False,prompt_changed_after_scoring=False,
  summary_path=str(OUT/'condition_only_summary.json'))
 dump(OUT/'condition_only_candidate_frozen.json',frozen)
 print(json.dumps(dict(state='complete',fit_ce_b=bias,summary=summary,recommended=chosen,recommended_b=methods[chosen]['b'])))
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');args=p.parse_args()
 evaluate(wait=args.wait)

