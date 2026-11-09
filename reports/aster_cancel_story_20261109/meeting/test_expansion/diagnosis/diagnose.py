
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from pathlib import Path
import json,csv,sys,re,time
from collections import defaultdict,Counter
import numpy as np
ROOT=Path('/home/gcx/aster_cr_repair_v2_run_20261108')
OLD=Path('/home/gcx/aster_cancel_story_20261109/meeting/real_completion')
OPT=Path('/home/gcx/aster_optimization_20261109')
OUT=Path('/home/gcx/aster_next_20261109/diagnosis')
OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,'/home/gcx/TRIAGE-worktrees/aster-cr-20261108')
from experiments.aster_cr.paths import execute
from experiments.aster_closed_form.data import canonical_guide,prompt
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()] if Path(p).exists() else []
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def savecsv(p,rr):
 fields=list(dict.fromkeys(k for r in rr for k in r))
 with Path(p).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in rr:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else v for k,v in r.items()})
def main():
 start=time.time();frozen=json.loads((OPT/'final/selected_calibration_ablation.json').read_text())
 metadata=json.loads((OPT/'shared/metadata.json').read_text())
 allfeatures=np.load(OPT/'shared/features.npz')
 fitdevviews=[v for v in metadata['views'] if v['split'] in ('fit','dev')]
 devpaths=[r for r in metadata['paths'] if r['split']=='dev']
 allowed=[v['view_id'] for v in fitdevviews]+['path|'+r['id'] for r in devpaths]
 fullix={v:i for i,v in enumerate(allfeatures['ids'].tolist())}
 pos=np.array([fullix[v] for v in allowed]);ix={v:i for i,v in enumerate(allowed)}
 ell=allfeatures['ell'][pos].astype(float);t=allfeatures['r16'][pos]
 scores={'raw':ell.copy()};corrections={'raw':np.zeros(len(ell))}
 oldw=np.load(OLD/'../evidence_preserving/weights.npz')
 corrections['old_linear']=t@oldw['bounded_0.1']
 scores['old_linear']=ell+corrections['old_linear']+float(oldw['bounded_0.1_b'])
 configs={'raw':dict(b=0.),'old_linear':dict(b=float(oldw['bounded_0.1_b']),rank=16,lambda_=0.01,tau=.1)}
 for method,key in [('balanced_fit_ce','optimized_balanced_fit_calibration'),('strict_fit_ce','optimized_strict_fit_calibration')]:
  c=frozen['methods'][key];a=np.load(c['prediction_file']);predix={s:i for i,s in enumerate(a['ids'].tolist())}
  index=np.array([predix[v] for v in allowed]);delta=a['delta'][index]*float(c['gain'])
  corrections[method]=delta;scores[method]=ell+delta+float(c['b'])
  configs[method]=dict(frozen_name=key,b=float(c['b']),gain=float(c['gain']),prediction_file=c['prediction_file'],kind=c['kind'])
 sourceviews=[v for v in rows(ROOT/'data/views.jsonl') if v['split'] in ('fit','dev')]
 byview={v['view_id']:v for v in sourceviews}
 exactfull=defaultdict(list)
 for v in sourceviews:
  if v['variant']=='full':exactfull[v['prompt']].append(v)
 node_by_group={v['group_id']:v for v in rows(ROOT/'data/nodes.jsonl') if v['split'] in ('fit','dev')}
 rules_by_group={v['group_id']:v for v in rows(ROOT/'data/rules.jsonl') if v['split'] in ('fit','dev')}
 pairs=[p for p in metadata['pairs'] if p['split'] in ('fit','dev')]
 aps=defaultdict(list)
 for p in pairs:
  if p['kind'] in ('A','P'):
   aps[p['left']].append(p);aps[p['right']].append(p)
 cases=[c for c in rows(ROOT/'data/path_cases.jsonl') if c['split']=='dev']
 rawpaths={r['id']:r for r in devpaths}
 records=[];path_results=[];examples=[];gold_node_records=[]
 fitn=[v for v in fitdevviews if v['split']=='fit' and v['variant']=='full' and v['original']]
 fi=np.array([ix[v['view_id']] for v in fitn])
 for method in scores:
  cbar=float(np.mean(corrections[method][fi]))
  configs[method]['fit_natural_mean_correction']=cbar
  configs[method]['effective_bias_after_centering_correction']=configs[method]['b']+cbar
 for case in cases:
  cid=case['case_id'];guide=canonical_guide(case['guideline']);nodes={n['id']:n for n in guide['nodes']}
  ns={method:{n['id']:float(scores[method][ix['path|'+cid+':'+n['id']]]) for n in guide['nodes'] if 'path|'+cid+':'+n['id'] in ix} for method in scores}
  bymethod={}
  for method in scores:
   r=execute(case,ns[method]);bymethod[method]=r
   path_results.append(dict(method=method,case_id=cid,source_id=case['source_id'],**r))
  for order,gold in enumerate(case['gold_path']):
   gid=gold['gid'];label=int(gold['label']);key='path|'+cid+':'+gid
   values={method:float(scores[method][ix[key]]) for method in scores}
   correct={method:int(values[method]>0)==label for method in scores}
   anyerror=not all(correct.values())
   condition=nodes[gid]['condition']
   # Exact prompt matching ensures A/P coverage concerns the same patient,
   # condition, definitions and original input, not merely a shared tree node.
   pp=prompt(case['guideline'],case['definitions'],case['patient'],gid)
   exactviews=exactfull.get(pp,[])
   matched_ids=[v['view_id'] for v in exactviews]
   pairlist={p['pair_id']:p for vid in matched_ids for p in aps.get(vid,[])}
   apinfo={}
   for method in scores:
    eligible=[]
    for p in pairlist.values():
     l=ix[p['left']];r=ix[p['right']]
     eligible.append(dict(pair_id=p['pair_id'],kind=p['kind'],left_score=float(scores[method][l]),right_score=float(scores[method][r]),
      left_correct=int(scores[method][l]>0)==int(p['left_label']),right_correct=int(scores[method][r]>0)==int(p['right_label']),
      delta_score=float(scores[method][r]-scores[method][l])))
    flips=[p for p in eligible if (p['left_score']>0)!=(p['right_score']>0)]
    allside=[z for p in eligible for z in (p['left_correct'],p['right_correct'])]
    apinfo[method]=dict(pairs=len(eligible),prediction_changing_pairs=len(flips),
      any_correct_endpoint=bool(any(allside)),all_endpoints_wrong=bool(allside) and not any(allside),
      max_abs_difference=max((abs(p['delta_score']) for p in eligible),default=None),pair_details=eligible)
   threshold_bin='abs_raw_le_1' if abs(values['raw'])<=1 else 'abs_raw_1_to_3' if abs(values['raw'])<=3 else 'abs_raw_3_to_5' if abs(values['raw'])<=5 else 'abs_raw_gt_5'
   rec=dict(case_id=cid,source_id=case['source_id'],gid=gid,gold_order=order,gold_label=label,condition=condition,
    raw_score=values['raw'],old_linear_score=values['old_linear'],balanced_score=values['balanced_fit_ce'],strict_score=values['strict_fit_ce'],
    raw_correct=correct['raw'],old_linear_correct=correct['old_linear'],balanced_correct=correct['balanced_fit_ce'],strict_correct=correct['strict_fit_ce'],
    threshold_bin=threshold_bin,all_four_wrong=not any(correct.values()),same_exact_prompt_views=matched_ids,
    observed_AP_pairs=len(pairlist),raw_AP_prediction_changing_pairs=apinfo['raw']['prediction_changing_pairs'],
    raw_wrong_with_correct_AP_endpoint=(not correct['raw']) and apinfo['raw']['any_correct_endpoint'],
    all_observed_AP_endpoints_wrong=apinfo['raw']['all_endpoints_wrong'],
    balanced_input_correction_centered=float(corrections['balanced_fit_ce'][ix[key]]-configs['balanced_fit_ce']['fit_natural_mean_correction']),
    balanced_effective_bias=configs['balanced_fit_ce']['effective_bias_after_centering_correction'])
   gold_node_records.append(rec)
   if anyerror:
    records.append(rec)
    groups=[byview[v]['group_id'] for v in matched_ids if v in byview]
    known_rules=[rules_by_group[g] for g in groups if g in rules_by_group]
    first_for={method:(bymethod[method]['first_error']==gid) for method in scores}
    # Semantic diagnosis is deliberately left for direct inspection of quoted
    # condition, patient and saved extraction; these tags are score observations.
    supported=[]
    if rec['raw_wrong_with_correct_AP_endpoint']:supported.append('observed_same_evidence_AP_variant_can_change_wrong_to_correct')
    if rec['all_observed_AP_endpoints_wrong'] and not rec['raw_correct']:supported.append('all_available_same_evidence_AP_endpoints_wrong')
    if rec['all_four_wrong'] and abs(values['raw'])>5:supported.append('confident_error_persists_across_all_four_methods')
    if abs(values['raw'])<=3:supported.append('near_original_decision_threshold_abs_score_le_3')
    if rec['raw_correct'] and not rec['balanced_correct']:supported.append('balanced_introduced_gold_node_error')
    if not rec['raw_correct'] and rec['balanced_correct']:supported.append('balanced_corrected_gold_node_error')
    examples.append(dict(**rec,first_error_for=first_for,patient=case['patient'],definitions=case['definitions'],gold_path=case['gold_path'],
      direct_observation_tags=supported,semantic_cause='requires inspection; no automatic causal label',
      existing_rule_extractions=known_rules,AP_details=apinfo))
 firsterrors=[]
 for row in path_results:
  if row['correct']:continue
  gid=row['first_error']
  match=next((r for r in gold_node_records if r['case_id']==row['case_id'] and r['gid']==gid),None)
  if match is not None:firsterrors.append(dict(method=row['method'],**match))
  else:firsterrors.append(dict(method=row['method'],case_id=row['case_id'],source_id=row['source_id'],gid=gid,error=row.get('error')))
 # Fit has no complete path cache. Diagnose available natural inputs only.
 naturalrows=[]
 for v in fitdevviews:
  if v['variant']!='full' or not v['original']:continue
  i=ix[v['view_id']];rr=dict(split=v['split'],view_id=v['view_id'],source_id=v['source_id'],case_id=v['case_id'],group_id=v['group_id'],label=v['label'])
  for method,s in scores.items():rr[method+'_score']=float(s[i]);rr[method+'_correct']=int(s[i]>0)==int(v['label'])
  if not all(rr[method+'_correct'] for method in scores):naturalrows.append(rr)
 summary={}
 for method in scores:
  pr=[r for r in path_results if r['method']==method];fe=[r for r in firsterrors if r['method']==method]
  source=Counter(r['source_id'] for r in fe);question=Counter((r['source_id'],r.get('gid'),r.get('condition','')) for r in fe)
  ne=[r for r in gold_node_records if not r[{'raw':'raw_correct','old_linear':'old_linear_correct','balanced_fit_ce':'balanced_correct','strict_fit_ce':'strict_correct'}[method]]]
  summary[method]=dict(paths=len(pr),correct_paths=sum(r['correct'] for r in pr),error_paths=len(fe),
   unique_first_error_source_nodes=len(question),first_error_sources=len(source),
   source_counts=dict(source),first_error_margin_bins=dict(Counter(r.get('threshold_bin','unavailable') for r in fe)),
   first_error_gold_labels=dict(Counter(str(r.get('gold_label')) for r in fe)),
   gold_node_error_instances=len(ne),unique_gold_error_source_nodes=len({(r['source_id'],r['gid']) for r in ne}),
   first_error_repeated_nodes=[dict(source_id=k[0],gid=k[1],condition=k[2],paths=n) for k,n in question.most_common()],
   raw_wrong_with_correct_AP_endpoint_first_errors=sum(bool(r.get('raw_wrong_with_correct_AP_endpoint')) for r in fe),
   first_error_AP_covered=sum(r.get('observed_AP_pairs',0)>0 for r in fe))
 transitions={}
 base={r['case_id']:r for r in path_results if r['method']=='raw'}
 for method in scores:
  if method=='raw':continue
  rr={r['case_id']:r for r in path_results if r['method']==method}
  transitions[method]=dict(corrected_cases=[cid for cid in base if not base[cid]['correct'] and rr[cid]['correct']],
   new_error_cases=[cid for cid in base if base[cid]['correct'] and not rr[cid]['correct']],
   same_error_cases=[cid for cid in base if not base[cid]['correct'] and not rr[cid]['correct']])
 snapshot=dict(protocol='fit/dev diagnosis only; no old held-out test tuning or test metrics read',
  methods=configs,fit_path_cache=dict(available=False,cases_expected=1712,node_records=0,note='existing fit path index files are empty; no fit path accuracy inferred'),
  dev_path_cache=dict(cases=420,node_records=len(devpaths),all_gold_nodes_scored=True),
  summary=summary,case_transitions_vs_raw=transitions,
  observations=dict(all_gold_node_instances=len(gold_node_records),gold_node_error_union=len(records),
   all_four_wrong_gold_instances=sum(r['all_four_wrong'] for r in records),
   all_four_wrong_unique_source_nodes=len({(r['source_id'],r['gid']) for r in records if r['all_four_wrong']})),
  semantic_labeling_note='Tags report only observed scores, exact-input A/P experiments, and quoted existing data. They do not claim an internal reading failure or validated clinical cause.',
  seconds=time.time()-start)
 dump(OUT/'diagnosis.json',snapshot);dump(OUT/'error_examples.json',examples)
 savecsv(OUT/'first_errors.csv',firsterrors);savecsv(OUT/'gold_node_errors.csv',records);savecsv(OUT/'natural_node_errors.csv',naturalrows)
 with (OUT/'path_results.jsonl').open('w',encoding='utf-8') as f:
  for row in path_results:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 dump(OUT/'status.json',dict(state='complete',seconds=time.time()-start))
 print(json.dumps(dict(summary={k:{kk:vv for kk,vv in v.items() if kk not in ('first_error_repeated_nodes','source_counts')} for k,v in summary.items()},transitions={k:{j:len(vv) for j,vv in v.items()} for k,v in transitions.items()},observations=snapshot['observations']),ensure_ascii=True))
if __name__=='__main__':main()

