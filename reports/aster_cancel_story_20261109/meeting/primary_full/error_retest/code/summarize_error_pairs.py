
"""Describe fixed old-main-method error pairs from already saved score CSVs.

No model inference, fitting, recalibration or parameter selection is performed.
Error pools are selected only by the old16-dimensional method's paired correctness.
"""
from pathlib import Path
import argparse,csv,json,time
from collections import defaultdict,Counter
import numpy as np
BASE=Path('/home/gcx/aster_next_20261109')
OLD=Path('/home/gcx/aster_cancel_story_20261109/meeting/real_completion')
OPT=Path('/home/gcx/aster_optimization_20261109/final/results_calibration_ablation')
ROOT=Path('/home/gcx/aster_cr_repair_v2_run_20261108')
METHODS=['raw','old_linear','optimized_balanced_fit_calibration','optimized_strict_fit_calibration','trace_only_balanced_fit_calibration']
def read_csv(path):
 with Path(path).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def read_rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def dump(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def boolean(value):return str(value).lower() in ('true','1')
def finite(value):
 value=float(value)
 return value if np.isfinite(value) else None
def keyed_scores(path,alias=None):
 alias=alias or {};out={}
 for r in read_csv(path):
  method=alias.get(r['method'],r['method'])
  if method not in METHODS:continue
  out[method,r['pair_id']]=r
 return out
def score_correct(row,meta):
 return int(float(row['left_score'])>0)==int(meta['left_label']) and int(float(row['right_score'])>0)==int(meta['right_label'])
def stats(pairids,metadata,scores,method,group_relation,normalization_rms=None):
 available=[p for p in pairids if (method,p) in scores and ('raw',p) in scores and p in metadata]
 if not available:return dict(n=0,expected=len(pairids),missing=len(pairids),status='no_old_error_pairs' if not pairids else 'missing_scores')
 yl=np.array([metadata[p]['left_label'] for p in available],dtype=int);yr=np.array([metadata[p]['right_label'] for p in available],dtype=int)
 sl=np.array([float(scores[method,p]['left_score']) for p in available]);sr=np.array([float(scores[method,p]['right_score']) for p in available])
 rl=np.array([float(scores['raw',p]['left_score']) for p in available]);rr=np.array([float(scores['raw',p]['right_score']) for p in available])
 raw=rr-rl;dy=sr-sl;dc=dy-raw
 co=((sl>0)==yl)&((sr>0)==yr)
 raw_rms=float(np.sqrt(np.mean(raw**2)));delta_rms=float(np.sqrt(np.mean(dc**2)))
 result=dict(n=len(available),expected=len(pairids),missing=len(pairids)-len(available),paired_correct=int(co.sum()),paired_accuracy=float(co.mean()),
  raw_difference_rms=raw_rms,difference_rms=float(np.sqrt(np.mean(dy**2))),correction_rms=delta_rms,
  difference_rms_ratio=finite(np.sqrt(np.mean(dy**2))/raw_rms) if raw_rms else None,
  correction_rms_over_error_pool_raw_rms=delta_rms/raw_rms if raw_rms else None,
  raw_difference_mean=float(np.mean(raw)),difference_mean=float(np.mean(dy)),
  raw_difference_sign_changed_or_zero=int(np.sum((raw!=0)&(np.sign(raw)*dy<=0))))
 if normalization_rms is not None:result['correction_rms_over_full_raw_AP_rms']=delta_rms/normalization_rms
 if group_relation=='changing':
  direction=yr-yl;original=direction*raw;changed=direction*dy
  result.update(raw_signed_response_mean=float(np.mean(original)),signed_response_mean=float(np.mean(changed)),
   signed_response_retention=finite(np.mean(changed)/np.mean(original)) if np.mean(original) else None,
   raw_direction_positive=int(np.sum(original>0)),direction_positive=int(np.sum(changed>0)),
   response_direction_degraded=int(np.sum((original>0)&(changed<=0))),
   response_direction_improved=int(np.sum((original<=0)&(changed>0))))
 else:
  result['variance_reduction']=finite(1-np.var(dy)/np.var(raw)) if np.var(raw) else None
 return result
def population_changes(pairids,metadata,scores,method):
 valid=[p for p in pairids if ('old_linear',p) in scores and (method,p) in scores and p in metadata]
 old=np.array([score_correct(scores['old_linear',p],metadata[p]) for p in valid])
 new=np.array([score_correct(scores[method,p],metadata[p]) for p in valid])
 return dict(full_pairs=len(valid),old_error_pairs=int((~old).sum()),originally_correct_pairs=int(old.sum()),
  fixed_old_errors=int(np.sum(~old&new)),still_wrong=int(np.sum(~old&~new)),
  new_errors_on_originally_correct=int(np.sum(old&~new)),net_correct_change=int(new.sum()-old.sum()))
def main(out_dir):
 started=time.time();out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
 old_scores=keyed_scores(OLD/'test_pair_results.csv',{'linear_output':'old_linear'})
 new_scores=keyed_scores(OPT/'test_pair_results.csv',{'linear_output':'old_linear'})
 # Preserve the original score CSV for pool definition and original references.
 mega_scores=dict(new_scores)
 for key,value in old_scores.items():
  if key[0] in ('raw','old_linear'):mega_scores[key]=value
 metadata_all={p['pair_id']:p for p in read_rows(ROOT/'data/pairs.jsonl') if p['split']=='test'}
 mega_ids=sorted(p for p in metadata_all if ('old_linear',p) in old_scores)
 mega_errors={p for p in mega_ids if not boolean(old_scores['old_linear',p]['paired_correct'])}
 # Check that published pair flags agree with endpoints and released labels.
 flag_mismatches=[p for p in mega_ids if score_correct(old_scores['old_linear',p],metadata_all[p])!=boolean(old_scores['old_linear',p]['paired_correct'])]
 if flag_mismatches:raise ValueError('Old MEGA score/label flags disagree')
 raw_ap=[float(mega_scores['raw',p]['right_score'])-float(mega_scores['raw',p]['left_score']) for p in mega_ids if metadata_all[p]['kind'] in ('A','P')]
 denominator=float(np.sqrt(np.mean(np.square(raw_ap))))
 native_file=BASE/'primary_results/nli4ct_all/pair_summary/pair_results.csv'
 nli_scores=keyed_scores(native_file)
 native_metadata={p['pair_id']:p for p in read_rows(BASE/'inventory/nli4ct_native_pairs.jsonl')}
 nli_ids=sorted(p for p in native_metadata if ('old_linear',p) in nli_scores)
 nli_errors={p for p in nli_ids if not boolean(nli_scores['old_linear',p]['paired_correct'])}
 mismatch=[p for p in nli_ids if score_correct(nli_scores['old_linear',p],native_metadata[p])!=boolean(nli_scores['old_linear',p]['paired_correct'])]
 if mismatch:raise ValueError('Old NLI pair flags disagree with released endpoint labels')
 groups={};flat=[];selected_rows=[];coverage={}
 datasets=dict(
  MEGA_previous_controlled=dict(description='Previously used MEGA controlled held-out input pairs; this is not the new7458-case official path set',
   full_pairs=len(mega_ids),old_error_pairs=len(mega_errors),error_counts_by_kind=dict(Counter(metadata_all[p]['kind'] for p in mega_errors)),
   full_raw_AP_rms=denominator,pool_definition_file=str(OLD/'test_pair_results.csv'),
   labels_file=str(ROOT/'data/pairs.jsonl')),
  NLI4CT_2024_native=dict(description='Native released base/variant relations; same-label is not automatically semantic invariance',
   full_pairs=len(nli_ids),old_error_pairs=len(nli_errors),
   error_counts_by_split=dict(Counter(native_metadata[p]['dataset_split'] for p in nli_errors)),
   pool_definition_file=str(native_file),labels_file=str(BASE/'inventory/nli4ct_native_pairs.jsonl')))
 def add_group(key,dataset,split,category,relation,fullids,errorids,meta,scoremap,normalization=None):
  group=dict(dataset=dataset,split=split,category=category,relation=relation,
   full_pair_count=len(fullids),error_pair_count=len(errorids),methods={},full_population_changes={})
  for method in METHODS:
   point=stats(errorids,meta,scoremap,method,relation,normalization)
   if relation=='changing':
    if dataset=='MEGA_previous_controlled':point['mean_evidence_retention']=point.get('signed_response_retention')
    else:point['mean_signed_response_retention']=point.get('signed_response_retention')
   group['methods'][method]=point
   group['full_population_changes'][method]=population_changes(fullids,meta,scoremap,method)
   flat.append(dict(group=key,dataset=dataset,split=split,category=category,relation=relation,method=method,
    full_pair_count=len(fullids),error_pair_count=len(errorids),**point))
  groups[key]=group
 for group,kinds,relation in [('A',('A',),'preserving'),('P',('P',),'preserving'),('AP',('A','P'),'preserving'),
  ('C',('C',),'changing'),('F',('F',),'changing'),('CF',('C','F'),'changing'),('F0',('F0',),'same_label_fact_change')]:
  fullids=[p for p in mega_ids if metadata_all[p]['kind'] in kinds];errorids=[p for p in fullids if p in mega_errors]
  add_group('MEGA_previous_controlled:'+group,'MEGA_previous_controlled','previous_test',group,relation,fullids,errorids,metadata_all,mega_scores,denominator)
 for split in ['test','practice','dev','train']:
  subset=[p for p in nli_ids if native_metadata[p]['dataset_split']==split]
  if not subset:continue
  interventions=['all']+sorted({native_metadata[p]['intervention'] for p in subset})
  for intervention in interventions:
   for relation in ['preserving','changing']:
    fullids=[p for p in subset if native_metadata[p]['relation']==relation and (intervention=='all' or native_metadata[p]['intervention']==intervention)]
    if not fullids:continue
    errors=[p for p in fullids if p in nli_errors]
    add_group('NLI4CT-2024:'+split+':'+intervention+':'+relation,'NLI4CT-2024',split,intervention,relation,fullids,errors,native_metadata,nli_scores)
 for dataset,errorids,meta,scoremap in [('MEGA_previous_controlled',mega_errors,metadata_all,mega_scores),('NLI4CT-2024',nli_errors,native_metadata,nli_scores)]:
  missing={}
  for method in METHODS:missing[method]=sum((method,p) not in scoremap for p in errorids)
  coverage[dataset]=dict(selected_old_error_pairs=len(errorids),missing_method_scores=missing,endpoint_labels_available=all(p in meta for p in errorids))
  for p in sorted(errorids):
   for method in METHODS:
    if (method,p) not in scoremap:continue
    m=meta[p];r=scoremap[method,p]
    selected_rows.append(dict(dataset=dataset,pair_id=p,method=method,split=m.get('dataset_split','previous_test'),
     category=m.get('kind',m.get('intervention')),relation=m.get('relation'),
     left_label=int(m['left_label']),right_label=int(m['right_label']),
     left_score=float(r['left_score']),right_score=float(r['right_score']),
     paired_correct=score_correct(r,m)))
 result=dict(selection=dict(reference_method='old_linear',MEGA_original_method_name='linear_output',
  rule='paired_correct==False in the saved pre-optimization16-dimensional method',
  optimized_scores_used_for_pool_selection=False,endpoint_label_source='existing pair metadata; no generated labels'),
  methods=METHODS,datasets=datasets,groups=groups,coverage=coverage,
  interpretation=dict(error_conditioned_subset='These fixed error subsets are diagnostic, not a new independent generalization test; old paired correctness is zero by construction.',
   MEGA_scope='The paired MEGA data were previously evaluated. They must not be reported as paired coverage of the new7458 official paths.',
   normalization='MEGA correction_rms_over_full_raw_AP_rms uses the original full2293-pair raw AP RMS; error-pool RMS denominators are also explicit.',
   NLI_scope='Native intervention names are retained, relation comes from actual released labels. Same-label Contradiction variants are not asserted semantically invariant.',
   F0='Same-label fact changes are separate from nuisance A/P changes; their difference is not assumed to require elimination.',
   response='Signed response retention is relative to raw response on the selected error subset, not an absolute truth score.',
   regression='Full-population fixed/still-wrong/new-error counts are included to expose regressions outside the old error pool.'),
  no_new_forward=True,no_fitting_or_configuration_changes=True,bootstrap='Not added; point estimates and actual conditional denominators only.',seconds=time.time()-started)
 dump(out/'pair_metrics.json',result)
 fields=list(dict.fromkeys(k for r in flat for k in r))
 for filename in ['pair_summary.csv','summary.csv']:
  with (out/filename).open('w',newline='',encoding='utf-8') as f:
   writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(flat)
 with (out/'selected_error_pair_scores.csv').open('w',newline='',encoding='utf-8') as f:
  writer=csv.DictWriter(f,fieldnames=list(selected_rows[0]));writer.writeheader();writer.writerows(selected_rows)
 dump(out/'pair_analysis_done.json',dict(state='complete',MEGA_old_error_pairs=len(mega_errors),NLI_old_error_pairs=len(nli_errors),groups=len(groups),seconds=time.time()-started))
 print(json.dumps(dict(MEGA=datasets['MEGA_previous_controlled'],NLI=datasets['NLI4CT_2024_native'],groups=len(groups))))
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=BASE/'error_retest');main(p.parse_args().out)

