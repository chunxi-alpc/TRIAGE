"""Aggregate only clinical metrics; no patient text or patient identifiers."""
import argparse,json,random
from pathlib import Path
from collections import defaultdict,Counter
from .data import read,write
from .engine import METHODS
def ci(grouped):
 keys=sorted(grouped);rng=random.Random(20261030);vals=[]
 for _ in range(1000):
  chosen=[rng.choice(keys) for _ in keys];rows=[v for k in chosen for v in grouped[k]];vals.append(sum(rows)/len(rows))
 vals.sort();return [vals[24],vals[974]]
def load_units(run,phase,require_complete=True):
 rows=[]
 dynamic=phase=='official_test' and (run/'official_test_units').exists()
 manifest=run/'official_test_launch.json'
 workers=json.loads(manifest.read_text())['logical_workers'] if dynamic and manifest.exists() else [0,1]
 for shard in workers:
  root=run/(phase+str(shard));status=json.loads((root/'status.json').read_text())
  if require_complete and status['status']!='complete':raise ValueError('evaluation still running')
  source=run/'official_test_units' if phase=='official_test' and (run/'official_test_units').exists() else root/'units'
  if dynamic and shard!=workers[0]:continue
  for p in source.glob('*/*.json'):
   if p.name!='classification.json':rows.append(json.loads(p.read_text()))
 return rows
def summarize(run,phase):
 rows=load_units(run,phase)
 expected=len(read(run/'cases'/(phase+'.jsonl')))*len(METHODS)
 if len(rows)!=expected or len({(r['case_id'],r['method']) for r in rows})!=expected:raise ValueError('missing or duplicate evaluation units')
 out=run/('summary_'+phase);out.mkdir(exist_ok=True);index={(r['case_id'],r['method']):r for r in rows};summary=[]
 for method in METHODS:
  group=[r for r in rows if r['method']==method];metric=defaultdict(list);delta=defaultdict(list);macro=defaultdict(list);f1=[];exact=[];corrected=regressed=0;elig=actual=0;tokens=0;seconds=0.;invalid=0;limits=0
  for r in group:
   c=r.get('score',{});base=index[r['case_id'],'baseline'];ok=bool(c.get('final_leaf_correct',False));b=bool(base.get('score',{}).get('final_leaf_correct',False))
   if method!='native_official':
    metric[r['source_id']].append(int(ok));delta[r['source_id']].append(int(ok)-int(b));macro[r['source_id']].append(int(ok))
    f1.append(c.get('path_f1',0.));exact.append(c.get('exact_path',False));corrected+=ok and not b;regressed+=b and not ok
   result=r.get('result',{});elig+=any(result.get('plan',{}).get('gains',{}).values());actual+=result.get('edit',{}).get('changed_key_vectors',0)>0
   tokens+=len(result.get('token_ids',[]));seconds+=result.get('seconds',0.);invalid+=c.get('invalid_generation',False);limits+=result.get('output_limit',False)
  n=len(group);record={'method':method,'denominator':n,'guideline_clusters':len({r['source_id'] for r in group}),'technical_failures':sum(r['status']=='failed' for r in group),
   'invalid_generations':invalid,'output_limits':limits,'scoring_failures':sum(r.get('score',{}).get('scoring_failure',False) for r in group),
   'routing_edit_coverage':elig/n,'actual_key_change_coverage':actual/n,'generation_tokens':tokens,'generation_seconds_sum':seconds,
   'primary_structural_metric_applicable':method!='native_official','semantic_Qwen_score':'separate sidecars, not GPT'}
  if method!='native_official':
   record.update(leaf_correct=sum(v for vs in metric.values() for v in vs),leaf_accuracy=sum(v for vs in metric.values() for v in vs)/n,
    macro_guideline_accuracy=sum(sum(v)/len(v) for v in macro.values())/len(macro),leaf_accuracy_ci=ci(metric),delta_accuracy=sum(v for vs in delta.values() for v in vs)/n,delta_ci=ci(delta),
    path_rule_f1=sum(f1)/n,exact_path_accuracy=sum(exact)/n,corrected=corrected,regressed=regressed)
  summary.append(record)
 write(out/'comparisons.json',summary)
 semantic=[]
 for method in METHODS:
  scores=[]
  source_roots=[run/'official_test_units'] if phase=='official_test' and (run/'official_test_units').exists() else [run/(phase+str(i))/'units' for i in [0,1]]
  for root in source_roots:
   for p in root.glob('*/official_scores/'+method+'.json'):scores.append(json.loads(p.read_text()))
  valid=[r for r in scores if r.get('status')=='complete']
  semantic.append({'method':method,'registered_denominator':expected//len(METHODS),'scored_records':len(scores),'scoring_successes':len(valid),'scoring_failures':len(scores)-len(valid),
   'pending_records':expected//len(METHODS)-len(scores),'Qwen_judged_official_accuracy_on_valid_scores':sum(r['Qwen_judged_official_accuracy'] for r in valid)/len(valid) if valid else None,
   'Qwen_judged_official_accuracy_failures_as_zero':sum(r['Qwen_judged_official_accuracy'] for r in valid)/(expected//len(METHODS)),
   'Qwen_judged_official_PCC_cost_on_valid_scores':sum(r['Qwen_judged_official_PCC'] for r in valid)/len(valid) if valid else None,'not_GPT_main_table':True})
 write(out/'official_Qwen_scores.json',semantic)
 write(out/'status.json',{'status':'complete','phase':phase,'units':len(rows),'expected_units':len(read(run/'cases'/(phase+'.jsonl')))*len(METHODS),
  'cases':len({r['case_id'] for r in rows}),'technical_failures':sum(r['status']=='failed' for r in rows),'raw_patient_data_published':False})
 return summary
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--phase',choices=['confirm','official_test'],required=True);a=p.parse_args();rows=summarize(Path(a.run),a.phase)
 print(json.dumps([{k:r.get(k) for k in ['method','denominator','leaf_correct','leaf_accuracy','path_rule_f1','technical_failures','corrected','regressed']} for r in rows]))
if __name__=='__main__':main()

