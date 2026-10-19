#!/usr/bin/env python3
import sys,json,csv,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.acre_pilot.evaluator import evaluate_cases,paired_comparisons
from scripts.run_acre_pilot import write
p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--references',type=Path,required=True);p.add_argument('--historical',action='store_true');a=p.parse_args()
cases=[json.loads(f.read_text()) for f in sorted(a.directory.glob('case_*.json')) if not f.stem.endswith('_oracle')]
refs=json.loads(a.references.read_text());refs={r['case_id']:r for r in refs['samples']} if 'samples' in refs else refs
if {c['case_id'] for c in cases} != set(refs):raise ValueError('Incomplete case results: cannot silently exclude failed or unfinished samples')
if a.historical:
    manifest=json.loads((ROOT/'results/acre_pilot/history_manifest.json').read_text())
    for item in manifest['samples']:
        # Valid only for this complete exclusive decision tree regression.
        cid=item['case_id'];refs[cid]['rule_states']={r['rule_id']:('applicable' if r['rule_id']==refs[cid]['gold_rule_id'] else 'inapplicable') for r in item['guideline']['rules']}
for c in cases:
    oracle=a.directory/(Path('case_'+c['case_id'].replace(':','_')).name+'_oracle.json')
    if oracle.exists():c['variants'].update(json.loads(oracle.read_text())['variants'])
metrics,ann,details=evaluate_cases(cases,refs)
metadata={'n':len(cases),'historical_regression_only':a.historical,'positive_evidence_allowed':not a.historical,'no_guideline_acf_or_nca_reported':True,'all_valid_cases_included':True}
write(a.directory/'metrics.json',{'metadata':metadata,'methods':metrics});write(a.directory/'annotation_metrics.json',ann)
(a.directory/'case_results.jsonl').write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in cases))
(a.directory/'evaluation_details.jsonl').write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in details))
methods=set(metrics);pairs=[(x,y) for x,y in [('B1@1.0','B0'),('B3@1.0','B2@1.0'),('B3@1.0','B4'),('B3@1.0','B5'),('B3@1.0','B6@1.0')] if x in methods and y in methods]
comparison=paired_comparisons(details,pairs)
write(a.directory/'comparisons.json',comparison)
if comparison:
 with (a.directory/'comparisons.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(comparison[0]),lineterminator='\n');w.writeheader();w.writerows(comparison)
errors=[r for r in details if not r['correct']]
write(a.directory/'error_analysis.json',{'incorrect_decisions':errors,'annotation_failure_cases':[c['case_id'] for c in cases if c['annotation_errors']],'out_of_scope_cases':[c['case_id'] for c in cases if c.get('out_of_single_rule_scope')],'mapping_errors':[{'case_id':c['case_id'],'rule_id':r['rule_id'],'errors':r['mapping_errors']} for c in cases for r in c['annotations'] if r['mapping_errors']]})
print(json.dumps({m:v['accuracy'] for m,v in metrics.items()}))
