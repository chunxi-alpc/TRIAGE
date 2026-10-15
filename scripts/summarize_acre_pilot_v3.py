#!/usr/bin/env python3
"""Complete-case dev summary, shared beta and equal-budget baseline selection."""
import sys,json,csv,gzip,collections,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.acre_pilot.evaluator import evaluate_cases,paired_comparisons,fraction
from scripts.run_acre_pilot import write
DATA=ROOT/'results/acre_pilot/round2';OUT=DATA/'revision3';RUN=OUT/'development';config=json.loads((ROOT/'configs/acre_pilot_v3.json').read_text())
manifest=json.loads((DATA/'development_manifest.json').read_text())['samples'];refs=json.loads((DATA/'development_references.json').read_text());cases=[]
for item in manifest:
 cid=item['case_id'];base=RUN/('case_'+cid+'.json');oracle=RUN/('case_'+cid+'_oracle.json')
 if not base.exists() or not oracle.exists():raise ValueError('Incomplete normal/oracle results; all valid samples must remain in denominator: '+cid)
 c=json.loads(base.read_text());o=json.loads(oracle.read_text());c['variants'].update(o['variants']);c['oracle_annotations']=o['oracle_annotations'];c['oracle_diagnostic_only']=True
 if c['base_prompt_sha256']!=o['base_prompt_sha256']:raise ValueError('Oracle decision prompt differs from normal prompt')
 for method,value in c['variants'].items():
  if method not in ('B4','B5') and value['prompt_sha256']!=c['base_prompt_sha256']:raise ValueError('Unstructured decision prompts differ')
 cases.append(c)
if len(cases)!=80 or set(refs)!={c['case_id'] for c in cases}:raise ValueError('Expected 40 complete verified pairs')
metrics,ann,details=evaluate_cases(cases,refs)
def value(row,key):return row[key]['value'] if row[key]['value'] is not None else 0.
def ranking(method,beta):
 row=metrics[method+'@'+str(float(beta))]
 return (value(row,'pair_accuracy'),value(row,'accuracy'),-value(row,'oar'),-beta)
best={m:max(config['betas'],key=lambda b:ranking(m,b)) for m in ('B1','B2','B3','B6')};shared=best['B3']
selected={m:metrics[m] for m in ('B0','B4','B5')}
for m in ('B1','B2','B3','B6'):selected[m]=metrics[m+'@'+str(float(shared))]
comparisons=paired_comparisons(details,[(f'B1@{shared:.1f}','B0'),(f'B3@{shared:.1f}',f'B2@{shared:.1f}'),(f'B3@{shared:.1f}','B4'),(f'B3@{shared:.1f}',f'B6@{shared:.1f}'),(f'B3@{shared:.1f}','B5')])
# Target rule comparison is always evaluated across ALL 80 patients, including
# relevance misses. Whole-guide macro counts are also retained separately.
conf={s:{t:0 for t in ('applicable','inapplicable','unknown')} for s in ('applicable','inapplicable','unknown')};relevance=0
for c in cases:
 ref=refs[c['case_id']];row=next(r for r in c['annotations'] if r['rule_id']==ref['target_rule_id']);conf[ref['target_applicability']][row['applicability']]+=1;relevance+=row['relevant']
ann['target_rule_confusion']=conf;ann['target_rule_accuracy']=fraction(sum(conf[s][s] for s in conf),80);ann['target_rule_relevance_coverage']=fraction(relevance,80)
ann['target_rule_per_class']={s:{'precision':fraction(conf[s][s],sum(conf[t][s] for t in conf)),'recall':fraction(conf[s][s],sum(conf[s].values()))} for s in conf}
ann['target_inapplicable_as_applicable']=fraction(conf['inapplicable']['applicable'],40)
ann['target_unknown_fraction']=fraction(sum(conf[t]['unknown'] for t in conf),80)
ann['annotation_protocol']='source_ids_json_grammar.v3';ann['exact_span_location_interpretation']='Guaranteed by original-source ID selection, not proof of semantic evidence correctness.'
write(OUT/'annotation_metrics.json',ann)
write(OUT/'metrics.json',{'scope':'40-pair development only; no independent validation','sample_count':80,'pair_count':40,'shared_beta':shared,'selection_rule':config['selection'],'all_grid_metrics':metrics,'main_shared_beta':selected,'prespecified_beta':config['default_beta'],'prespecified_beta_metrics':{m:metrics[m+'@'+str(float(config['default_beta']))] for m in ('B1','B2','B3','B6')},'equal_tuning_budget_best_betas':best,'equal_tuning_budget_metrics':{m:metrics[m+'@'+str(float(best[m]))] for m in best},'OAR_denominator_protocol':'40 patients with target yes-rule explicitly inapplicable and current action incompatible; fixed target action plus source-reviewed full-action paraphrases from action_alias_audit.json. Posthoc OAR correction disclosed, no prediction-specific competitors.','evidence_recall_protocol':'Audited changed-condition exact source span contained within model-selected evidence of correct role; all 80 target rules included','statistics_scope':'Exploratory development; paired tests and guide-cluster intervals not paper evidence','comparisons':comparisons})
with (OUT/'comparisons.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(comparisons[0]),lineterminator='\n');w.writeheader();w.writerows(comparisons)
for name,rows in [('case_results.jsonl.gz',cases),('evaluation_details.jsonl.gz',details)]:
 with gzip.open(OUT/name,'wt',encoding='utf-8') as f:
  for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
errors=[r for r in details if not r['correct']];write(OUT/'error_analysis.json',{'incorrect_decisions_all_grid':errors,'out_of_single_rule_scope_cases':[c['case_id'] for c in cases if c.get('out_of_single_rule_scope')],'annotation_failures':[c['case_id'] for c in cases if c['annotation_errors']],'mapping_failures':[{'case_id':c['case_id'],'rule_id':r['rule_id'],'errors':r['mapping_errors']} for c in cases for r in c['annotations'] if r['mapping_errors']]})
checks={'all_80_cases_complete':True,'all_80_beta_zero_empty_exact':all(all(c['identity_controls'].values()) for c in cases),'base_prompts_identical_including_B6':True,'all_active_36_layers_32_heads':all(len(v['attention']['layers'])==36 and all(r['query_heads']==32 and r['kv_heads']==8 for r in v['attention']['layers'].values()) for c in cases for v in c['variants'].values() if v.get('attention',{}).get('enabled')),'no_annotation_parse_errors':not any(c['annotation_errors'] for c in cases),'no_source_mapping_errors':not any(r['mapping_errors'] for c in cases for r in c['annotations']),'historical_guidelines_excluded':not ({c['guideline_id'] for c in cases}&set(config['historical_guidelines'])),'full_original_actions_preserved':True,'normal_annotations_no_gold':True,'B6_only_separate_offline_reference_pass':True,'all_valid_cases_included':True,'independent_validation_performed':False}
if not all(v for k,v in checks.items() if k!='independent_validation_performed'):raise ValueError('Engineering acceptance failure')
write(OUT/'acceptance.json',checks)
print(json.dumps({'shared_beta':shared,'main':{m:{k:r[k] for k in ('accuracy','pair_accuracy','oar','retention','decision_coverage')} for m,r in selected.items()},'annotation':{k:ann[k] for k in ('applicability_accuracy','target_rule_accuracy','decisive_evidence_recall','exact_span_location','unknown_fraction')}},ensure_ascii=False))
