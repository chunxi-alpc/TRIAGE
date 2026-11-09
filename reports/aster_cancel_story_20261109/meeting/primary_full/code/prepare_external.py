
"""Adapt existing labeled NLI4CT records for frozen zero-shot binary evaluation.

No examples, labels, counterfactual variants, model weights or parameters are
generated or fitted. All source records, including missing-material records, are
retained and their official split names remain separate.
"""
from pathlib import Path
import json,datetime
from collections import Counter,defaultdict
OUT=Path('/home/gcx/aster_next_20261109/inventory')
NLI=Path('/home/gcx/TRIAGE/data/NLI4CT-2024')
SPECS=(
 ('train','original/train.json','original/train.json','NLI4CT-2024:train'),
 ('dev','original/dev.json','original/dev.json','NLI4CT-2024:dev'),
 ('practice','gold_practice_test.json','practice_test.json','NLI4CT-2024:practice_test'),
 ('test','gold_test.json','test.json','NLI4CT-2024:official_test'),
)
LABELS={'Entailment':1,'Contradiction':0}
def dump(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def load(path):return json.loads(Path(path).read_text())
def dummy_guide():
 condition='Is the statement entailed by the provided clinical-trial material?'
 return dict(groups=[dict(group_id='G0',mutually_exclusive=True,ancestor_conditions=[],rules=[
  dict(rule_id='R0Y',condition=condition,action='L1: Entailment'),
  dict(rule_id='R0N',condition=condition,action='L0: Contradiction')])])
def render_prompt(statement,section,trials):
 header='请根据提供的临床试验记录，判断下面的陈述是否被材料支持。本任务采用二元标签：蕴含（Entailment）输出“是”，矛盾（Contradiction）输出“否”。只依据所给材料，不补充外部知识。'
 chunks=[header]
 for trial in trials:
  role='主要试验' if trial['role']=='primary' else '次要试验'
  lines=trial['section_text']
  if isinstance(lines,list):body='\n'.join(str(line) for line in lines)
  elif lines is None:body='[试验材料缺失]'
  else:body=str(lines)
  chunks.append(role+'：'+trial['trial_id']+'\n章节：'+section+'\n'+body)
 chunks.append('当前陈述：\n'+statement+'\n\n请判断该陈述。只输出“是”或“否”。\n答案：')
 return '\n\n'.join(chunks)
def prepare():
 OUT.mkdir(parents=True,exist_ok=True)
 ctroot=NLI/'original/CT json'
 ctfiles={p.stem:p for p in ctroot.glob('*.json')}
 ctcache={}
 allcases=[];counts={};missing_rows=[];native_ids={};references={};families={}
 input_differences={}
 for split,goldname,inputname,evaluation_group in SPECS:
  gold=load(NLI/goldname);public=load(NLI/inputname)
  native_ids[split]=set(gold);references[split]=set();families[split]=set()
  splitcases=[];changed=[]
  for identifier,record in gold.items():
   reference=public.get(identifier)
   if reference is not None:
    for field in ('Type','Section_id','Primary_id','Secondary_id','Statement'):
     if reference.get(field)!=record.get(field):changed.append(dict(record_id=identifier,field=field))
   section=record.get('Section_id','')
   trials=[];missing=[]
   for field,role in [('Primary_id','primary'),('Secondary_id','secondary')]:
    trialid=record.get(field)
    if not trialid:continue
    references[split].add(trialid)
    if trialid not in ctcache:
     ctcache[trialid]=load(ctfiles[trialid]) if trialid in ctfiles else None
    trial=ctcache[trialid]
    lines=trial.get(section) if trial is not None else None
    if trial is None:missing.append(dict(kind='clinical_trial_file',trial_id=trialid))
    elif section not in trial:missing.append(dict(kind='clinical_trial_section',trial_id=trialid,section=section))
    trials.append(dict(role=role,trial_id=trialid,section_text=lines,
     source_file=str(ctfiles[trialid]) if trialid in ctfiles else None))
   label=LABELS.get(record.get('Label'))
   if label is None:missing.append(dict(kind='label',native_label=record.get('Label')))
   causal=record.get('Causal_type')
   base_id=causal[1] if isinstance(causal,list) and len(causal)>1 else identifier
   families[split].add(base_id)
   prompt_text=render_prompt(record.get('Statement',''),section,trials)
   source_ids=[record[k] for k in ('Primary_id','Secondary_id') if record.get(k)]
   source_key='|'.join(sorted(source_ids))
   case_id='NLI4CT-2024:'+split+':'+identifier
   case=dict(case_id=case_id,id=case_id,view_id=case_id,
    dataset='NLI4CT-2024',dataset_split=split,split=split,evaluation_group=evaluation_group,
    task_type='binary',source_id='NLI4CT-2024:trials:'+source_key,source_key=source_key,
    native_id=identifier,family=base_id,base_id=base_id,
    family_cluster_id='NLI4CT-2024:'+split+':'+base_id,
    primary_id=record.get('Primary_id'),secondary_id=record.get('Secondary_id'),
    clinical_trial_ids=source_ids,section_id=section,native_type=record.get('Type'),
    native_label=record.get('Label'),label=label,
    native_intervention=record.get('Intervention'),native_causal_type=causal,
    statement=record.get('Statement',''),prompt_text=prompt_text,original_prompt=prompt_text,
    guideline=dummy_guide(),definitions=[],gold_path=[dict(gid='G0',label=label)],
    gold_leaf='L1' if label==1 else 'L0' if label==0 else None,
    positive_leaf='L1',negative_leaf='L0',
    status='ready' if not missing else 'missing_input',missing_input=missing,
    source_gold_file=str(NLI/goldname),source_input_file=str(NLI/inputname),
    source_trial_files=[t['source_file'] for t in trials],
    transfer_protocol='Frozen MEGA-trained Qwen readouts applied directly; no NLI training or calibration.')
   if missing:missing_rows.append(dict(case_id=case_id,split=split,missing_input=missing))
   splitcases.append(case);allcases.append(case)
  input_differences[split]=dict(records_missing_from_public=len(set(gold)-set(public)),input_field_differences=len(changed))
  counts[split]=dict(records=len(splitcases),labeled=sum(c['label'] is not None for c in splitcases),
   ready=sum(c['status']=='ready' for c in splitcases),missing_input=sum(c['status']!='ready' for c in splitcases),
   native_labels=dict(Counter(c['native_label'] for c in splitcases)),
   native_types=dict(Counter(c['native_type'] for c in splitcases)),
   native_interventions=dict(Counter(c['native_intervention'] or 'original' for c in splitcases)),
   native_base_families=len(families[split]),unique_trial_ids=len(references[split]),
   prompt_characters=dict(min=min(len(c['prompt_text']) for c in splitcases),max=max(len(c['prompt_text']) for c in splitcases),
    mean=sum(len(c['prompt_text']) for c in splitcases)/len(splitcases)),
   manifest=str(OUT/('nli4ct_'+split+'_cases.jsonl')))
  with (OUT/('nli4ct_'+split+'_cases.jsonl')).open('w',encoding='utf-8') as f:
   for case in splitcases:f.write(json.dumps(case,ensure_ascii=False)+'\n')
 priority={'test':0,'practice':1,'dev':2,'train':3}
 allcases.sort(key=lambda case:priority[case['dataset_split']])
 tmp=OUT/'nli4ct_all_cases.jsonl.tmp'
 with tmp.open('w',encoding='utf-8') as f:
  for case in allcases:f.write(json.dumps(case,ensure_ascii=False)+'\n')
 tmp.replace(OUT/'nli4ct_all_cases.jsonl')
 # Index only the released native variant/base relations. There are no extra
 # prompts or annotations, and these pairs need no additional model forward.
 by_native={(c['dataset_split'],c['native_id']):c for c in allcases}
 native_pairs=[];missing_bases=[]
 for case in allcases:
  if not case['native_causal_type']:continue
  base=by_native.get((case['dataset_split'],case['base_id']))
  if base is None:
   missing_bases.append(dict(case_id=case['case_id'],base_id=case['base_id']))
   continue
  same=base['label']==case['label']
  native_pairs.append(dict(pair_id=case['case_id']+':base='+case['base_id'],
   dataset='NLI4CT-2024',dataset_split=case['dataset_split'],evaluation_group=case['evaluation_group'],
   family=case['base_id'],base_id=case['base_id'],left=base['case_id'],right=case['case_id'],
   left_native_id=base['native_id'],right_native_id=case['native_id'],
   left_label=base['label'],right_label=case['label'],
   relation='preserving' if same else 'changing',
   intervention=case['native_intervention'],published_causal_type=case['native_causal_type'][0],
   relation_source='actual released base/variant labels; never inferred from intervention name',
   trial_section_ids_unchanged=(base['primary_id'],base['secondary_id'],base['section_id'])==(case['primary_id'],case['secondary_id'],case['section_id'])))
 with (OUT/'nli4ct_native_pairs.jsonl').open('w',encoding='utf-8') as f:
  for pair in native_pairs:f.write(json.dumps(pair,ensure_ascii=False)+'\n')
 pair_summary={}
 for split in ['test','practice','dev','train']:
  rr=[pair for pair in native_pairs if pair['dataset_split']==split]
  pair_summary[split]=dict(pairs=len(rr),relations=dict(Counter(pair['relation'] for pair in rr)),
   by_intervention={name:dict(Counter(pair['relation'] for pair in rr if pair['intervention']==name)) for name in sorted({pair['intervention'] for pair in rr})})
 dump(OUT/'nli4ct_native_pair_inventory.json',dict(pairs=len(native_pairs),splits=pair_summary,missing_bases=missing_bases,
  no_new_forward_required=True,no_new_annotations=True,index_file=str(OUT/'nli4ct_native_pairs.jsonl'),
  analysis_scheme='Join each frozen-method case score by left/right case_id, then summarize same-label difference/variance for preserving pairs and signed response retention/correction for changing pairs, separately by split and intervention. This is native NLI variation, not newly named A/P/C/F supervision.',
  caution='Contradiction and Numerical_contradiction are native intervention names; some retain the original truth label, so label-changing responses use actual released label differences.'))
 overlaps={}
 for a in counts:
  for b in counts:
   if a<b:overlaps[a+'__'+b]=dict(native_record_ids=len(native_ids[a]&native_ids[b]),clinical_trial_ids=len(references[a]&references[b]))
 inventory=dict(dataset='NLI4CT-2024',prepared_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  records=len(allcases),labeled_records=sum(c['label'] is not None for c in allcases),
  unique_native_record_ids=len(set().union(*native_ids.values())),
  case_ids_unique=len({c['case_id'] for c in allcases})==len(allcases),
  official_splits_preserved=True,execution_order=['test','practice','dev','train'],counts=counts,split_overlap=overlaps,native_pair_index=str(OUT/'nli4ct_native_pairs.jsonl'),native_pair_inventory=str(OUT/'nli4ct_native_pair_inventory.json'),
  clinical_trial_files_available=len(ctfiles),missing_input_records=missing_rows,
  labeled_vs_public_inputs=input_differences,
  manifest=str(OUT/'nli4ct_all_cases.jsonl'),
  evaluator_compatibility=dict(task='standard binary clinical-trial textual entailment',
   labels='Entailment=1/yes; Contradiction=0/no',prompt_field='prompt_text',node='G0',positive_leaf='L1',negative_leaf='L0',
   materials='Every sentence of the dataset-selected Section_id from the primary and optional secondary trial, with original ordering and wording. No gold evidence selection.',
   native_annotation_use='Label is evaluation metadata only. Intervention/Causal_type/base_id are copied as metadata and are absent from model prompts.'),
  transfer_protocol='Zero-shot transfer of frozen MEGA-trained raw/linear/balanced readouts; NLI train/dev are evaluation groups and are not used for fitting or parameter selection.',
  no_new_data_or_annotations=True,no_training=True,no_test_selection=True,
  count_note='9542 is the number of supplied split records, not independent examples. The200 original dev IDs also appear in the practice set. Derived variants preserve native family/base_id metadata.',
  syllogism=dict(status='not_located_in_provided_resources',real_data_records=0,
   existing_simulation_directory='/home/gcx/aster_cancel_story_20261109/simulated_study',
   action='Keep existing simulations labeled as simulated; do not generate or substitute new syllogisms.'),
  other_existing_resources=dict(
   MedGUIDE=dict(path='/home/gcx/TRIAGE/data/MedGUIDE/data/train-00000-of-00001.parquet',records=7747,
    task='native multiple-choice clinical guideline questions; no directly provided binary tree/gold path schema',
    legacy_subset_cases=800,legacy_input_path='/home/gcx/three_dataset_analysis_20261105/data/medguide_scores_and_inputs.jsonl',
    action='Inventory only; no new task conversion or manifest generated.'),
   MedPAIR=dict(legacy_cases=180,legacy_split=dict(dev=91,test=89),task='multiple-choice clinical questions',action='Inventory only.'),
   MedEinst=dict(legacy_pair_analysis=dict(train_pairs=10244,test_pairs=5372),task='paired clinical diagnostic multiple-choice cases',action='Inventory only.')))
 dump(OUT/'nli4ct_inventory.json',inventory)
 dump(OUT/'external_inventory.json',dict(primary_prepared='NLI4CT-2024',inventory=str(OUT/'nli4ct_inventory.json'),
  syllogism=inventory['syllogism'],other_existing_resources=inventory['other_existing_resources']))
 print(json.dumps(dict(records=len(allcases),unique_native_records=inventory['unique_native_record_ids'],ready=sum(c['status']=='ready' for c in allcases),missing=len(missing_rows),counts={k:v['records'] for k,v in counts.items()})))
if __name__=='__main__':prepare()

