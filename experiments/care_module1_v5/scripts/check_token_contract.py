from datetime import datetime, timezone
from pathlib import Path
import hashlib, importlib.util, itertools, json, sys, traceback
ROOT=Path('/home/gcx/TRIAGE-v5-pr4'); EXP=ROOT/'experiments/care_module1_v5'
OUT=Path('/tmp/care_v5_pipeline_token_compatibility.json')
sys.path.insert(0,str(ROOT))
result={'status':'running','checked_utc':datetime.now(timezone.utc).isoformat(),
 'purpose':'synthetic interface check only; fake TrialMatchAI and NLI outputs; no model accuracy meaning',
 'tokenizer':'/home/gcx/models/care_module1_v5/phi-4','device':'cpu','model_inference':False,
 'qwen_calls':0,'module2_search_run':False,'module3_attention_run':False,'frozen_code_modified':False,
 'report_location_note':'Temporary output required by automatic approval review; current server worktree is untouched.',
 'datasets':[],'failures':[]}
def read(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
try:
 from transformers import AutoTokenizer
 from src.module1.care_v5 import CAREModule1V5,eval_logic,map_trialmatch_label
 spec=importlib.util.spec_from_file_location('existing_pipeline_tokens',ROOT/'src/pipeline/tokens.py')
 tokens=importlib.util.module_from_spec(spec);spec.loader.exec_module(tokens)
 tokenizer=AutoTokenizer.from_pretrained(result['tokenizer'],local_files_only=True,use_fast=True)
 assert tokenizer.is_fast
 result['tokenizer_class']=type(tokenizer).__name__
 result['implementation_hashes']={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'src/module1/care_v5.py',ROOT/'src/module1/compat_v5.py',ROOT/'src/pipeline/tokens.py']}
 class FakeTrialMatch:
  def __init__(self,states):self.states=states
  def judge(self,patient,condition,criterion_kind='inclusion'):
   state=self.states[condition['text']]
   label=('Violated' if state=='T' else 'Not Violated') if criterion_kind=='exclusion' else ('Met' if state=='T' else 'Not Met')
   return {'raw_output':'SYNTHETIC INTERFACE FIXTURE','raw_label':label,'state':map_trialmatch_label(label,criterion_kind),'criterion_kind':criterion_kind}
 class FakeNLI:
  id2label={0:'entailment',1:'neutral',2:'contradiction'}
  def __init__(self,states):self.states=states
  def predict(self,pairs,batch_size=32):
   rows=[]
   for premise,hypothesis in pairs:
    atom=hypothesis.removeprefix('For this patient, the following condition is true: ')
    relation='entailment' if self.states[atom]=='T' else 'contradiction'
    rows.append({'relation':relation,'class_id':0 if relation=='entailment' else 2,'scores':{k:float(k==relation) for k in self.id2label.values()}})
   return rows
 def numeric_sentence(c):
  field=c.get('field',c['numeric'].get('field'));v=c['numeric']['value'];n=str(int(v)) if float(v).is_integer() else str(v)
  formats={'pt':'Original pathological T stage is pT{n}.','pn':'Original pathological N stage is pN{n}.',
   'ct':'Preoperative clinical T stage is cT{n}.','cn':'Preoperative clinical N stage is cN{n}.',
   'ypn':'Post-neoadjuvant pathological N stage is ypN{n}.','age':'Patient age is {n} years.',
   'relapse_months':'Relapse interval is {n} months.','tumor_cm':'Invasive tumor diameter is {n} cm.',
   'margin_mm':'Closest resection margin is {n} mm.','tmb':'Tumor mutational burden is {n} mutations/Mb.',
   'sln_count':'Sentinel lymph node assessment: {n} positive sentinel lymph nodes.',
   'ax_nodes':'Axillary lymph node assessment: {n} positive axillary lymph nodes.'}
  return formats[field].format(n=n)
 sources={r['rule_id']:r['text'] for r in read(EXP/'exploratory/data/private/source_trees.jsonl')}
 for name,path,count in [('v4_regression',EXP/'data/rules_v5.jsonl',16),('new_synthetic',EXP/'exploratory/data/private/rules.jsonl',4)]:
  rules=read(path);assert len(rules)==count
  dataset={'name':name,'rule_count':len(rules),'rules_sha256':sha(path),'cases':[]}
  for rule in rules:
   atoms=rule['conditions']+rule.get('exceptions',[]);ids=[c['condition_id'] for c in atoms]
   assignment=next((dict(zip(ids,vals)) for vals in itertools.product(('T','F'),repeat=len(ids)) if eval_logic(rule['logic'],dict(zip(ids,vals)))=='T' and eval_logic(rule['exception_logic'],dict(zip(ids,vals)))=='F'),None)
   assert assignment is not None
   states={c['text']:assignment[c['condition_id']] for c in atoms};sentences=[];measured=set()
   for c in atoms:
    if c.get('numeric'):
     field=c.get('field',c['numeric'].get('field'))
     if field in measured:continue
     measured.add(field);sentences.append(numeric_sentence(c))
    else:sentences.append(('Synthetic finding: ' if states[c['text']]=='T' else 'Synthetic explicit negation: it is false that ')+c['text'])
   patient='\n'.join(sentences)
   graph=CAREModule1V5(FakeTrialMatch(states),FakeNLI(states)).run([rule],sentences,'Synthetic interface mapping check')
   prompt,encoded,mapping=tokens.build_prompt_and_mapping(tokenizer,patient,'Synthetic interface mapping check',graph)
   actual=tokenizer(prompt,add_special_tokens=False,return_offsets_mapping=True);offsets=actual['offset_mapping']
   assert actual['input_ids']==encoded['input_ids'][0].tolist()
   assert encoded['input_ids'].device.type=='cpu'
   for part in mapping['character_spans']:assert part['tokens']==tokens.char_tokens(offsets,part['start'],part['end'])
   patient_base=next(p['start'] for p in mapping['character_spans'] if p['field']==['patient'])
   relation_map={(r['rule_id'],r['condition_id']):r for r in mapping['relations']};evidence_spans=0
   for relation in graph['relation_weights']:
    expected=set()
    for span in relation['evidence']:
     assert patient[span['start']:span['end']]==span['text']
     expected.update(tokens.char_tokens(offsets,patient_base+span['start'],patient_base+span['end']));evidence_spans+=1
    assert relation_map[relation['rule_id'],relation['condition_id']]['evidence_tokens']==sorted(expected)
   for c in atoms:
    span=c['source_span'];assert c['text'] in rule['text'][span['start']:span['end']]
    assert mapping['rules'][rule['rule_id']]['condition_tokens'][c['condition_id']]
   state=graph['rules'][0]['applicability'];action_tokens=mapping['rules'][rule['rule_id']]['action_tokens'];original_action_checked=False
   if name=='new_synthetic':
    assert state=='T', 'Synthetic satisfying assignment must expose new-rule action'
    assert rule['action'] and action_tokens
    part=next(p for p in mapping['character_spans'] if p['field']==['action',rule['rule_id']]);assert prompt[part['start']:part['end']]==rule['action']
    action=rule['action_span'];assert rule['text'][action['start']:action['end']]==rule['action']
    original=rule['original_action_span'];raw=sources[rule['rule_id']][original['start']:original['end']];assert json.loads('"'+raw+'"')==rule['action'];original_action_checked=True
   else:assert not rule.get('action') and not action_tokens
   dataset['cases'].append({'rule_id':rule['rule_id'],'synthetic_fixture':True,'applicability':state,
    'condition_states':[c['status'] for c in graph['rules'][0]['conditions']+graph['rules'][0]['exceptions']],
    'schema_version':graph['schema_version'],'implementation_version':graph['implementation_version'],
    'prompt_tokens':len(actual['input_ids']),'condition_count':len(atoms),'evidence_spans_checked':evidence_spans,
    'condition_offsets_exact':True,'patient_evidence_substrings_exact':True,'token_mapping_matches_real_offsets':True,
    'action_present':bool(rule.get('action')),'action_token_count':len(action_tokens),'original_action_substring_checked':original_action_checked})
  result['datasets'].append(dataset)
 result['qwen_module_imports']=[name for name in ('src.module1.modeler','src.module1.backend') if name in sys.modules];assert not result['qwen_module_imports']
 import torch
 result['cuda_initialized']=torch.cuda.is_initialized();assert not result['cuda_initialized']
 result['status']='passed';result['rules_checked']=sum(d['rule_count'] for d in result['datasets'])
 result['new_rule_actions_mapped']=sum(c['action_token_count']>0 for d in result['datasets'] if d['name']=='new_synthetic' for c in d['cases'])
 result['limitations']=['Synthetic fixture outputs are not real predictions and must not enter experiment evaluation.',
  'No module-two head search, module-three attention intervention, or final-generation model was run.',
  'V4 rules have no action text; only the four new rules exercise nonempty action mapping.']
except Exception as exc:
 result['status']='failed';result['failures'].append({'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc(limit=8)})
if OUT.exists():raise FileExistsError('Refusing to overwrite previous interface check')
OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':result['status'],'rules_checked':result.get('rules_checked'),'new_rule_actions_mapped':result.get('new_rule_actions_mapped'),'output':str(OUT),'failures':[{k:v for k,v in x.items() if k!='traceback'} for x in result['failures']]},ensure_ascii=False))
if result['status']!='passed':raise SystemExit(2)
