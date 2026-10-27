"""Validate repair contracts and freeze hashes without reading model outputs."""
from pathlib import Path
import collections,hashlib,json,re
from run_p1 import render
from repair_data import hide_question_treatments,split_action
D=Path(__file__).resolve().parent
rows=[json.loads(s) for s in (D/'private/cases.jsonl').read_text().splitlines()];audit=json.loads((D/'results/data_audit.json').read_text());groups=collections.defaultdict(dict)
for r in rows:groups[r['id']][r['variant']]=r
source={(r['id'],r['variant']):r for r in map(json.loads,Path(audit['source']).read_text().splitlines())}
contracts=collections.Counter()
for cid,vs in groups.items():
 n=vs['NC'];g=int(n['gold'])-1;other=1-g
 leaves={q['condition'].split(' > ')[-1] for q in n['rules']}
 approved=leaves in [{'Transplant candidate','Not transplant candidate'},{'pN0 or cN0','pN2–3 (≥4 positive axillary nodes)'},{'Yes to all','No to at least one'},{'Negative margins for invasive disease','Positive margins for invasive disease'},{'<6 mo from completion of primary therapy','≥6 mo from completion of primary therapy'}]
 assert approved,(cid,leaves)
 assert n['rules'][0]['condition'].split(' > ')[:-1]==n['rules'][1]['condition'].split(' > ')[:-1]
 contracts['explicitly_distinct_entry_conditions']+=1
 full=source[cid,'NC']['patient'];blind,spans=hide_question_treatments(full)
 assert n['patient']==full and n['blind_patient']==blind
 # Reconstruct the blind string from exactly the declared treatment spans.
 expected='';last=0
 for span in spans:
  assert full[span['start']:span['end']]==span['text'];expected+=full[last:span['start']]+'[treatment]';last=span['end']
 expected+=full[last:];assert expected==blind
 contracts['all_nonredacted_source_characters_preserved']+=1
 assert len({render(vs[v],'blind') for v in ['NC','IC','AC','SW']})==1
 for v in ['NC','IC','AC','SW']:
  r=vs[v];assert r['gold']==n['gold'] and r['patient']==full and r['blind_patient']==blind
  assert r['options']==n['options']
  assert [(q['condition'],q['execution_qualifications']) for q in r['rules']]==[(q['condition'],q['execution_qualifications']) for q in n['rules']]
  assert r['rules'][g]['source_path_index']==source[cid,v]['gold_rule']
  for q in r['rules']:
   assert not re.search(r'\bif\b|\bcriteria\b|in selected patients|in patients with|based on other risk factors',q['action'],re.I)
   for qual in q['execution_qualifications']:assert qual in render(r,'blind') and qual in render(r,'visible')
  contracts['paired_source_labels_conditions_and_qualifications']+=1
 assert vs['AC']['rules'][g]['action']!=n['rules'][g]['action'] and vs['AC']['rules'][other]['action']==n['rules'][other]['action']
 assert vs['IC']['rules'][g]['action']==n['rules'][g]['action'] and vs['IC']['rules'][other]['action']!=n['rules'][other]['action']
 assert vs['SW']['rules'][g]['action']==n['rules'][other]['action'] and vs['SW']['rules'][other]['action']==n['rules'][g]['action']
 if 'CS' in vs:
  c=vs['CS'];proof=c['threshold_proof'];truth=[]
  for q in c['rules']:
   m=re.search(r'([<≥])([0-9.]+) mo from completion of primary therapy',q['condition']);assert m
   truth.append(proof['patient_value']<float(m[2]) if m[1]=='<' else proof['patient_value']>=float(m[2]))
  assert sum(truth)==1 and truth[int(c['gold'])-1] and c['patient']==full
  contracts['verified_CS_scalar_flip']+=1
for model,path in [('qwen','Qwen3-8B'),('llama','Meta-Llama-3.1-8B-Instruct')]:
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained('/home/gcx/TRIAGE/models/modelscope/'+path,local_files_only=True);lengths=[];keys=set()
 for r in rows:
  for condition in ['blind','visible']:
   prompt=tok.apply_chat_template([{'role':'system','content':'Select a branch by matching its clinical conditions to the patient. Follow the supplied experimental rules. Output only 1 or 2.'},{'role':'user','content':render(r,condition)}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
   lengths.append(len(tok.encode(prompt,add_special_tokens=False)));keys.add(hashlib.sha256(prompt.encode()).hexdigest())
 assert max(lengths)<=8192
 contracts[model+'_max_prompt_tokens']=max(lengths);contracts[model+'_unique_P1_prompts']=len(keys)
result={'status':'PASS','checks':dict(contracts),'scope':'specified repair contracts, not human clinical adjudication','test_inference_count':0,'new_model_inference':0,'cases_sha256':hashlib.sha256((D/'private/cases.jsonl').read_bytes()).hexdigest()}
assert result['cases_sha256']==audit['cases_sha256']
(D/'results/repair_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
