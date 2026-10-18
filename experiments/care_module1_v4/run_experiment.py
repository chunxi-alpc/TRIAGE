"""Single frozen holdout run. Gold files are never read during inference."""
import argparse,hashlib,json,os,pathlib,platform,time
from specs import FIELDS,NUMERIC
from grounding import locate,normalize,proof
from logic import evaluate,valid_schema
def read(p):return [json.loads(s) for s in p.read_text().splitlines()]
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def parse(s):
 try:return json.loads(s)
 except Exception:
  try:return json.loads(s[s.index('{'):s.rindex('}')+1])
  except Exception:return None
POLICY='Use only stated patient facts. Missing, pending, uncertain or withdrawn entries are unknown, never negative or zero. Explicit absence is negative. Resolve negation and distinguish subject, specimen, lesion, units, historical event and assessment time. Explicit signed corrections replace the withdrawn entry. For historical fields use the specified historical event, not later treatment. Two incompatible reports for the SAME time, event and specimen with neither superseded mean conflict. Uncertainty in one field does not automatically make an AND/OR rule unknown: use strong three-valued logic. Bounds establish truth only if every permitted value has the same truth. A planned scan is not a result; patient-specific planned radiotherapy can be a fact if the field asks for that plan. Return JSON only.'
COMMON='Research predicate evaluation, not treatment selection. '+POLICY
def main(a):
 import torch,transformers
 from transformers import AutoTokenizer,AutoModelForCausalLM
 root=a.data;out=root/'model';out.mkdir(exist_ok=False)
 for p,h in json.loads(pathlib.Path('FREEZE.json').read_text())['sha256'].items():assert hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==h,p
 inputs=read(root/'private/inputs.jsonl');rules=read(root/'private/rules_inputs.jsonl');rb={r['rule_id']:r for r in rules}
 torch.manual_seed(20261018);tk=AutoTokenizer.from_pretrained(a.model,local_files_only=True,padding_side='left');tk.pad_token_id=tk.eos_token_id
 t=time.perf_counter();m=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda:0').eval()
 env={'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,'cuda':torch.version.cuda,'model_path':str(a.model),'gpu':torch.cuda.get_device_name(0),'visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],'model_load_seconds':time.perf_counter()-t,'start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'generation':{'do_sample':False,'enable_thinking':False,'dtype':'bfloat16','batch':a.batch,'seed':20261018}}
 dump(out/'environment.json',env)
 def generate(jobs,stage,maxtokens):
  res={};timings=[]
  for start in range(0,len(jobs),a.batch):
   bs=jobs[start:start+a.batch];texts=[tk.apply_chat_template([{'role':'system','content':COMMON},{'role':'user','content':j['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=False) for j in bs]
   enc=tk(texts,padding=True,return_tensors='pt').to('cuda:0');torch.cuda.synchronize();t=time.perf_counter()
   with torch.inference_mode():seq=m.generate(**enc,max_new_tokens=maxtokens,do_sample=False,pad_token_id=tk.pad_token_id,eos_token_id=tk.eos_token_id)
   torch.cuda.synchronize();elapsed=time.perf_counter()-t;timings.append({'batch_start':start,'n':len(bs),'seconds':elapsed,'input_tokens_padded':enc.input_ids.shape[1],'output_tokens_padded':seq.shape[1]-enc.input_ids.shape[1]})
   with (out/(stage+'_raw.jsonl')).open('a') as f:
    for j,ids,txt in zip(bs,seq,texts):
     raw=tk.decode(ids[enc.input_ids.shape[1]:],skip_special_tokens=True);rec={'id':j['id'],'raw_output':raw,'parsed':parse(raw),'prompt_sha256':hashlib.sha256(txt.encode()).hexdigest(),'amortized_seconds':elapsed/len(bs)};res[j['id']]=rec;f.write(json.dumps(rec,ensure_ascii=False)+'\n')
   print(json.dumps({'stage':stage,'completed':start+len(bs),'total':len(jobs)}),flush=True)
  dump(out/(stage+'_timing.json'),timings);return res
 dictionary=json.dumps(FIELDS)
 rp='Translate ONLY this predicate into {"mode":"all|any","conditions":[{"field":FIELD,"op":"lt|le|gt|ge|eq","value":number or "positive" or "negative"}],"exceptions":[]}. Use exact field keys. Inclusive ranges need ge and le. Do not add inferred eligibility conditions. Dictionary: '+dictionary+'\nPredicate: '
 rex=generate([{'id':r['rule_id'],'prompt':rp+r['text']} for r in rules],'rule_extraction',512)
 keys={rid:sorted({c.get('field') for c in (x['parsed'] or {}).get('conditions',[]) if isinstance(c,dict) and c.get('field') in FIELDS}) for rid,x in rex.items()}
 base={};directjobs=[];factjobs=[]
 form='Return {"facts":{FIELD:VALUE},"evidence":{FIELD:[{"sentence_id":INTEGER,"quote":"verbatim supporting substring"}]},"status":{FIELD:"known|missing|conflict"}}. VALUE is a number for numeric fields; positive/negative for categorical fields; a bounds object such as {"gt":50} or {"ge":1,"le":2} for ranges; null for missing/conflict. Numeric strings are not needed. For conflict cite both incompatible reports. For missing use []. Quote 8-160 characters, exactly from a single numbered sentence, including the relevant subject and value or negation. Do not return a rule label or invent a point from a bound.'
 for x in inputs:
  sid=x['sample_id'];rid=x['rule_id'];fs=keys[rid];case='\n'.join(f'[{i+1}] {s}' for i,s in enumerate(x['sentences']));base[sid]=case
  directjobs.append({'id':sid,'prompt':'Field meanings: '+dictionary+'\nDecide the predicate directly. Return {"label":"T|F|U","evidence_sentence_ids":[integer IDs]}. T=satisfied, F=contradicted, U=insufficient/conflicting. Cite a sufficient proof. Missing-data U uses []; conflicting-data U cites both reports.\nPredicate: '+rb[rid]['text']+'\nComplete case:\n'+case})
  factjobs.append({'id':sid,'prompt':'Field meanings: '+dictionary+'\nExtract only '+json.dumps(fs)+'. '+form+'\nComplete case:\n'+case})
 direct=generate(directjobs,'direct',160);facts=generate(factjobs,'facts',896)
 vjobs=[]
 for x in inputs:
  sid=x['sample_id'];fs=keys[x['rule_id']]
  vjobs.append({'id':sid,'prompt':'Field meanings: '+dictionary+'\nVerify and correct the following candidate extraction against the COMPLETE case. Independently check every required field '+json.dumps(fs)+'. Correct incorrect values and wrong sentence numbers, recover explicitly stated facts, keep true missingness, and preserve unresolved conflicts. A correct paraphrased fact can be supported by a differently worded source sentence; do not demand adjacent literal keywords. Only original source text, not candidate quotes, is evidence. Candidate: '+json.dumps(facts[sid]['parsed'])+'\n'+form+'\nComplete case:\n'+base[sid]})
 verified=generate(vjobs,'verification',896);rows=[];records=[];t=time.perf_counter()
 for x in inputs:
  sid=x['sample_id'];rid=x['rule_id'];rule=rex[rid]['parsed'];fs=keys[rid];sents=x['sentences'];d=direct[sid]['parsed'] or {};di,da,ds=locate(d.get('evidence_sentence_ids'),sents)
  raw=normalize(facts[sid]['parsed'],sents,fs,NUMERIC,False);full=normalize(verified[sid]['parsed'],sents,fs,NUMERIC,True)
  valid=valid_schema(rule) and bool(fs) and all(c['field'] in FIELDS for c in rule['conditions']+rule.get('exceptions',[]))
  def pred(rec):return {'label':evaluate(rule,rec['facts']) if valid else 'INVALID','evidence_sentence_ids':proof(rule,rec) if valid else [],'valid':valid}
  rows.append({'sample_id':sid,'rule_id':rid,'direct':{'label':d.get('label') if d.get('label') in ['T','F','U'] else 'INVALID','evidence_sentence_ids':di,'valid':d.get('label') in ['T','F','U']},'structured_no_guard':pred(raw),'structured':pred(full)})
  records.append({'sample_id':sid,'unverified':raw,'verified':full})
 for name,items in [('predictions.jsonl',rows),('normalized_records.jsonl',records)]:
  (out/name).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in items))
 env.update({'postprocessing_seconds':time.perf_counter()-t,'end_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30});dump(out/'environment.json',env);print('COMPLETE',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',type=pathlib.Path,default=pathlib.Path('data'));p.add_argument('--model',required=True);p.add_argument('--batch',type=int,default=4);main(p.parse_args())
