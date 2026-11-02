"""Pinned official MEGA evaluation with local Qwen transport, explicitly not GPT."""
import argparse,json,sys,time
from contextlib import nullcontext
from .prefill_replay import replay_prefill
from pathlib import Path
from types import SimpleNamespace
import torch
from .data import ROOT,BASE,REV,write
from .engine import parse_prediction
OFFICIAL=BASE/'official_mega'
sys.path.insert(0,str(OFFICIAL/'mega_cdp/scripts/evaluate'))
from parse_prediction_output_singleturn import parse_record
import pathway_consistency_eval_batch_singleturn as pcc
import accuracy_eval_batch_singleturn as acc
from gt_path_references import load_gt_path_references
from pathway_metrics import dynamic_time_warping
SOURCE_COMMIT='15a3a12e73a9169a870bff2ac431ec2e9968b8c7'
class LocalOfficialJudge:
 def __init__(self,model,tok,exact_execution=False):
  self.model,self.tok=model,tok;self.refs={s:load_gt_path_references(ROOT/s/'gt_paths.jsonl') for s in ['train_val','test']}
  self.args=argparse.Namespace(expected_results=5,qwen_max_tokens=8192);self.exact_execution=exact_execution;self.calls=[]
 def canonical(self,case,text):
  raw={'id':case['case_id'],'case_path_id':case['guideline_id'],'path_index':case['path_id'],'filename':case['filename'],'language':case['language'],'llm_output':text}
  try:parse_record(raw);return text,'raw_official_headers'
  except Exception:pass
  leaf,path,invalid=parse_prediction(text)
  if invalid:raise ValueError('no official-format response or parseable declared tree outcome')
  rules={r['rule_id']:r for g in case['guideline']['groups'] for r in g['rules']}
  steps=[rules[r]['condition'] for r in path]
  if leaf=='NONE':outcome='The documented facts do not support a complete applicable guideline pathway.'
  else:
   node=int(leaf[1:]);found=[r for r in case['tree_mapping'].values() if r['node']==node and r['leaf']]
   if not found:raise ValueError('predicted leaf id is not in the source guideline')
   outcome=found[0]['description']
  zh=case['language']=='zh'
  header='### 临床决策过程：' if zh else '### Clinical Decision Process:'
  final='### 临床决策结论：' if zh else '### Clinical Decision Outcome:'
  return header+'\n'+'\n'.join(f'{i+1}. {s}' for i,s in enumerate(steps or ['Cannot establish an applicable decision from the available facts.']))+'\n'+final+' '+outcome,'source_tree_id_rendering_no_gold'
 def chat(self,client,args,prompt):
  call_start=time.perf_counter()
  rendered=self.tok.apply_chat_template([{'role':'system','content':prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
  inputs=self.tok(rendered,return_tensors='pt',add_special_tokens=False).to(self.model.device)
  if inputs['input_ids'].shape[1]+args.qwen_max_tokens>self.model.config.max_position_embeddings:raise ValueError('judge context budget exceeded')
  with torch.inference_mode():
   out=self.model.generate(**inputs,max_new_tokens=args.qwen_max_tokens,do_sample=True,temperature=.7,top_p=.8,top_k=20,use_cache=True,pad_token_id=self.tok.eos_token_id)
  self.calls.append({'input_tokens':int(inputs['input_ids'].shape[1]),'generated_tokens':int(out.shape[1]-inputs['input_ids'].shape[1]),'seconds':time.perf_counter()-call_start})
  return self.tok.decode(out[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
 def score(self,case,text,seed):
  torch.manual_seed(seed);start=time.time();self.calls=[]
  canonical,rendering=self.canonical(case,text)
  record={'id':case['case_id'],'case_path_id':case['guideline_id'],'path_index':case['path_id'],'filename':case['filename'],'language':case['language'],'llm_output':canonical}
  parsed=parse_record(record);refs=self.refs[case['source_split']]
  pcc.qwen_chat=self.chat;acc.qwen_chat=self.chat
  context=replay_prefill(self.model) if self.exact_execution else nullcontext()
  with context as replay:
   path=pcc.evaluate_record(parsed,refs,None,self.args);outcome=acc.evaluate_record(parsed,refs,None,self.args)
   prefill=replay.stats() if replay is not None else None
  ok='results' in path and 'results' in outcome and len(path['results'])==len(outcome['results'])==5 and not path.get('skipped') and not outcome.get('skipped')
  accuracy=sum((x-1)/4 for x in outcome['results'])/5 if 'results' in outcome else None
  cost=None
  if 'results' in path:
   matrix=torch.tensor(path['results'],dtype=torch.float64).mean(0);_,cost,_=dynamic_time_warping((1-(matrix-1)/4).tolist())
  return {'status':'complete' if ok else 'failed','scoring_failure':not ok,'Qwen_judged_official_accuracy':accuracy,'Qwen_judged_official_PCC':cost,
   'path_draws':path,'outcome_draws':outcome,'expected_draws':5,'independent_generate_calls':len(self.calls),'generate_call_timing':self.calls,'prefill_replay':prefill,'judge_model':'Qwen/Qwen3-8B','judge_revision':REV,'official_source_commit':SOURCE_COMMIT,
   'transport':'in-process Transformers replaces SGLang HTTP; no remote clinical-data transfer','rendering':rendering,'seed':seed,'seconds':time.time()-start,
   'not_GPT_main_table':True,'canonical_text_private':canonical}
def score_case(judge,case,casepath,methods,case_index):
 cached=[]
 for mi,method in enumerate(methods):
  path=casepath/(method+'.json');output=casepath/'official_scores'/(method+'.json')
  if output.exists():
   prior=json.loads(output.read_text())
   if 'canonical_text_private' in prior:cached.append((prior['canonical_text_private'],prior))
   continue
  if not path.exists():continue
  row=json.loads(path.read_text());seed=20261030+case_index*100+mi
  try:
   if row['status']!='complete':raise ValueError('prediction unit failed')
   representation,rendering=judge.canonical(case,row['result']['text'])
   match=next((score for text,score in cached if text==representation),None)
   if match is None:
    result=judge.score(case,row['result']['text'],seed);cached.append((representation,result))
   else:result=dict(match,shared_identical_representation=True)
  except Exception as e:result={'status':'failed','scoring_failure':True,'error_type':type(e).__name__,'error':str(e),'seed':seed,'not_GPT_main_table':True}
  output.parent.mkdir(parents=True,exist_ok=True);temp=output.with_suffix('.partial');write(temp,result);temp.replace(output)

