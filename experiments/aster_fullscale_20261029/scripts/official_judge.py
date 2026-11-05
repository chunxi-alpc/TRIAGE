"""Official MEGA parser, five Qwen judge samples and official DTW.
The only transport adaptation is in-process Transformers in place of SGLang HTTP.
"""
import sys,json,argparse
from pathlib import Path
import torch
ROOT=Path('/home/gcx/aster_fullscale_20261030/official_mega')
sys.path.insert(0,str(ROOT/'mega_cdp/scripts/evaluate'))
from parse_prediction_output_singleturn import parse_record
import pathway_consistency_eval_batch_singleturn as pcc
import accuracy_eval_batch_singleturn as acc
from gt_path_references import load_gt_path_references
from pathway_metrics import dynamic_time_warping
class Judge:
 def __init__(self,e,source='/home/gcx/TRIAGE/data/MEGA-CDP'):
  self.e=e;self.refs={}
  for split in ('train_val','test'):
   self.refs[split]=load_gt_path_references(Path(source)/split/'gt_paths.jsonl')
  self.args=argparse.Namespace(expected_results=5,qwen_max_tokens=8192)
 def chat(self,client,args,prompt):
  e=self.e;e.active=None
  text=e.tok.apply_chat_template([{'role':'system','content':prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
  inp,_=e.encode(text)
  with torch.inference_mode():
   out=e.model.generate(**inp,max_new_tokens=args.qwen_max_tokens,do_sample=True,temperature=.7,top_p=.8,top_k=20,use_cache=True,pad_token_id=e.tok.eos_token_id)
  return e.tok.decode(out[0,inp['input_ids'].shape[1]:],skip_special_tokens=True)
 def score(self,c,raw):
  _,split,gid,pid=c['case_id'].split(':');gid=int(gid);pid=int(pid)
  refs=self.refs[split];ref=next(v for k,v in refs.items() if k[0]==gid and k[1]==pid)
  record={'id':c['case_id'],'case_path_id':gid,'path_index':pid,'filename':ref['filename'],'language':ref['language'],'llm_output':raw}
  try:parsed=parse_record(record)
  except Exception as ex:return False,True,{'official_pcc':1.,'official_accuracy':0.,'official_failure':str(ex)}
  pcc.qwen_chat=self.chat;acc.qwen_chat=self.chat
  p=pcc.evaluate_record(parsed,refs,None,self.args);a=acc.evaluate_record(parsed,refs,None,self.args)
  accuracy=sum((x-1)/4 for x in a['results'])/5 if 'results' in a else 0.
  cost=1.
  if 'results' in p:
   matrix=torch.tensor(p['results']).mean(0);difference=(1-(matrix-1)/4).tolist();_,cost,_=dynamic_time_warping(difference)
  return accuracy==1.,False,{'official_pcc':cost,'official_accuracy':accuracy,'official_pathway_judge':p,'official_accuracy_judge':a,'official_judge_failed':bool(p.get('skipped') or a.get('skipped')),'parsed_native_output':parsed['llm_output']}
