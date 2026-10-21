"""Identical-input Qwen paired scoring; gold is loaded only by summarize.py."""
import argparse,hashlib,json,time,os,traceback
from pathlib import Path
import torch
from qwen_engine import QwenEngine,MODEL
P=Path(__file__).resolve().parent
class Engine(QwenEngine):
 def encode_case(self,c,states):
  prompt='Use the following clinical guideline rules to answer the multiple-choice question.\nGuideline rules:\n';spans={}
  for r in c['rules']:
   prefix=r['text'][:-len(r['action'])-1];prompt+=r['rule_id']+': '+prefix
   start=len(prompt);prompt+=r['action'];spans[r['rule_id']]=(start,len(prompt));prompt+='\n'
  prompt+='\nPatient: '+c['patient']+'\nSelect the best option from the following choices:\n'
  prompt+='\n'.join(chr(65+i)+': '+o for i,o in enumerate(c['options']))
  prompt+='\n\nReturn only the letter corresponding to your choice and nothing else. Answer:'
  enc=self.tokenizer(prompt,return_tensors='pt',return_offsets_mapping=True,add_special_tokens=False);offsets=enc.pop('offset_mapping')[0].tolist()
  signs=torch.zeros(len(offsets),device=self.device,dtype=torch.float32);targets=[]
  for r in c['rules']:
   sign={'T':1,'F':-1}.get(states.get(r['rule_id']),0)
   if not sign:continue
   a,b=spans[r['rule_id']];assert prompt[a:b]==r['action']
   inds=[i for i,(x,y) in enumerate(offsets) if y>a and x<b and y>x]
   assert inds
   signs[inds]=sign;targets.append({'rule_id':r['rule_id'],'sign':sign,'token_indices':inds})
  return prompt,{k:v.to(self.device) for k,v in enc.items()},signs,targets
 def pair(self,c,states):
  prompt,inputs,signs,targets=self.encode_case(c,states)
  ids=[self.tokenizer.encode(' '+chr(65+i),add_special_tokens=False) for i in range(len(c['options']))]
  assert all(len(x)==1 for x in ids)
  result={'case_id':c['case_id'],'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'prompt_tokens':len(signs),'target_token_count':int((signs!=0).sum()),'targets':targets,'arms':{}}
  for arm,layers in [('baseline',[]),('a1_attention',list(range(18,24)))]:
   self.current={'layers':layers,'strength':1.,'query_window':1,'method':'direct'};self.token_signs=signs;self.has_targets=bool(targets);self.layer_diagnostics=[]
   start=time.monotonic()
   with torch.inference_mode(): logits=self.model(**inputs,use_cache=False,logits_to_keep=1).logits[0,-1].float()
   v=logits[[x[0] for x in ids]].cpu().tolist();pred=chr(65+max(range(len(v)),key=v.__getitem__))
   result['arms'][arm]={'prediction':pred,'candidate_logits':v,'seconds':time.monotonic()-start,'attention_diagnostics':self.layer_diagnostics,'status':'completed'}
   self.current=None
  if not targets:assert result['arms']['baseline']['candidate_logits']==result['arms']['a1_attention']['candidate_logits']
  return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True);args=ap.parse_args();k=args.shard
 cases=[json.loads(l) for l in (P/f'private/input_{k}.jsonl').read_text().splitlines()]
 records={r['case_id']:r for r in map(json.loads,(P/f'private/a1_{k}/predictions.jsonl').read_text().splitlines())}
 assert set(records)=={c['case_id'] for c in cases}
 out=P/f'private/qwen_{k}.jsonl';done={json.loads(l)['case_id'] for l in out.read_text().splitlines()} if out.exists() else set()
 start=time.monotonic();torch.cuda.reset_peak_memory_stats();engine=Engine();manifest={'status':'running','shard':k,'n':len(cases),'provenance':engine.provenance}
 mp=P/f'private/qwen_{k}_manifest.json';mp.write_text(json.dumps(manifest,indent=2))
 with out.open('a',buffering=1) as f:
  for c in cases:
   if c['case_id'] in done:continue
   states={r['rule_id']:r['state'] for r in records[c['case_id']]['rules']}
   try:r=engine.pair(c,states)
   except Exception as e:
    r={'case_id':c['case_id'],'error':repr(e),'arms':{x:{'status':'failed','prediction':None} for x in ['baseline','a1_attention']}};traceback.print_exc();torch.cuda.empty_cache()
   f.write(json.dumps(r)+'\n')
 manifest.update(status='completed',elapsed_seconds=time.monotonic()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
 mp.write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)
if __name__=='__main__':main()
