"""Numerical native full-attention comparison at prefill and subsequent queries."""
import argparse,json,time
from pathlib import Path
import torch
from .data import read,write
from .classify import model_and_classifier
from .native_memory import query_chunked_attention
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--disagreement',action='store_true');a=p.parse_args();run=Path(a.run);out=run/('native_numeric_disagreement' if a.disagreement else 'native_numeric_reference');out.mkdir(exist_ok=False)
 torch.set_num_threads(4);torch.manual_seed(20261030);tok,model,_=model_and_classifier();cases=read(run/'cases/confirm.jsonl');refs=[]
 for i,c in enumerate(cases):
  p=run/('confirm'+str(i%2))/'units'/c['case_id'].replace(':','_')/'native_official.json';r=json.loads(p.read_text())
  if r['status']=='complete':
   if a.disagreement:
    repaired=run/('native_memory_repair'+str(i%2))/'units'/(c['case_id'].replace(':','_')+'.json')
    if not repaired.exists() or json.loads(repaired.read_text()).get('reference_tokens_equal') is not False:continue
   refs.append((r['result']['input_tokens'],c,r))
 _,case,old=min(refs,key=lambda x:x[0]);lang='zh' if case['language']=='zh' else 'en';name='ZH_DECISION_MAKING_PREDICTION_TEMPLATE.txt' if lang=='zh' else 'DECISION_MAKING_PREDICTION_TEMPLATE.txt'
 path=run/'official_mega/mega_cdp/template'/('template_'+lang)/name
 prompt=path.read_text().replace('{Guideline}',Path(case['original_guideline_path']).read_text()).replace('{Case}',case['patient'])
 text=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
 ids=tok(text,return_tensors='pt',add_special_tokens=False).to('cuda:0')
 with torch.inference_mode():
  full=model(**ids,use_cache=True,return_dict=True,logits_to_keep=1)
  with query_chunked_attention(model,256):chunk=model(**ids,use_cache=True,return_dict=True,logits_to_keep=1)
  logits=[]
  for offset in range(3):
   a0=full.logits[0,-1].float();b0=chunk.logits[0,-1].float();logits.append({'query_offset':offset,'max_abs_error':float((a0-b0).abs().max()),'mean_abs_error':float((a0-b0).abs().mean()),'argmax_equal':int(a0.argmax())==int(b0.argmax())})
   if offset<2:
    token=torch.tensor([[old['result']['token_ids'][offset]]],device='cuda:0')
    full=model(input_ids=token,past_key_values=full.past_key_values,use_cache=True,return_dict=True,logits_to_keep=1)
    with query_chunked_attention(model,256):chunk=model(input_ids=token,past_key_values=chunk.past_key_values,use_cache=True,return_dict=True,logits_to_keep=1)
  keys=[];values=[]
  for (ka,va),(kb,vb) in zip(full.past_key_values.to_legacy_cache(),chunk.past_key_values.to_legacy_cache()):
   keys.append(float((ka.float()-kb.float()).abs().max()));values.append(float((va.float()-vb.float()).abs().max()))
 write(out/'status.json',{'status':'complete','input_tokens':ids['input_ids'].shape[1],'logits':logits,'max_key_error':max(keys),'max_value_error':max(values),'key_error_by_layer':keys,'value_error_by_layer':values,'chunk_size':256,'full_attention_no_key_window':True,'rope_unchanged':True,'dtype_unchanged':True})
if __name__=='__main__':main()

