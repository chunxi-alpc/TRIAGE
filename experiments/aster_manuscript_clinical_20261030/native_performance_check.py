"""Verify original prefix-retention policy after rejecting higher-memory reuse."""
import json,time
import torch
from .data import BASE,read,write
from .classify import model_and_classifier
from .freeze import classification_records
from .engine import ClinicalExecutor
from .decode_graphs import decode_mlp_graphs
from src.aster.manuscript_core import load_expert

def main():
 torch.set_num_threads(4);r=BASE;tok,model,_=model_and_classifier();c=read(r/'cases/development.jsonl')[0]
 raw=next(x for x in classification_records(r) if x['case_id']==c['case_id'])['raw_decisions']
 f=json.loads((r/'frozen.json').read_text());f.update(native_attention_query_chunk_size=2048,native_attention_balanced=True);expert=load_expert(r/'analysis/expert.json')
 e=ClinicalExecutor(model,tok,expert,f);e.execute(c,raw,'text_clip');a=e.execute(c,raw,'native_official')
 e.prefix=None;del e;torch.cuda.empty_cache()
 with decode_mlp_graphs(model):
  e=ClinicalExecutor(model,tok,expert,f,exact_execution=True);e.execute(c,raw,'text_clip');b=e.execute(c,raw,'native_official')
 report={'status':'passed' if a['token_ids']==b['token_ids'] else 'failed','selection':'first development case; no gold-based selection','original_prefix_policy_restored':True,'tokens_equal':a['token_ids']==b['token_ids'],'reference_seconds':a['seconds'],'optimized_seconds':b['seconds'],'tokens':len(a['token_ids']),'peak_memory_gb':torch.cuda.max_memory_allocated()/1e9}
 write(r/'performance01/native_prefix_policy.json',report)
if __name__=='__main__':main()
