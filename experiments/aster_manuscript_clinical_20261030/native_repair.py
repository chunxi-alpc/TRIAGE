"""New run for all native cases, retaining every old OOM and numerical reference."""
import argparse,json,time,traceback
from pathlib import Path
import torch
from .data import read,write
from .classify import model_and_classifier
from .engine import ClinicalExecutor
from src.aster.manuscript_core import load_expert
from .native_memory import query_chunked_attention
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--shard',type=int,required=True);a=p.parse_args();run=Path(a.run);out=run/('native_memory_repair'+str(a.shard));out.mkdir(exist_ok=False);torch.set_num_threads(4);torch.manual_seed(20261030);start=time.time()
 tok,model,_=model_and_classifier();freeze=json.loads((run/'frozen.json').read_text());engine=ClinicalExecutor(model,tok,load_expert(run/'analysis/expert.json'),freeze)
 cases=read(run/'cases/confirm.jsonl');refs=[];failures=[];complete=0
 for i,case in enumerate(cases):
  if i%2!=a.shard:continue
  original=run/('confirm'+str(i%2))/'units'/case['case_id'].replace(':','_')/'native_official.json'
  old=json.loads(original.read_text());destination=out/'units'/(case['case_id'].replace(':','_')+'.json');destination.parent.mkdir(exist_ok=True)
  try:
   with query_chunked_attention(model,256):result=engine.native(case)
   comparison=None if old['status']!='complete' else result['token_ids']==old['result']['token_ids']
   if comparison is not None:refs.append(comparison)
   write(destination,{'case_id':case['case_id'],'source_id':case['source_id'],'method':'native_official_query_chunked_full_attention','status':'complete','old_status':old['status'],'reference_tokens_equal':comparison,'result':result,'old_prediction_preserved':True})
  except Exception as e:
   failures.append(type(e).__name__+': '+str(e));write(destination,{'case_id':case['case_id'],'source_id':case['source_id'],'status':'failed','error_type':type(e).__name__,'error':str(e)});traceback.print_exc();torch.cuda.empty_cache()
  complete+=1;write(out/'status.json',{'status':'running','cases':complete,'failures':len(failures),'reference_cases':len(refs),'reference_tokens_equal':sum(refs)})
 write(out/'status.json',{'status':'complete','cases':complete,'failures':len(failures),'reference_cases':len(refs),'reference_tokens_equal':sum(refs),'elapsed_seconds':time.time()-start,'peak_memory_gb':torch.cuda.max_memory_allocated()/1e9,'query_chunk_size':256,'all_keys_preserved':True,'context_limit_changed':False,'old_results_overwritten':False})
if __name__=='__main__':main()

