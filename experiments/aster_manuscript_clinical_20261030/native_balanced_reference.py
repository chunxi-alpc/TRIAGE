"""Recheck every previously successful native generation, not a score-selected subset."""
import argparse,json,time
from pathlib import Path
import torch
from src.aster.manuscript_core import load_expert
from .data import read,write
from .classify import model_and_classifier
from .engine import ClinicalExecutor
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--shard',type=int,required=True);a=p.parse_args();run=Path(a.run);out=run/('native_balanced_reference'+str(a.shard));out.mkdir(exist_ok=False);torch.set_num_threads(4);torch.manual_seed(20261030);start=time.time()
 tok,model,_=model_and_classifier();freeze=json.loads((run/'frozen.json').read_text());freeze.update(native_attention_query_chunk_size=2048,native_attention_balanced=True)
 engine=ClinicalExecutor(model,tok,load_expert(run/'analysis/expert.json'),freeze);n=0;same=0;records=[]
 for i,c in enumerate(read(run/'cases/confirm.jsonl')):
  if i%2!=a.shard:continue
  old=json.loads((run/('confirm'+str(i%2))/'units'/c['case_id'].replace(':','_')/'native_official.json').read_text())
  if old['status']!='complete':continue
  result=engine.native(c);equal=result['token_ids']==old['result']['token_ids'];n+=1;same+=equal
  records.append({'case_id':c['case_id'],'equal':equal,'old_tokens':len(old['result']['token_ids']),'new_tokens':len(result['token_ids'])})
  write(out/'status.json',{'status':'running','reference_cases':n,'token_identical_cases':same,'chunk_size':2048,'balanced':True})
 write(out/'private_reference_records.json',records)
 write(out/'status.json',{'status':'complete','reference_cases':n,'token_identical_cases':same,'all_reference_generations_identical':same==n,'chunk_size':2048,'balanced':True,'elapsed_seconds':time.time()-start,'selection_basis':'all native technical successes, regardless of correctness'})
if __name__=='__main__':main()

