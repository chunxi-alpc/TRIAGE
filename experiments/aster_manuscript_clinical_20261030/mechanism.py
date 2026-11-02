"""Gold-root mechanism diagnosis on clinical triples; never a deployable ranking."""
import argparse,json,time
from pathlib import Path
import torch
from src.aster.manuscript_core import load_expert
from .data import read,write,clinical_validator
from .analyze import load
from .classify import model_and_classifier
from .engine import ClinicalExecutor,parse_prediction
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--shard',type=int,required=True);a=p.parse_args();run=Path(a.run);out=run/('mechanism'+str(a.shard));out.mkdir(exist_ok=False);torch.set_num_threads(4);torch.manual_seed(20261030);start=time.time()
 f=json.loads((run/'frozen.json').read_text());expert=load_expert(run/'analysis/expert.json');tok,model,cls=model_and_classifier();engine=ClinicalExecutor(model,tok,expert,f)
 rows=[r for r in load(run) if r['row']['split']=='confirm'];methods=['baseline','zero','spectral','mean','random','reverse','score_bias','enhance_only','suppress_only'];units=0;failures=0
 with (out/'results_private.jsonl').open('x') as handle:
  for i,r in enumerate(rows):
   if i%2!=a.shard:continue
   row=r['row'];truths=clinical_validator(row)
   for state,patient in row['states'].items():
    truth=truths[state];expected='NONE' if truth is None else 'R0Y' if truth else 'R0N'
    case={'case_id':row['pair_id']+':'+state,'source_id':row['source_id'],'guideline':row['guideline'],'patient':patient,'question':row['question'],'language':'zh'}
    raw={g['group_id']:None for g in row['guideline']['groups']};raw['G0']=r['classification'][state] if 'logits' in r['classification'].get(state,{}) else None
    oracle={gid:None for gid in raw};oracle['G0']={'prediction':expected,'abstain':False,'diagnostic_only':True}
    for route in ['gold_root_diagnostic','predicted_root_diagnostic']:
     for method in methods:
      record={'pair_id':row['pair_id'],'source_id':row['source_id'],'state':state,'route':route,'method':method,'expected_root':expected,'clinical_root_only':True,'full_leaf_gold_available':False}
      try:
       result=engine.execute(case,raw,method,decisions_override=oracle if route=='gold_root_diagnostic' else None);result.pop('first_logits',None)
       leaf,path,invalid=parse_prediction(result['text']);predicted=path[0] if path else 'NONE' if leaf=='NONE' else None
       record.update(status='complete',root_correct=predicted==expected,result=result)
      except Exception as e:failures+=1;record.update(status='failed',error_type=type(e).__name__,error=str(e),root_correct=False)
      handle.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');handle.flush();units+=1
   write(out/'status.json',{'status':'running','units':units,'technical_failures':failures})
 write(out/'status.json',{'status':'complete','units':units,'technical_failures':failures,'elapsed_seconds':time.time()-start,'scope':'isolated clinical root mechanism, not complete clinical path gold'})
if __name__=='__main__':main()

