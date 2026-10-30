"""Cache unedited backbone group logits for independent calibration/development."""
import argparse,json,time,traceback,subprocess
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from src.aster import ActionBlindBranchClassifier,HFLabelScorer
from .data import read,write,MODEL,REV
def model_and_classifier():
 tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
 model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager').to('cuda:0').eval()
 classifier=ActionBlindBranchClassifier(HFLabelScorer(model,tok,max_input_tokens=40960),temperature=1.,probability_threshold=0.,margin_threshold=0.)
 return tok,model,classifier
def classify_case(classifier,case):
 raw={};failures={};start=time.time()
 for group in case['guideline']['groups']:
  gid=group['group_id']
  try:raw[gid]=classifier.run({'groups':[group]},case['patient'],include_logits=True)['groups'][0]
  except (ValueError,RuntimeError) as e:raw[gid]=None;failures[gid]=type(e).__name__+': '+str(e)
 return {'raw_decisions':raw,'classification_failures':failures,'classification_seconds':time.time()-start,'classification_forwards':len(raw)}
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--shard',required=True,type=int);a=p.parse_args();run=Path(a.run);out=run/('classification'+str(a.shard));out.mkdir(exist_ok=False);torch.set_num_threads(4);torch.manual_seed(20261030);start=time.time()
 tok,model,classifier=model_and_classifier();rows=read(run/'cases/calibration.jsonl')+read(run/'cases/development.jsonl')
 complete=0;failed=0
 with (out/'records.jsonl').open('x') as f:
  for i,case in enumerate(rows):
   if i%2!=a.shard:continue
   record={'case':case,'case_id':case['case_id'],'model_revision':REV}
   if 'source_failure' in case:record.update(status='failed',error=case['source_failure']);failed+=1
   else:
    try:record.update(classify_case(classifier,case),status='complete')
    except Exception as e:record.update(status='failed',error=type(e).__name__+': '+str(e));failed+=1;traceback.print_exc()
   f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();complete+=1
   write(out/'status.json',{'status':'running','cases':complete,'technical_failures':failed})
 write(out/'status.json',{'status':'complete','cases':complete,'technical_failures':failed,'elapsed_seconds':time.time()-start,'model_revision':REV,'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()})
if __name__=='__main__':main()

