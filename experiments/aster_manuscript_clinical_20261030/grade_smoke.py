"""Two source-first clinical cases exercise the actual pinned official Qwen scorer."""
import argparse,json,time
from pathlib import Path
import torch
from .data import read,write
from .classify import model_and_classifier
from .engine import METHODS
from .official_score import LocalOfficialJudge,score_case
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--shard',type=int,required=True);a=p.parse_args();run=Path(a.run);out=run/('official_smoke'+str(a.shard));out.mkdir(exist_ok=False)
 torch.set_num_threads(4);torch.manual_seed(20261030);start=time.time();tok,model,_=model_and_classifier();judge=LocalOfficialJudge(model,tok)
 cases=read(run/'cases/confirm.jsonl');first=[];sources=set()
 for i,c in enumerate(cases):
  if c['source_id'] not in sources:first.append((i,c));sources.add(c['source_id'])
 selected=[v for j,v in enumerate(first) if j%2==a.shard];counts={'complete':0,'failed':0}
 for i,case in selected:
  parent=run/('confirm'+str(i%2))/'units'/case['case_id'].replace(':','_')
  score_case(judge,case,parent,METHODS,i)
  for m in METHODS:
   path=parent/'official_scores'/(m+'.json')
   if path.exists():counts[json.loads(path.read_text())['status']]+=1
 write(out/'status.json',{'status':'complete','source_first_cases':len(selected),'graded_units':sum(counts.values()),'scores':counts,'scope':'two-source first-case scorer smoke only; not full clinical semantic assessment','elapsed_seconds':time.time()-start})
if __name__=='__main__':main()

