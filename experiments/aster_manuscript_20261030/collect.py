"""Independent single-case reference collection on one allowed physical GPU."""
import argparse,json,time,traceback,subprocess
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from src.aster.manuscript_runtime import PromptBuilder,ManuscriptBackbone,aligned_triplet
from src.aster import ActionBlindBranchClassifier,HFLabelScorer
from .prepare import read,write,verify

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--shard',type=int,required=True);a=p.parse_args()
 out=Path(a.output);out.mkdir(exist_ok=False,parents=True);torch.set_num_threads(4);torch.manual_seed(20261030);start=time.time()
 rows=read(Path(a.data)/'pairs.jsonl');path='/home/gcx/TRIAGE/models/modelscope/Qwen3-8B'
 tok=AutoTokenizer.from_pretrained(path,local_files_only=True);builder=PromptBuilder(tok,chat_template_kwargs={'enable_thinking':False})
 model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager').to('cuda:0').eval()
 backend=ManuscriptBackbone(model,model_id='Qwen/Qwen3-8B',model_revision='local-modelscope-snapshot-20261014-upstream-revision-unavailable')
 classifier=ActionBlindBranchClassifier(HFLabelScorer(model,tok,max_input_tokens=8192),temperature=1.,probability_threshold=0.,margin_threshold=0.)
 failures=[];complete=[]
 for i,row in enumerate(rows):
  if i%2!=a.shard:continue
  dest=out/(str(i)+'.pt')
  try:
   verify(row);target,prompts=aligned_triplet(builder,row)
   snapshots={s:backend.extract(q,target,keys=True,queries=False) for s,q in prompts.items()}
   on={s:builder.prepare(row['guideline'],row['states'][s],row['question'],online=True) for s in ['positive','negative']}
   q={s:backend.extract(x,target,keys=False,queries=True) for s,x in on.items()}
   assert torch.equal(on['positive'].input_ids[:,:on['positive'].prefix_length],on['negative'].input_ids[:,:on['negative'].prefix_length])
   cls={}
   if row['split'] in ['calibration','development','heldout_exploratory']:
    for s in row['states']:
     try:cls[s]=classifier.run(row['guideline'],row['states'][s],include_logits=True)['groups'][0]
     except (ValueError,RuntimeError) as e:cls[s]={'failure':str(e)}
   record={'row':row,'offline':{s:snap.keys for s,snap in snapshots.items()},'queries':{s:snap.queries for s,snap in q.items()},
    'cos':q['positive'].cos,'sin':q['positive'].sin,'classification':cls,'identity':backend.identity,'template_id':builder.template_id,
    'tokens':{s:{'ids':p.input_ids[0].tolist(),'positions':p.positions} for s,p in prompts.items()}}
   torch.save(record,dest);complete.append(i)
  except Exception as e:
   failures.append({'index':i,'pair_id':row['pair_id'],'split':row['split'],'reason':type(e).__name__+': '+str(e)})
   traceback.print_exc()
  write(out/'status.json',{'status':'running','complete':complete,'failures':failures,'submitted':len([j for j in range(len(rows)) if j%2==a.shard])})
 write(out/'status.json',{'status':'complete','complete':complete,'failures':failures,'submitted':len([j for j in range(len(rows)) if j%2==a.shard]),
  'elapsed_seconds':time.time()-start,'peak_memory_gb':torch.cuda.max_memory_allocated()/1e9,'identity':backend.identity,
  'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()})
if __name__=='__main__':main()
