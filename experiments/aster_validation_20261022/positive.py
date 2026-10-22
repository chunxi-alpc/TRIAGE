"""Official SEKA projection + CounterFact smoke reproduction, distinct from ASTER."""
import argparse,hashlib,json,sys,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--seka-root',required=True);p.add_argument('--model',required=True);p.add_argument('--projection',required=True);p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--limit',type=int,default=200);a=p.parse_args()
 sys.path.insert(0,a.seka_root)
 from src.model import SEKALLM
 import torch
 model=SEKALLM(a.model,pos_pt=a.projection,layers='last10',amplify_pos=1.56,amplify_neg=0.,torch_dtype=torch.bfloat16,attn_implementation='eager',local_files_only=True)
 data=json.loads(Path(a.data).read_text());data=list(data.values()) if isinstance(data,dict) else data
 output=[];started=time.time()
 for ex in data[:a.limit]:
  rw=ex['requested_rewrite'];question=rw['prompt'].format(rw['subject']);target=rw['target_new']['str'];prior=rw['target_true']['str']
  text='Follow the supplied fact for this question.\nFact: **'+question+' '+target+'**\nQuestion: '+question+'\nAnswer:'
  r={'case_id':ex['case_id'],'target':target,'prior':prior}
  for steer in (False,True):
   answer=model.generate(text,steer=steer,do_sample=False,max_new_tokens=32,pad_token_id=model.tok.eos_token_id)
   r['steered' if steer else 'baseline']={'answer':answer,'new_target_prefix':answer.strip().casefold().startswith(target.strip().casefold())}
  output.append(r)
 Path(a.output).write_text(''.join(json.dumps(r)+'\n' for r in output))
 summary={'n':len(output),'data_sha256':hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),'projection_sha256':hashlib.sha256(Path(a.projection).read_bytes()).hexdigest(),'seconds':time.time()-started,'baseline':sum(r['baseline']['new_target_prefix'] for r in output),'seka':sum(r['steered']['new_target_prefix'] for r in output),'protocol':'CounterFact efficacy smoke with official SEKALLM, 200 cases, regenerated 200-example synthetic projections; not an exact paper-table reproduction'}
 Path(a.output).with_suffix('.summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
if __name__=='__main__':main()
