"""One explicitly ordered experiment stage; completed prediction cells are retained."""
import argparse,time,json,traceback
from pathlib import Path
import torch
from fullscale_engine import Engine
from fullscale_run import select_strength,evaluate,report
from fullscale_data import write
p=argparse.ArgumentParser(__doc__);p.add_argument('--run',required=True);p.add_argument('--phase',choices=['select','confirm','official_test','med'],required=True);p.add_argument('--worker',type=int,default=0);p.add_argument('--attempt',type=int,default=0);a=p.parse_args();a.started=time.time();a.stage='full'
root=Path(a.run)/'serial';root.mkdir(exist_ok=True)
out=root/a.phase/(('attempt_'+str(a.attempt)) if a.phase=='select' else ('worker_'+str(a.worker)));out.mkdir(parents=True,exist_ok=True)
status=out/'status.json'
try:
 write(status,{'status':'running','phase':a.phase,'worker':a.worker})
 e=Engine('/home/gcx/TRIAGE/models/modelscope/Qwen3-8B')
 if a.phase=='select':
  artifact=torch.load(Path(a.run)/'parallel_fit/directions.pt',weights_only=False,map_location='cpu')
  artifact['selection']=select_strength(e,a,out,artifact)
  temp=root/'frozen.tmp';torch.save(artifact,temp);temp.replace(root/'frozen.pt')
  write(root/'frozen.json',{'selection':artifact['selection'],'training_pairs':artifact['training_pairs'],'official_test_used':False})
 else:
  artifact=torch.load(root/'frozen.pt',weights_only=False,map_location='cpu')
  if a.phase=='med':evaluate(e,a,out,artifact,'med',['heldout'],tag='med_heldout')
  else:evaluate(e,a,out,artifact,'mega',['confirm' if a.phase=='confirm' else 'official_test'],tag='mega_'+a.phase)
  report(out)
 write(status,{'status':'complete','phase':a.phase,'worker':a.worker,'seconds':time.time()-a.started,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30})
except Exception as ex:
 write(status,{'status':'failed','phase':a.phase,'worker':a.worker,'reason':str(ex),'traceback':traceback.format_exc()});raise
