"""Launch four independent case workers once; wait for completion without polling."""
import json,os,subprocess,sys,time
from pathlib import Path
from .data import BASE,REV,read,write
from .engine import METHODS

def main():
 run=BASE;gpus=[0,1,5,6];start=time.time()
 manifest=run/'official_test_launch.json'
 if manifest.exists():raise FileExistsError('launch already registered; resume workers from saved commands instead')
 used=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free','--format=csv,noheader,nounits'],text=True)
 observed={int(line.split(',')[0]):[int(x.strip()) for x in line.split(',')[1:]] for line in used.strip().splitlines()}
 if any(observed[g][0]>1024 or observed[g][1]<70000 for g in gpus):raise RuntimeError('an authorized GPU became occupied; no workers started')
 refs=[json.loads((run/f'native_balanced_reference{i}'/'status.json').read_text()) for i in [0,1]]
 if not all(r.get('all_reference_generations_identical') for r in refs):raise RuntimeError('native numerical equivalence not passed')
 commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
 cfg={'status':'launching','physical_gpus':gpus,'logical_workers':list(range(4)),'model_revision':REV,'execution_code_commit':commit,
  'cases':len(read(run/'cases/official_test.jsonl')),'methods':METHODS,'expected_units':len(read(run/'cases/official_test.jsonl'))*len(METHODS),
  'native_query_chunk_size':2048,'native_balanced':True,'tensor_batch_size':1,'official_Qwen_judge':True,
  'method_freeze':json.loads((run/'frozen.json').read_text()),'preflight_memory_mib':{str(g):observed[g] for g in gpus},
  'resource_estimate':'four independent 80GB cards, typically 25-50GB per card; 2-4 days estimated, context-dependent',
  'completion_policy':'wait once; no polling or persistent monitoring; failures preserved; missing units only on resume'}
 with manifest.open('x') as f:json.dump(cfg,f,ensure_ascii=False,indent=2)
 children=[];handles=[]
 for shard,gpu in enumerate(gpus):
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false')
  log=(run/f'official_test{shard}.log').open('x');handles.append(log)
  cmd=[sys.executable,'-m','experiments.aster_manuscript_clinical_20261030.worker','--run',str(run),'--phase','official_test','--shard',str(shard),'--dynamic','--official-judge','--native-query-chunks','2048','--native-balanced','--resume']
  proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT);children.append(proc)
  (run/f'official_test{shard}.pid').write_text(str(proc.pid)+'\n')
 cfg['status']='running';cfg['worker_pids']=[p.pid for p in children];write(manifest,cfg)
 codes=[p.wait() for p in children]
 for h in handles:h.close()
 cfg['worker_exit_codes']=codes;cfg['elapsed_seconds']=time.time()-start
 if any(codes):cfg['status']='interrupted';cfg['reason']='one or more workers exited unsuccessfully; all saved method units retained'
 else:
  try:
   from .report import summarize
   summarize(run,'official_test');cfg['status']='complete'
  except Exception as e:cfg.update(status='incomplete',reason=str(e))
 write(manifest,cfg)
 if cfg['status']!='complete':sys.exit(1)
if __name__=='__main__':main()
