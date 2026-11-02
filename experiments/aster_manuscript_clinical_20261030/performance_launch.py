"""Resume missing units with validated exact execution optimizations on 0/1/5/6."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from .data import BASE,write

def main():
 p=argparse.ArgumentParser();p.add_argument('--graphs',action='store_true');a=p.parse_args();run=BASE;perf=run/'performance01';gpus=[0,1,5,6];start=time.time()
 benchmark=json.loads((perf/('graph_benchmark.json' if a.graphs else 'exact_benchmark.json')).read_text())
 if benchmark.get('status')!='passed':raise RuntimeError('real-weight execution equivalence not passed')
 old=json.loads((run/'official_test_launch.json').read_text())
 if (perf/'optimized_launch.json').exists():raise FileExistsError('performance launch already registered')
 used=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free','--format=csv,noheader,nounits'],text=True)
 observed={int(s.split(',')[0]):[int(v) for v in s.split(',')[1:]] for s in used.strip().splitlines()}
 if any(observed[g][0]>1024 or observed[g][1]<70000 for g in gpus):raise RuntimeError('authorized GPU not free; no process started')
 commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();tags=['perf1_'+str(i) for i in range(4)]
 cfg=dict(old,status='launching',logical_workers=tags,physical_gpus=gpus,execution_code_commit=commit,previous_execution_code_commit=old['execution_code_commit'],exact_execution_optimizations=True,decode_graphs=a.graphs,native_uses_reference_execution=True,guide_prefix_retention='original_single_slot',execution_profile='performance01',worker_exit_codes=None,previous_units_preserved=True,freeze_modified=False)
 cfg.pop('reason',None);cfg.pop('elapsed_seconds',None)
 write(perf/'launch_before_optimization.json',old);children=[];handles=[]
 for shard,gpu in enumerate(gpus):
  log=(run/f'official_test_{tags[shard]}.log').open('x');handles.append(log)
  cmd=[sys.executable,'-u','-m','experiments.aster_manuscript_clinical_20261030.worker','--run',str(run),'--phase','official_test','--shard',str(shard),'--worker-tag',tags[shard],'--dynamic','--official-judge','--native-query-chunks','2048','--native-balanced','--exact-execution','--resume']
  if a.graphs:cmd.append('--decode-graphs')
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false')
  child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT);children.append(child)
  (run/f'official_test_{tags[shard]}.pid').write_text(str(child.pid)+'\n')
 cfg.update(status='running',worker_pids=[p.pid for p in children]);write(run/'official_test_launch.json',cfg);write(perf/'optimized_launch.json',cfg)
 codes=[p.wait() for p in children]
 for f in handles:f.close()
 cfg.update(worker_exit_codes=codes,elapsed_seconds=time.time()-start)
 if any(codes):cfg.update(status='interrupted',reason='worker exit; saved units retained, resume only missing units')
 else:
  try:
   from .report import summarize
   summarize(run,'official_test');cfg['status']='complete'
  except Exception as e:cfg.update(status='incomplete',reason=str(e))
 write(run/'official_test_launch.json',cfg);write(perf/'optimized_launch.json',cfg)
 if cfg['status']!='complete':sys.exit(1)
if __name__=='__main__':main()
