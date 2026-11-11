"""Resume the two 27B models on two GPUs each, sharing all completed cached records."""
import os,subprocess,sys,time,json
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from .fit_controlled import REPO,OUT,dump
WORK=Path('/home/gcx/aster_cross_model_mega_20261111')
JOBS=[('gemma3',0,0),('gemma3',5,1),('medgemma',1,0),('medgemma',6,1)]
def worker(model,gpu,shard):
    out=OUT/model
    with (out/('run_gpu'+str(gpu)+'_shard'+str(shard)+'.log')).open('a',buffering=1) as log:
        r=subprocess.run([sys.executable,'-u','-m','experiments.cross_model_mega.run_paths',
          '--model',model,'--gpu',str(gpu),'--shard',str(shard),'--shards','2'],cwd=REPO,
          env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT)
    return dict(model=model,gpu=gpu,shard=shard,state='complete' if r.returncode==0 else 'failed',exit_code=r.returncode)
def main():
    started=time.time();completed=[]
    dump(OUT/'pipeline_status.json',dict(state='running',phase='remaining_27b_two_shards',gpus=[0,1,5,6],results=[]))
    plan=json.loads((OUT/'execution_plan.json').read_text())
    plan.update(gpus=[0,1,5,6],queues={'gemma3':{'0':0,'1':5},'medgemma':{'0':1,'1':6}},
       scheduling='Retain all completed model outputs; two disjoint [shard::2] case lists per 27B model, global cache reuse.',
       lock='per-model per-shard',model_inputs_weights_budgets_unchanged=True)
    dump(OUT/'execution_plan.json',plan)
    with ThreadPoolExecutor(max_workers=4) as pool:
        fs=[pool.submit(worker,*job) for job in JOBS]
        for f in as_completed(fs):
            completed.append(f.result())
            dump(OUT/'pipeline_status.json',dict(state='running',phase='remaining_27b_two_shards',gpus=[0,1,5,6],
              results=completed,seconds_since_reschedule=time.time()-started))
    result=subprocess.run([sys.executable,'-m','experiments.cross_model_mega.summarize'],cwd=REPO)
    actual={}
    for model in ('llama31','gemma2','gemma3','medgemma'):
        done=[json.loads(p.read_text()) for p in sorted((WORK/model).glob('shard*/done.json'))]
        gpus=sorted({r['gpu'] for r in done if 'gpu' in r});actual[model]=dict(gpus=gpus,shards=done)
        info=dict(model=model,state='complete' if all(r.get('state')=='complete' for r in done) else 'failed',
          gpus=gpus,metadata_source='Actual per-shard done.json')
        dump(OUT/model/'pipeline_result.json',info)
        cfg=json.loads((OUT/model/'config.json').read_text());cfg['execution_gpus']=gpus;cfg.pop('execution_gpu',None);dump(OUT/model/'config.json',cfg)
    state=dict(state='complete' if all(r['state']=='complete' for r in completed) and result.returncode==0 else 'failed',
      phase='remaining_27b_two_shards',results=completed,actual_execution=actual,summary_exit_code=result.returncode,
      seconds_since_reschedule=time.time()-started)
    dump(OUT/'pipeline_status.json',state)
    dump(OUT/'TERMINAL_READY.json',dict(state=state['state'],summary_exit_code=result.returncode,
      actual_execution=actual,utc=time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime())))
if __name__=='__main__':main()
