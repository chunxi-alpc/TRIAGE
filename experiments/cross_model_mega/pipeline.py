"""Two GPU queues with automatic summaries; checkpoints are resumable."""
import os,subprocess,sys,time,json,traceback
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from .fit_controlled import REPO,OUT,dump
WORK=Path('/home/gcx/aster_cross_model_mega_20261111')
def queue(gpu,models):
    records=[]
    for model in models:
        out=OUT/model;out.mkdir(parents=True,exist_ok=True)
        log=(out/'run.log').open('a',buffering=1)
        env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        if not (out/'controlled_complete.json').exists():
            result=subprocess.run([sys.executable,'-m','experiments.cross_model_mega.fit_controlled','--model',model],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                records.append(dict(model=model,state='failed',phase='controlled',exit_code=result.returncode));log.close();continue
        result=subprocess.run([sys.executable,'-m','experiments.cross_model_mega.run_paths','--model',model,'--gpu',str(gpu)],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
        records.append(dict(model=model,state='complete' if result.returncode==0 else 'failed',exit_code=result.returncode,gpu=gpu))
        dump(out/'pipeline_result.json',records[-1]);log.close()
    return records
def main():
    OUT.mkdir(parents=True,exist_ok=True);WORK.mkdir(exist_ok=True)
    dump(OUT/'execution_plan.json',dict(gpus=[0,1,5,6],queues={'0':['llama31'],'1':['gemma2'],'5':['gemma3'],'6':['medgemma']},
      official_cases_per_model=7458,gold_nodes_per_model=45627,controlled_dev_paths=420,controlled_test_paths=806,
      expected_peak_gib={'8b_9b':[20,30],'27b':[55,70]},estimated_hours_total=[5,10],
      selection_frozen_before_official=True,missing_only=True,methods=['raw','bias_only','aster_cr','random_trace']))
    started=time.time();all_records=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(queue,0,['llama31']),pool.submit(queue,1,['gemma2']),pool.submit(queue,5,['gemma3']),pool.submit(queue,6,['medgemma'])]
        for f in as_completed(futures):
            all_records+=f.result();dump(OUT/'pipeline_status.json',dict(state='running',results=all_records,seconds=time.time()-started))
    result=subprocess.run([sys.executable,'-m','experiments.cross_model_mega.summarize'],cwd=REPO)
    dump(OUT/'pipeline_status.json',dict(state='complete' if all(r['state']=='complete' for r in all_records) and result.returncode==0 else 'failed',
      results=all_records,summary_exit_code=result.returncode,seconds=time.time()-started))
if __name__=='__main__':main()
