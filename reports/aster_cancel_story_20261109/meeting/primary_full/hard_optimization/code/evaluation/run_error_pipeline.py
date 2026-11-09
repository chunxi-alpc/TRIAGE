"""Run only fixed original errors; reuse canonical cache, fill newly reached nodes lazily."""
from pathlib import Path
import subprocess,json,sys,time,traceback
ROOT=Path('/home/gcx/aster_hard_optimization_20261109');OLD=Path('/home/gcx/aster_next_20261109');EV=ROOT/'evaluation';PY=Path('/home/gcx/gcx261014/bin/python')
def save(obj):(EV/'status.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def main():
 plan=json.loads((EV/'plan.json').read_text());completed=[]
 try:
  for stage in plan['stages']:
   out=Path(plan['results_root'])/stage['name'];out.mkdir(parents=True,exist_ok=True)
   save(dict(phase='running',stage=stage['name'],expected=stage['expected'],completed=completed))
   processes=[];logs=[]
   for rank,gpu in enumerate(plan['gpus']):
    log=(out/f'worker{rank}.log').open('a',buffering=1);logs.append(log)
    cmd=[PY,ROOT/'run_hard_errors.py','--cases',stage['cases'],'--out-root',out,'--manifest',plan['manifest'],'--methods',*plan['methods'],'--gpu',gpu,'--shard',rank,'--shards',len(plan['gpus']),'--reuse-root',OLD/'results','--reuse-root',OLD/'primary_results'/stage['name']]
    processes.append(subprocess.Popen([str(x) for x in cmd],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT))
   (out/'worker_pids.json').write_text(json.dumps({str(i):dict(pid=p.pid,gpu=plan['gpus'][i]) for i,p in enumerate(processes)},indent=2)+'\n')
   codes=[p.wait() for p in processes]
   for log in logs:log.close()
   if any(codes):raise RuntimeError('Worker exit codes: '+str(codes)+'; records can resume.')
   save(dict(phase='summarizing',stage=stage['name'],completed=completed))
   cmd=[PY,OLD/'analysis/summarize_paths.py','--cases',stage['cases'],'--results',*[out/f'shard{i}/results.jsonl' for i in range(len(plan['gpus']))],'--out',out/'summary']
   subprocess.run([str(x) for x in cmd],check=True)
   completed.append(stage['name'])
  save(dict(phase='complete',completed=completed,scope='Only fixed old_linear errors; no new full-population accuracy or newly wrong counts.'))
 except Exception as exc:save(dict(phase='error',error=str(exc),completed=completed));traceback.print_exc();raise
if __name__=='__main__':main()
