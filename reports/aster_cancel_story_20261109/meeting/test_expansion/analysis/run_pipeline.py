import json,os,subprocess,sys,time,traceback
from pathlib import Path
BASE=Path('/home/gcx/aster_next_20261109')
PUB=Path('/home/gcx/aster-real-completion-publish-20261109')
PY=Path('/home/gcx/gcx261014/bin/python')
REPORT='reports/aster_cancel_story_20261109'
status=BASE/'pipeline_status.json'

def save(phase,**kwargs):
    data={'phase':phase,'utc':time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime()),**kwargs}
    tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(status)
    print(json.dumps(data),flush=True)

def run(args,cwd=None):
    p=subprocess.run([str(a) for a in args],cwd=cwd)
    if p.returncode:raise RuntimeError('Command failed: '+str(args[0])+' '+str(args[1])+'; exit '+str(p.returncode))

def main():
    os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
    (BASE/'results').mkdir(exist_ok=True)
    dirty=subprocess.run(['git','status','--porcelain'],cwd=PUB,capture_output=True,text=True,check=True).stdout
    if dirty.strip():raise RuntimeError('Publishing checkout has existing edits; preserve them and stop before launch')
    save('running',gpus=[0,1],cases=7458)
    workers=[];logs=[]
    for shard in [0,1]:
        log=(BASE/'results'/('worker'+str(shard)+'.log')).open('a',buffering=1);logs.append(log)
        command=[str(PY),str(BASE/'inference/run.py'),'--gpu',str(shard),'--shard',str(shard),'--shards','2']
        workers.append(subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=BASE))
    (BASE/'worker_pids.json').write_text(json.dumps({str(i):p.pid for i,p in enumerate(workers)},indent=2))
    codes=[p.wait() for p in workers]
    for log in logs:log.close()
    if any(codes):raise RuntimeError('Worker failures: '+str(codes)+'; outputs are resumable')
    save('summarizing',worker_exit_codes=codes)
    output=BASE/'results/summary'
    run([PY,BASE/'analysis/summarize_paths.py','--cases',BASE/'inventory/path_cases.jsonl','--results',BASE/'results/shard0/results.jsonl',BASE/'results/shard1/results.jsonl','--out',output])
    metrics=json.loads((output/'metrics.json').read_text())
    expected_methods=set(json.loads((BASE/'frozen_methods.json').read_text())['comparison_names'])
    if set(metrics['groups']['all']['methods'])!=expected_methods:raise RuntimeError('Expanded method coverage does not match the frozen comparison')
    missing={k:{x:v for x,v in m.items() if x in ['evaluated','missing','failed']} for k,m in metrics['groups']['all']['methods'].items() if m['missing'] or m['failed']}
    save('publishing',incomplete_methods=missing)
    run(['git','fetch','origin','aster-cr-20261108'],cwd=PUB)
    run(['git','merge','--ff-only','origin/aster-cr-20261108'],cwd=PUB)
    run([PY,BASE/'analysis/update_documents.py','--metrics',output/'metrics.json'])
    paths=[REPORT+'/paper_story_latest.zh.md',REPORT+'/README.md',REPORT+'/meeting/meeting_brief.zh.md',REPORT+'/meeting/test_expansion']
    run(['git','add','--',*paths],cwd=PUB)
    staged=subprocess.run(['git','diff','--cached','--quiet'],cwd=PUB)
    if staged.returncode==1:run(['git','commit','-m','Evaluate frozen corrections on full MEGA-CDP official test'],cwd=PUB)
    elif staged.returncode:raise RuntimeError('Unable to inspect staged publishing changes')
    run(['git','push','origin','HEAD:aster-cr-20261108'],cwd=PUB)
    commit=subprocess.run(['git','rev-parse','--short','HEAD'],cwd=PUB,capture_output=True,text=True,check=True).stdout.strip()
    save('complete_with_unavailable' if missing else 'complete',commit=commit,incomplete_methods=missing)
if __name__=='__main__':
    try:main()
    except Exception as e:
        save('error',error=str(e));traceback.print_exc();sys.exit(1)
