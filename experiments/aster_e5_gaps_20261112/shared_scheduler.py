"""Finite shared queue for the authorized physical GPUs 0, 1, 4, 5, 6."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
GPUS=(0,1,4,5,6)
TERMINAL={'complete','failed','blocked','cancelled'}
def read(p):return json.loads(Path(p).read_text())
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def alive(pid):
    try:return Path('/proc',str(pid),'stat').read_text().split(') ',1)[1].split()[0]!='Z'
    except (FileNotFoundError,ProcessLookupError):return False
def normalize(raw,owner):
    j=dict(raw)
    j['id']=j.get('id',j.get('job_id'))
    j['command']=j.get('command',j.get('argv',j.get('cmd')))
    j['dependencies']=j.get('dependencies',j.get('depends_on',[]))
    j['owner']=owner;j.setdefault('gpu_count',1)
    j.setdefault('priority',100);j.setdefault('estimated_vram_gib',50 if j['gpu_count'] else 0)
    j.setdefault('env',{});j.setdefault('cwd','/home/gcx')
    if 'eligible_gpus' in j:j.setdefault('allowed_gpus',j['eligible_gpus'])
    if 'gpu' in j and 'assigned_gpu' not in j:j['assigned_gpu']=j['gpu']
    if not j.get('id'):raise ValueError('Missing job ID')
    if j['gpu_count'] not in (0,1):raise ValueError('Jobs must be CPU or single-GPU units: '+j['id'])
    return j
def load_jobs(paths):
    jobs={}
    for path in paths:
        if not path.exists():continue
        m=read(path);owner=m.get('owner',path.stem)
        for raw in m.get('jobs',[]):
            j=normalize(raw,owner);j.setdefault('state','pending')
            if j['id'] in jobs:raise ValueError('Duplicate job ID '+j['id'])
            jobs[j['id']]=j
        for raw in m.get('running',[]):
            j=normalize(raw,owner);j['state']='running';j['external']=True
            if j.get('assigned_gpu') is None:j['gpu_count']=0
            if j['id'] in jobs:jobs[j['id']].update(j)
            else:jobs[j['id']]=j
    for j in jobs.values():
        for dep in j['dependencies']:
            if dep not in jobs:raise ValueError('Unknown dependency '+dep+' for '+j['id'])
    done=set()
    while len(done)<len(jobs):
        ready={i for i,j in jobs.items() if i not in done and set(j['dependencies'])<=done}
        if not ready:raise ValueError('Cyclic job dependencies')
        done.update(ready)
    return jobs
def occupied_gpus():
    result=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    mapping=subprocess.run(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    index={r.split(',')[1].strip():int(r.split(',')[0]) for r in mapping.stdout.splitlines() if ',' in r}
    return {index[r.split(',')[0].strip()] for r in result.stdout.splitlines() if ',' in r and r.split(',')[0].strip() in index}
def run(args):
    jobs=load_jobs(args.manifest);root=args.output;root.mkdir(parents=True,exist_ok=True)
    if args.validate_only:
        print(json.dumps(dict(valid=True,jobs=len(jobs),gpu_jobs=sum(j['gpu_count'] for j in jobs.values()),adopted=sum(j.get('external',False) for j in jobs.values()))));return
    import fcntl
    instance=(root/'scheduler.lock').open('a')
    fcntl.flock(instance,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'completion.json').exists():raise RuntimeError('This queue already ended.')
    if (root/'status.json').exists():
        old=read(root/'status.json').get('jobs',{})
        for i,prior in old.items():
            if i in jobs and prior['state'] in TERMINAL|{'running'}:
                jobs[i].update(prior)
                if prior['state']=='running':jobs[i]['external']=True
    write(root/'launch.json',dict(pid=os.getpid(),gpus=list(GPUS),manifests=list(map(str,args.manifest)),started_at=time.strftime('%Y-%m-%d %H:%M:%S'),poll_seconds=args.poll_seconds))
    processes={};started=time.time()
    for j in jobs.values():
        marker=j.get('completion_marker')
        if j['state']=='pending' and marker and Path(marker).exists():j['state']='complete';j['reused_completed_output']=True
    while True:
        fresh=load_jobs(args.manifest)
        for i,j in fresh.items():
            if i not in jobs:jobs[i]=j
        missing_manifests=[str(path) for path in args.manifest if not path.exists()]
        changed=False
        for i,j in jobs.items():
            if j['state']!='running':continue
            if i in processes:
                rc=processes[i].poll()
                if rc is None:continue
                ok=rc==0
                if ok and j.get('completion_marker'):ok=Path(j['completion_marker']).exists()
                j.update(returncode=rc,state='complete' if ok else 'failed')
                del processes[i]
            elif not alive(j['pid']):
                # Adopted tasks provide an existing scientific completion artifact.
                marker=j.get('completion_marker')
                ok=bool(marker and Path(marker).exists())
                j.update(state='complete' if ok else 'failed',adopted_exit_verified_by_marker=ok)
            else:continue
            j['finished_at']=time.strftime('%Y-%m-%d %H:%M:%S');changed=True
        for j in jobs.values():
            if j['state']!='pending':continue
            deps=[jobs[d]['state'] for d in j['dependencies']]
            if not j.get('run_on_failure') and any(x in ('failed','blocked','cancelled') for x in deps):
                j.update(state='blocked',reason='A prerequisite failed or was cancelled; no automatic rerun.');changed=True
        ready=[]
        for j in jobs.values():
            if j['state']!='pending':continue
            deps=[jobs[d]['state'] for d in j['dependencies']]
            if all(x in TERMINAL for x in deps) if j.get('run_on_failure') else all(x=='complete' for x in deps):ready.append(j)
        ready.sort(key=lambda j:(j['priority'],list(jobs).index(j['id'])))
        used={j['assigned_gpu'] for j in jobs.values() if j['state']=='running' and j.get('gpu_count') and j.get('assigned_gpu') is not None}
        free=[]
        if any(j['gpu_count'] for j in ready):
            free=[g for g in GPUS if g not in used and g not in occupied_gpus()]
        cpu_active=sum(j['state']=='running' and not j['gpu_count'] for j in jobs.values())
        for j in ready:
            if j['gpu_count']:
                allowed=j.get('allowed_gpus',list(GPUS));candidates=[g for g in free if g in allowed]
                if not candidates:continue
                gpu=candidates[0];free.remove(gpu)
            else:
                if cpu_active>=2:continue
                gpu=None;cpu_active+=1
            env=dict(os.environ,**{str(k):str(v) for k,v in j['env'].items()})
            env['CUDA_VISIBLE_DEVICES']='' if gpu is None else str(gpu)
            env.setdefault('OMP_NUM_THREADS','4')
            logs=root/'logs';logs.mkdir(exist_ok=True)
            logpath=logs/(j['id']+'.log')
            try:
                with logpath.open('a') as log:
                    proc=subprocess.Popen(j['command'],cwd=j['cwd'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                processes[j['id']]=proc
                j.update(state='running',external=False,pid=proc.pid,assigned_gpu=gpu,log=str(logpath),started_at=time.strftime('%Y-%m-%d %H:%M:%S'))
            except Exception as exc:j.update(state='failed',error=str(exc))
            changed=True
        counts={state:sum(j['state']==state for j in jobs.values()) for state in ('pending','running','complete','failed','blocked','cancelled')}
        status=dict(state='running' if counts['running'] or counts['pending'] else 'complete',pid=os.getpid(),allowed_gpus=list(GPUS),updated_at=time.strftime('%Y-%m-%d %H:%M:%S'),counts=counts,jobs=jobs,continuous_model_monitoring=False,awaiting_manifests=missing_manifests)
        write(root/'status.json',status)
        if not counts['running'] and not counts['pending'] and not missing_manifests:
            status.update(state=('completed_with_cancellations' if counts['cancelled'] else 'complete') if not counts['failed'] and not counts['blocked'] else 'completed_with_failures',all_jobs_succeeded=not counts['failed'] and not counts['blocked'] and not counts['cancelled'],active_scope_succeeded=not counts['failed'] and not counts['blocked'],wall_seconds=time.time()-started)
            write(root/'status.json',status);write(root/'completion.json',status)
            write(root/'gpu_release.json',dict(complete=True,released_gpus=list(GPUS),no_gpu_jobs_inflight=True,all_jobs_succeeded=status['all_jobs_succeeded'],released_at=status['updated_at']))
            return
        time.sleep(args.poll_seconds)
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--poll-seconds',type=int,default=15)
    p.add_argument('--validate-only',action='store_true')
    args=p.parse_args()
    try:run(args)
    except Exception as exc:
        import traceback
        write(args.output/'scheduler_error.json',dict(error=str(exc),traceback=traceback.format_exc()))
        raise
if __name__=='__main__':main()
