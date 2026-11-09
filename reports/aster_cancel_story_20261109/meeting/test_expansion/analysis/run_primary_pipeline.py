"""Prioritize complete frozen-method test results, then all labeled task splits."""
import argparse,gzip,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
BASE=Path('/home/gcx/aster_next_20261109')
PY=Path('/home/gcx/gcx261014/bin/python')
PUB=Path('/home/gcx/aster-real-completion-publish-20261109')
REPORT='reports/aster_cancel_story_20261109'

def save(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)

def command(args,cwd=None):
    subprocess.run([str(a) for a in args],cwd=cwd,check=True)

def main(plan_path):
    plan=json.loads(plan_path.read_text());out=Path(plan['results_root']);out.mkdir(parents=True,exist_ok=True)
    state_path=BASE/'primary_pipeline_status.json'
    def status(phase,**kwargs):
        value=dict(phase=phase,utc=time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime()),gpus=plan['gpus'],**kwargs)
        save(state_path,value);print(json.dumps(value),flush=True)
    def publish(stage):
        status('publishing',stage=stage['name'])
        command(['git','fetch','origin','aster-cr-20261108'],cwd=PUB)
        command(['git','merge','--ff-only','origin/aster-cr-20261108'],cwd=PUB)
        official=out/'mega_official_test/summary/metrics.json'
        if official.exists():
            command([PY,BASE/'analysis/update_documents.py','--metrics',official])
        command([PY,BASE/'analysis/update_primary_documents.py','--manifest',plan_path,'--results-root',out])
        packet=PUB/REPORT/'meeting/primary_full';packet.mkdir(parents=True,exist_ok=True)
        shutil.copy2(plan_path,packet/'primary_plan.json')
        shutil.copy2(BASE/'frozen_methods.json',packet/'frozen_methods.json')
        code=packet/'code';code.mkdir(exist_ok=True)
        for rel in ['analysis/run_primary_pipeline.py','inference/run_primary.py','inference/frozen.py','analysis/summarize_paths.py','inventory/prepare_all.py','inventory/prepare_external.py']:
            path=BASE/rel
            if path.exists():shutil.copy2(path,code/path.name)
        for item in plan['stages']:
            stage_out=out/item['name']
            if not (stage_out/'summary/metrics.json').exists():continue
            dest=packet/item['name'];dest.mkdir(exist_ok=True)
            for name in ['metrics.json','summary.csv']:shutil.copy2(stage_out/'summary'/name,dest/name)
            for rank in range(len(plan['gpus'])):
                path=stage_out/f'shard{rank}/results.jsonl'
                if path.exists():
                    with path.open('rb') as inp,gzip.open(dest/f'path_results_shard{rank}.jsonl.gz','wb',compresslevel=6) as target:
                        shutil.copyfileobj(inp,target,1024*1024)
        paths=[REPORT+'/paper_story_latest.zh.md',REPORT+'/README.md',REPORT+'/meeting/meeting_brief.zh.md',REPORT+'/meeting/test_expansion',REPORT+'/meeting/primary_full']
        command(['git','add','--',*paths],cwd=PUB)
        diff=subprocess.run(['git','diff','--cached','--quiet'],cwd=PUB)
        if diff.returncode==1:command(['git','commit','-m','Evaluate frozen main methods: '+stage['name']],cwd=PUB)
        elif diff.returncode:raise RuntimeError('Unable to inspect staged results')
        command(['git','push','origin','HEAD:aster-cr-20261108'],cwd=PUB)
        return subprocess.run(['git','rev-parse','--short','HEAD'],cwd=PUB,capture_output=True,text=True,check=True).stdout.strip()
    try:
        os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
        completed=[];commits={};incomplete={};publication_errors={}
        for stage in plan['stages']:
            manifest=Path(stage['cases']);stage_out=out/stage['name'];stage_out.mkdir(parents=True,exist_ok=True)
            if not manifest.exists():raise RuntimeError('Prepared dataset manifest missing: '+str(manifest))
            status('running',stage=stage['name'],expected=stage['expected'],completed_stages=completed)
            processes=[];logs=[]
            for rank,gpu in enumerate(plan['gpus']):
                log=(stage_out/f'worker{rank}.log').open('a',buffering=1);logs.append(log)
                args=[str(PY),str(BASE/'inference/run_primary.py'),'--cases',str(manifest),'--out-root',str(stage_out),'--methods',*plan['methods'],'--gpu',str(gpu),'--shard',str(rank),'--shards',str(len(plan['gpus'])),'--reuse-root',str(BASE/'results')]
                for previous in completed:args.extend(['--reuse-root',str(out/previous)])
                processes.append(subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,cwd=BASE))
            save(stage_out/'worker_pids.json',{str(rank):dict(pid=p.pid,gpu=plan['gpus'][rank]) for rank,p in enumerate(processes)})
            codes=[p.wait() for p in processes]
            for log in logs:log.close()
            if any(codes):raise RuntimeError('Worker failure in '+stage['name']+': '+str(codes)+'; saved records can be resumed')
            status('summarizing',stage=stage['name'],worker_exit_codes=codes,completed_stages=completed)
            result_paths=[stage_out/f'shard{rank}/results.jsonl' for rank in range(len(plan['gpus']))]
            command([PY,BASE/'analysis/summarize_paths.py','--cases',manifest,'--results',*result_paths,'--out',stage_out/'summary'])
            metrics=json.loads((stage_out/'summary/metrics.json').read_text());overall=metrics['groups']['all']
            if set(overall['methods'])!=set(plan['methods']):raise RuntimeError('Saved methods do not match requested fixed methods')
            if overall['case_count']!=stage['expected']:raise RuntimeError('Prepared case count differs from recorded plan')
            missing={name:{key:item[key] for key in ['evaluated','missing','failed']} for name,item in overall['methods'].items() if item['missing'] or item['failed']}
            incomplete[stage['name']]=missing
            completed.append(stage['name'])
            try:commits[stage['name']]=publish(stage)
            except Exception as exc:
                publication_errors[stage['name']]=str(exc);traceback.print_exc()
                status('publication_deferred',stage=stage['name'],error=str(exc),completed_stages=completed)
            save(stage_out/'complete.json',dict(state='complete_with_unavailable' if missing else 'complete',commit=commits.get(stage['name']),publication_error=publication_errors.get(stage['name']),incomplete_methods=missing))
        status('computed_with_publication_errors' if publication_errors else ('complete_with_unavailable' if any(incomplete.values()) else 'complete'),completed_stages=completed,commits=commits,incomplete_methods=incomplete,publication_errors=publication_errors,missing_data=plan.get('missing_data',[]))
    except Exception as exc:
        status('error',error=str(exc));traceback.print_exc();raise
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);args=parser.parse_args();main(args.plan)
