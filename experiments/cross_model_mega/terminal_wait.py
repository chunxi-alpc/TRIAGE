"""Wait for the controller's terminal marker; do not query GPUs or running metrics."""
import argparse,json,os,time
from pathlib import Path
from .fit_controlled import OUT,dump

def reconcile():
    state=json.loads((OUT/'pipeline_status.json').read_text());actual={}
    for model in ('llama31','gemma2','gemma3','medgemma'):
        out=OUT/model
        done=out/'shard0_done.json'
        if not done.exists():
            done=Path('/home/gcx/aster_cross_model_mega_20261111')/model/'shard0/done.json'
        if not done.exists():continue
        r=json.loads(done.read_text());gpu=r.get('gpu');actual[model]=dict(gpu=gpu,state=r.get('state'),seconds=r.get('seconds'))
        if gpu is None:continue
        p=out/'pipeline_result.json';info=json.loads(p.read_text()) if p.exists() else {}
        info.update(model=model,gpu=gpu,state=r.get('state'),execution_metadata_source=str(done))
        dump(p,info)
        p=out/'config.json';cfg=json.loads(p.read_text());cfg.update(implementation_git_sha='1ae4f03',execution_gpu=gpu)
        dump(p,cfg)
    for r in state.get('results',[]):
        if r.get('model') in actual:r['gpu']=actual[r['model']]['gpu']
    state['actual_execution']=actual;dump(OUT/'pipeline_status.json',state)
    dump(OUT/'TERMINAL_READY.json',dict(state=state['state'],actual_execution=actual,
      summary_exit_code=state.get('summary_exit_code'),utc=time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime())))
    return state
def main():
    path=OUT/'pipeline_status.json'
    while True:
        if path.exists():
            state=json.loads(path.read_text())
            if state.get('state') in ('complete','failed'):
                print(json.dumps(reconcile(),ensure_ascii=False),flush=True)
                return
        time.sleep(60)
if __name__=='__main__':main()
