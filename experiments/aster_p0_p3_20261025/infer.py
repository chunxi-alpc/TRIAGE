"""Fresh GPU inference; old predictions are never counted as new model calls."""
import argparse,hashlib,json,os,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'aster_e0_e8_20261024'))
from engine import Engine
from improve_branch import judge,judge_payload,condition_key
from continue_run import key,safe,atomic
import torch
BASE=Path('/home/gcx/aster_e0_e8_20261024/private')
ROOT=Path('/home/gcx/aster_p0_p3_20261025')
def read(p):
 return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--name',choices=['qwen','llama'],required=True);ap.add_argument('--phase',choices=['mismatch','test'],required=True);a=ap.parse_args()
 assert os.environ['CUDA_VISIBLE_DEVICES'] in ('0','1')
 out=ROOT/a.phase/a.name;out.mkdir(parents=True,exist_ok=True)
 rows=read(BASE/'cases.jsonl');rows=[c for c in rows if (c['split']=='dev' and c['seed']==0 and c['variant']=='NC') if a.phase=='mismatch'] if a.phase=='mismatch' else [c for c in rows if c['split']=='test']
 rows.sort(key=key)
 model='/home/gcx/TRIAGE/models/modelscope/'+('Qwen3-8B' if a.name=='qwen' else 'Meta-Llama-3.1-8B-Instruct')
 manifest={'model':model,'phase':a.phase,'data_sha256':sha(BASE/'cases.jsonl'),'code_sha256':sha(Path(__file__)),'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'rows':len(rows),'provenance':'fresh_gpu_forward; exact-input condition memoization explicitly recorded'}
 if a.phase=='test':
  lock=HERE/'results/frozen.json';assert lock.exists();manifest['frozen_sha256']=sha(lock)
 mp=out/'manifest.json'
 if mp.exists():assert json.loads(mp.read_text())==manifest
 else:atomic(mp,manifest)
 done={r['key']:r for r in read(out/'predictions.jsonl')};cache={r['condition_key']:r for r in read(out/'condition_cache.jsonl')}
 start=time.time();e=Engine(model);calls=0
 bytree={t:sorted([r for r in rows if r['tree']==t],key=lambda c:c['id']) for t in {r['tree'] for r in rows}} if a.phase=='mismatch' else {}
 with (out/'predictions.jsonl').open('a') as f,(out/'condition_cache.jsonl').open('a') as cf:
  for i,c in enumerate(rows):
   if key(c) in done:continue
   payload=judge_payload(c);donor=None
   if a.phase=='mismatch':
    group=bytree[c['tree']];eligible=[r for r in group if r['id']!=c['id'] and r['patient']!=c['patient']]
    if not eligible:raise ValueError('No distinct within-tree mismatch donor')
    donor=eligible[int(hashlib.sha256(c['id'].encode()).hexdigest(),16)%len(eligible)]
    payload['patient']=donor['patient']
   ck=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest();reused=ck in cache
   if not reused:
    t=time.time();p,assess=judge(e,payload);calls+=len(payload['conditions'])
    rec={'condition_key':ck,'p':p,'assessments':assess,'seconds':time.time()-t,'provenance':'fresh_gpu_forward'}
    cache[ck]=rec;cf.write(json.dumps(rec)+'\n');cf.flush()
   rec=cache[ck];r={k:c[k] for k in ['id','tree','seed','variant','split']};r.update(key=key(c),p=rec['p'],assessments=rec['assessments'],condition_key=ck,condition_reused_exact_input=reused)
   if donor:
    cc=safe(c);cc['patient']=donor['patient'];r.update(donor_id=donor['id'],blind=e.branch(cc,blind=True));calls+=1
   else:
    pred,scores=e.answer(safe(c));blind=e.branch(safe(c),blind=True);calls+=2
    r.update(none_prediction=pred,none_scores=scores,blind=blind,provenance='fresh_gpu_forward')
   f.write(json.dumps(r)+'\n');f.flush();done[key(c)]=r
   if (i+1)%10==0:
    atomic(out/'status.json',{'completed':len(done),'total':len(rows),'complete':False,'elapsed_seconds':time.time()-start,'new_call_units':calls,'peak_GiB':torch.cuda.max_memory_allocated()/2**30,'updated_unix':time.time()})
 atomic(out/'status.json',{'completed':len(done),'total':len(rows),'complete':len(done)==len(rows),'elapsed_seconds':time.time()-start,'new_call_units':calls,'peak_GiB':torch.cuda.max_memory_allocated()/2**30,'updated_unix':time.time()})
if __name__=='__main__':
 try:main()
 except BaseException:
  traceback.print_exc();raise
