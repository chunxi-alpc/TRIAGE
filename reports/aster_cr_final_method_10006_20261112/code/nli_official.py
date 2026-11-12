"""Complete the already-requested official 5500 NLI evaluation; no fitting."""
import os,sys,time,json,pathlib,subprocess
import numpy as np
import run as r
O=r.ROOT
N=pathlib.Path('/home/gcx/aster_next_20261109/inventory')
def setup():
 st=r.Store();cases=[c for c in r.rows(N/'nli4ct_all_cases.jsonl') if c['dataset_split']=='test']
 pairs=[p for p in r.rows(N/'nli4ct_native_pairs.jsonl') if p['dataset_split']=='test']
 assert len(cases)==5500 and len(pairs)==5000
 db={json.loads(meta).get('case_id'):uid for uid,meta in st.base.inputs.conn.execute("SELECT uid,metadata FROM inputs WHERE metadata LIKE '%NLI4CT%'")}
 vv=[];jobs=[];mapping={}
 for c in cases:
  key=c['case_id'];base=db.get(key)
  if base is not None:
   assert st.input(base)['prompt']==c['prompt_text'],key
   mapping[key]=base
  else:jobs.append(dict(view_id=key,prompt=c['prompt_text']))
  original=c['native_id']==c['family']
  vv.append(dict(view_id=key,case_id=key,source_id=c['family_cluster_id'],family=c['family'],split='test',label=c['label'],original=original,variant='full' if original else 'native_variant',kind='original' if original else 'native_relation'))
 for p in pairs:
  p['source_id']='NLI4CT-2024:test:'+p['family'];p['kind']='native_relation';p['split']='test'
 assert sum(v['original'] for v in vv)==500
 r.lines(O/'configs/nli_official_views.jsonl',vv);r.lines(O/'configs/nli_official_pairs.jsonl',pairs);r.lines(O/'configs/nli_official_missing.jsonl',jobs);r.write(O/'configs/nli_official_mapping.json',mapping)
 r.write(O/'status/nli_official.json',dict(stage='prepared',reused=len(mapping),missing=len(jobs),total=5500,gpus=[0,1,5,6]))
def gpu(shard):
 import torch
 from experiments.aster_cr.engine import Engine
 torch.set_num_threads(4)
 jobs=r.rows(O/'configs/nli_official_missing.jsonl')[shard::4]
 dest=O/'cache/nli_official';dest.mkdir(exist_ok=True)
 record=dest/f'scores_{shard}.jsonl';done={v['view_id'] for v in r.rows(record)}
 eng=Engine('qwen');start=time.time()
 for i,v in enumerate(jobs):
  if v['view_id'] in done:continue
  context=eng.prepare(v['prompt']);out=eng.run(context,capture=True,grad=False)
  heads=out.pop('heads').numpy().astype(np.float16).reshape(1152,128);p=dest/(v['view_id'].replace(':','_')+'.npz')
  np.savez(p,heads=heads)
  r.worker.append(record,dict(out,view_id=v['view_id'],feature=str(p),status='ok',available_heads='all'))
  del context
  if i<3 or (i+1)%25==0:r.write(O/f'status/nli_gpu_{shard}.json',dict(completed=i+1,total=len(jobs),gpu=os.environ['CUDA_VISIBLE_DEVICES'],seconds=time.time()-start,peak_gib=torch.cuda.max_memory_allocated()/2**30))
 r.write(O/f'status/nli_gpu_{shard}.json',dict(complete=True,total=len(jobs),gpu=os.environ['CUDA_VISIBLE_DEVICES'],seconds=time.time()-start,peak_gib=torch.cuda.max_memory_allocated()/2**30))
class OfficialStore:
 def __init__(self):
  self.base=r.Store();self.mapping=r.read(O/'configs/nli_official_mapping.json')
  self.extra={v['view_id']:v for p in (O/'cache/nli_official').glob('scores_*.jsonl') for v in r.rows(p)}
 def ell(self,k):return self.base.ell(self.mapping[k]) if k in self.mapping else float(self.extra[k]['ell'])
 def get(self,k):
  if k in self.mapping:return self.base.get(self.mapping[k])
  z=np.load(self.extra[k]['feature']);return dict(heads=z['heads'].astype(np.float32),ell=self.ell(k))
def evaluate():
 import analysis as a
 a.STORE=OfficialStore();a.ITEMS={v['id']:v for v in r.rows(r.ev.TARGET/'medpic/items.jsonl')}
 r.ev.score_views=a.score;r.ev.decisions_for=a.decisions;r.ev.observations=a.observations
 spec=dict(views=r.rows(O/'configs/nli_official_views.jsonl'),pairs=r.rows(O/'configs/nli_official_pairs.jsonl'),native_task='nli4ct')
 a.evaluate('nli4ct_official5500',spec,a.full_models(),{'E4':dict(alpha=0.,beta=1.)})
 r.write(O/'status/nli_official.json',dict(stage='evaluated',total=5500,originals=500,preserving_pairs=4136,changing_pairs=864,calibration=False,gpus=[0,1,5,6]))
 subprocess.run([sys.executable,str(O/'code/finalize.py')],check=True)
if __name__=='__main__':
 try:
  if len(sys.argv)>1:gpu(int(sys.argv[1]))
  else:
   setup();procs=[]
   for i,g in enumerate([0,1,5,6]):
    log=(O/f'logs/nli_gpu_{g}.log').open('a');p=subprocess.Popen([sys.executable,'-u',__file__,str(i)],env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(g)),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT);procs.append((p,log))
   r.write(O/'status/nli_workers.json',dict(workers=[dict(pid=p.pid,gpu=g) for (p,l),g in zip(procs,[0,1,5,6])]))
   codes=[p.wait() for p,l in procs]
   for p,l in procs:l.close()
   if any(codes):raise RuntimeError(str(codes))
   evaluate()
 except Exception as exc:
  import traceback
  r.write(O/'status/nli_error.json',dict(error=str(exc),traceback=traceback.format_exc()));raise
