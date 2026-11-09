
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('OMP_NUM_THREADS','4')
from pathlib import Path
import json,time
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import numpy as np
from sklearn.utils.extmath import randomized_svd
OLD=Path('/home/gcx/aster_cancel_story_20261109/meeting/real_completion')
OUT=Path('/home/gcx/aster_optimization_20261109/shared')
ROOT=Path('/home/gcx/aster_cr_repair_v2_run_20261108')
OUT.mkdir(parents=True,exist_ok=True)
def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 start=time.time();a=dict(np.load(OLD/'coordinates.npz'));oldids=a.pop('ids').tolist();ell=a.pop('ell')
 meta=json.loads((OLD/'metadata.json').read_text());ix={v:i for i,v in enumerate(oldids)}
 fit=[ix[v['view_id']] for v in meta['views'] if v['split']=='fit' and v['variant']=='full' and v['original']]
 h=np.load(OLD/'residual.npy',mmap_mode='r')
 mu=h[fit].mean(0);sc=np.maximum(h[fit].std(0),1e-4)
 R=randomized_svd((h[fit]-mu)/sc,n_components=64,random_state=11)[2].T
 extra=[];paths=[]
 for split in ('fit','dev'):
  for file in sorted((ROOT/'qwen/paths').glob('features_'+split+'_*.jsonl')):
   for r in rows(file):
    if r.get('status')=='ok':paths.append(dict(split=split,**r));extra.append(r)
 ids=oldids+['path|'+r['id'] for r in extra];ix={v:i for i,v in enumerate(ids)}
 n=len(ids);nr=len(oldids)
 r16=np.empty((n,16),dtype=np.float32);r16[:nr]=a['r16']
 r32=np.empty((n,32),dtype=np.float32);r32[:nr]=a['r32']
 hp=np.empty((n,64),dtype=np.float32)
 for lo in range(0,nr,512):
  hi=min(nr,lo+512);hp[lo:hi]=((h[lo:hi]-mu)/sc)@R
 s=dict(np.load(ROOT/'qwen/supports_r16.npz'))
 oldfiles={}
 for file in sorted((ROOT/'qwen/scores').glob('*.jsonl')):
  for r in rows(file):
   if r.get('status')=='ok' and r['view_id'] in ix:oldfiles[r['feature']]=ix[r['view_id']]
 for r in meta['paths']:oldfiles[r['feature']]=ix['path|'+r['id']]
 pending=defaultdict(list)
 for j,r in enumerate(extra):
  i=nr+j;oi=oldfiles.get(r['feature'])
  if oi is not None:r16[i]=r16[oi];r32[i]=r32[oi];hp[i]=hp[oi]
  else:pending[r['feature']].append(i)
 def load(item):
  file,pos=item
  with np.load(file) as z:
   hh=z['heads'];raw=hh.reshape(-1,hh.shape[-1])[s['heads']].reshape(-1).astype(np.float32)
   hv=z['residual'].reshape(-1).astype(np.float32)
  raw=(raw-s['mu'])/s['scale']
  return pos,raw@s['N'],np.concatenate([raw@s['N'],raw@s['S']]),((hv-mu)/sc)@R
 with ThreadPoolExecutor(max_workers=8) as pool:
  for j,(pos,t,ts,p) in enumerate(pool.map(load,pending.items())):
   r16[pos]=t;r32[pos]=ts;hp[pos]=p
   if j%2000==0:dump(OUT/'status.json',dict(stage='path_cache',read=j,total=len(pending),seconds=time.time()-start))
 ell=np.concatenate([ell,np.array([r['ell'] for r in extra])])
 meta['paths']=[dict(split='test',**r) for r in meta['paths']]+paths
 np.savez_compressed(OUT/'features.npz',ids=np.array(ids),ell=ell,r16=r16,r32=r32,residual64=hp)
 np.savez_compressed(OUT/'residual_basis.npz',mu=mu,scale=sc,R=R)
 dump(OUT/'metadata.json',meta)
 dump(OUT/'status.json',dict(stage='complete',n=n,fitdev_path_records=len(extra),new_feature_files=len(pending),seconds=time.time()-start))
if __name__=='__main__':main()

