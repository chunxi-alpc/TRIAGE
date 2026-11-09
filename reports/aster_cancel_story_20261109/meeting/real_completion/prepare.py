
import os,json,time,sys
os.environ.setdefault("OPENBLAS_NUM_THREADS","2")
os.environ.setdefault("OMP_NUM_THREADS","2")
from pathlib import Path
import numpy as np
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from sklearn.utils.extmath import randomized_svd
ROOT=Path('/home/gcx/aster_cr_repair_v2_run_20261108')
OLD=Path('/home/gcx/aster_cancel_story_20261109/meeting/evidence_preserving')
OUT=Path('/home/gcx/aster_cancel_story_20261109/meeting/real_completion')
OUT.mkdir(exist_ok=True)
def rows(p):
 return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()] if Path(p).exists() else []
def dump(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 started=time.time()
 views=rows(ROOT/'data/views.jsonl'); pairs=rows(ROOT/'data/pairs.jsonl')
 scores={}
 for p in sorted((ROOT/'qwen/scores').glob('*.jsonl')):
  for r in rows(p):
   if r.get('status')=='ok':scores[r['view_id']]=r
 vs=[v for v in views if v['view_id'] in scores]
 rec=[dict(id=v['view_id'],ell=scores[v['view_id']]['ell'],feature=scores[v['view_id']]['feature']) for v in vs]
 paths={}
 for p in sorted((ROOT/'qwen/paths').glob('features_test_*.jsonl')):
  for r in rows(p):
   if r.get('status')=='ok':paths[r['id']]=r
 rec+= [dict(id='path|'+k,ell=r['ell'],feature=r['feature']) for k,r in sorted(paths.items())]
 ids=[r['id'] for r in rec];ix={v:i for i,v in enumerate(ids)}
 s=dict(np.load(ROOT/'qwen/supports_r16.npz'))
 dim=len(s['mu'])
 raw=np.lib.format.open_memmap(OUT/'raw_selected.npy',mode='w+',dtype='float32',shape=(len(rec),dim))
 # Only unique feature files are read; no model forwards.
 byfile={}
 for i,r in enumerate(rec): byfile.setdefault(r['feature'],[]).append(i)
 tasks=list(byfile.items())
 def load(item):
  f,positions=item
  with np.load(f) as a:
   heads=a['heads'];x=heads.reshape(-1,heads.shape[-1])[s['heads']].reshape(-1).astype('float32')
   h=a['residual'].reshape(-1).astype('float32')
  return positions,x,h
 hs=None
 with ThreadPoolExecutor(max_workers=8) as pool:
  for j,(positions,x,h) in enumerate(pool.map(load,tasks)):
   if hs is None:hs=np.lib.format.open_memmap(OUT/'residual.npy',mode='w+',dtype='float32',shape=(len(rec),len(h)))
   raw[positions]=x;hs[positions]=h
   if j%1000==0:dump(OUT/'status.json',dict(stage='reading_cache',files=j,total=len(tasks),seconds=time.time()-started))
 raw.flush();hs.flush()
 ell=np.array([r['ell'] for r in rec],dtype='float64')
 ps=[p for p in pairs if p['left'] in ix and p['right'] in ix]
 ap=[p for p in ps if p['split']=='fit' and p['kind'] in ('A','P')]
 cf=[p for p in ps if p['split']=='fit' and p['kind'] in ('C','F')]
 li=np.array([ix[p['left']] for p in ap]);ri=np.array([ix[p['right']] for p in ap])
 dx=(raw[ri]-raw[li])/s['scale']
 U=randomized_svd(dx,n_components=16,random_state=20261108)[2].T
 V=randomized_svd(hs[ri]-hs[li],n_components=16,random_state=11)[2].T
 R=np.linalg.qr(np.random.default_rng(11).normal(size=(dim,16)))[0].astype('float32')
 s8=dict(np.load(ROOT/'qwen/supports_r8.npz'))
 bases={'r16':s['N'],'r8':s8['N'],'U':U,'S':s['S'],'random':R,'r32':np.concatenate([s['N'],s['S']],axis=1)}
 coords={k:np.empty((len(rec),v.shape[1]),dtype='float64') for k,v in bases.items()}
 for lo in range(0,len(rec),512):
  x=(raw[lo:lo+512]-s['mu'])/s['scale']
  for k,b in bases.items():coords[k][lo:lo+512]=x@b
 np.savez_compressed(OUT/'coordinates.npz',ids=np.array(ids),ell=ell,**coords)
 np.savez_compressed(OUT/'bases.npz',V=V,**bases)
 keys=('view_id','source_id','group_id','variant','original','label','split','case_id','kind')
 meta=dict(views=[{k:v.get(k) for k in keys} for v in vs],pairs=ps,paths=list(paths.values()),
  coverage={sp:{k:dict(expected=sum(p['split']==sp and p['kind']==k for p in pairs),available=sum(p['split']==sp and p['kind']==k for p in ps)) for k in ('A','P','C','F','F0')} for sp in ('fit','dev','test')},
  view_coverage={sp:dict(expected=sum(v['split']==sp for v in views),available=sum(v['split']==sp for v in vs)) for sp in ('fit','dev','test')})
 dump(OUT/'metadata.json',meta)
 # Extract the real final norm and two vocabulary rows without instantiating the backbone.
 from safetensors import safe_open
 from transformers import AutoTokenizer
 model=Path('/home/gcx/TRIAGE/models/modelscope/Qwen3-8B')
 index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
 tok=AutoTokenizer.from_pretrained(model)
 label_ids=[tok.encode(x,add_special_tokens=False)[0] for x in ['\u662f','\u5426']]
 def tensor(name,indices=None):
  with safe_open(str(model/index[name]),framework='pt',device='cpu') as f:
   return (f.get_tensor(name) if indices is None else f.get_slice(name)[indices[0]:indices[0]+1]).float().numpy()
 norm=tensor('model.norm.weight')
 name='lm_head.weight' if 'lm_head.weight' in index else 'model.embed_tokens.weight'
 weights=np.concatenate([tensor(name,[i]) for i in label_ids])
 cfg=json.loads((model/'config.json').read_text())
 np.savez(OUT/'output_head.npz',norm=norm,weights=weights,eps=cfg['rms_norm_eps'],label_ids=label_ids)
 with np.load(OLD/'cached_coordinates.npz') as old:
  oi=[ix[v] for v in old['ids'].tolist()]
  error=float(np.max(np.abs(coords['r16'][oi]-old['t'])))
 dump(OUT/'preparation.json',dict(n=len(rec),unique_feature_files=len(tasks),views=meta['view_coverage'],
   coverage=meta['coverage'],fit_ap=len(ap),fit_cf=len(cf),old_trace_max_abs_difference=error,seconds=time.time()-started))
 dump(OUT/'status.json',dict(stage='prepared',seconds=time.time()-started))
if __name__=='__main__':main()

