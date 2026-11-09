
import os
os.environ["OPENBLAS_NUM_THREADS"]="4"
os.environ["OMP_NUM_THREADS"]="4"
from pathlib import Path
import json,sys,time
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import numpy as np
OUT=Path(__file__).resolve().parent
ROOT=Path('/home/gcx/aster_cr_repair_v2_run_20261108')
sys.path.insert(0,'/home/gcx/TRIAGE-worktrees/aster-cr-20261108')
from experiments.aster_cr.paths import execute
def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def get_path_features():
 saved=OUT/'dev_path_features.npz';metap=OUT/'dev_path_metadata.json'
 if saved.exists() and metap.exists():return dict(np.load(saved)),json.loads(metap.read_text())
 byid={}
 for p in sorted((ROOT/'qwen/paths').glob('features_dev_*.jsonl')):
  for r in rows(p):
   if r.get('status')=='ok':byid[r['id']]=r
 records=list(byid.values())
 s=dict(np.load(ROOT/'qwen/supports_r16.npz'))
 b=dict(np.load(OUT/'feature_bases.npz'))
 oldb=dict(np.load('/home/gcx/aster_cancel_story_20261109/meeting/real_completion/bases.npz'))
 raw=np.empty((len(records),len(s['mu'])),dtype='float32')
 byfile=defaultdict(list)
 for i,r in enumerate(records):byfile[r['feature']].append(i)
 def read(item):
  file,pos=item
  with np.load(file) as a:
   h=a['heads'];value=h.reshape(-1,h.shape[-1])[s['heads']].reshape(-1).astype('float32')
  return pos,value
 with ThreadPoolExecutor(max_workers=8) as pool:
  for pos,value in pool.map(read,byfile.items()):raw[pos]=value
 raw=(raw-s['mu'])/s['scale']
 N=raw@s['N'];S=raw@s['S'];r32=np.concatenate([N,S],axis=1)
 a={'ids':np.array([r['id'] for r in records]),'ell':np.array([r['ell'] for r in records]),'r16':N,'r32':r32,'U':raw@oldb['U']}
 r8=dict(np.load(ROOT/'qwen/supports_r8.npz'));a['r8']=raw@r8['N']
 AP=raw@b['AP'];en=raw@b['evidence_null']
 for dim in (64,128,256):a['AP'+str(dim)]=AP[:,:dim]
 for dim in (64,128):a['evidence_null'+str(dim)]=en[:,:dim]
 inp=(r32-b['rff_input_mean'])/b['rff_input_scale']
 for sigma in (1.,2.):a['RFF256_s'+str(sigma)]=np.sqrt(2/256)*np.cos(inp@b['rff_W']/sigma+b['rff_phase'])
 np.savez_compressed(saved,**a);dump(metap,dict(records=records))
 return a,dict(records=records)
def score(config,features):
 model=np.load(config['model_file']);z=(features[config['family']]-model['mu'])/model['scale']
 return features['ell']+(z@model['w'])*config.get('gain',1.)+config['b']
def main():
 start=time.time();a,meta=get_path_features();records=meta['records'];recidx=defaultdict(dict)
 for i,r in enumerate(records):recidx[r['case_id']][r['gid']]=i
 cases=[c for c in rows(ROOT/'data/path_cases.jsonl') if c['split']=='dev']
 cfgs=json.loads((OUT/'calibrated_top.json').read_text())
 result={};allrows=[]
 for r in cfgs:
  cfg=r['config'];ss=score(cfg,a);rr=[]
  for case in cases:
   ns={gid:float(ss[i]) for gid,i in recidx[case['case_id']].items()}
   rr.append(dict(case_id=case['case_id'],source_id=case['source_id'],**execute(case,ns)))
  count=sum(r['correct'] for r in rr)
  result[cfg['name']]=dict(n=len(rr),correct=count,accuracy=count/len(rr),config=cfg,dev=r['metrics']['dev'])
  allrows.extend(dict(method=cfg['name'],**r) for r in rr)
 with (OUT/'dev_path_results.jsonl').open('w') as f:
  for r in allrows:f.write(json.dumps(r)+'\n')
 dump(OUT/'dev_path_summary.json',result)
 dump(OUT/'path_done.json',dict(state='complete',candidates=len(cfgs),cases=len(cases),path_nodes=len(records),seconds=time.time()-start))
 print(json.dumps({k:dict(path=v['correct'],vr=v['dev']['AP']['variance_reduction'],cf_ratio=v['dev']['CF']['correction_rms_over_ap_rms']) for k,v in result.items()}))
if __name__=='__main__':main()

