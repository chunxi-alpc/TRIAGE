
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['OMP_NUM_THREADS']='4'
import json,sys,time
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.special import expit
from scipy.optimize import brentq
OUT=Path(__file__).resolve().parent
BASE=OUT.parent;OLD=Path('/home/gcx/aster_cancel_story_20261109/meeting/real_completion')
sys.path.insert(0,str(BASE/'diagnosis'))
from path_tools import select_dev_bias
sys.path.insert(0,str(OUT))
from path_calibrate import stable_interval_bias
sys.path.insert(0,'/home/gcx/TRIAGE-worktrees/aster-cr-20261108')
from experiments.aster_cr.paths import execute
def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 start=time.time();dest=OUT/'ensemble_predictions';dest.mkdir(exist_ok=True)
 rootc=json.loads((BASE/'selection/candidates.json').read_text())
 sample=np.load(rootc[0]['prediction_file']);ids=sample['ids'].tolist();ix={v:i for i,v in enumerate(ids)}
 sh=np.load(BASE/'shared/features.npz');sidx={v:i for i,v in enumerate(sh['ids'].tolist())};ell=sh['ell'][[sidx[v] for v in ids]]
 meta=json.loads((BASE/'shared/metadata.json').read_text());views=meta['views']
 cases=[c for c in rows(Path('/home/gcx/aster_cr_repair_v2_run_20261108/data/path_cases.jsonl')) if c['split']=='dev']
 dp=[r for r in meta['paths'] if r['split']=='dev'];pathidx=np.array([ix['path|'+r['id']] for r in dp])
 pathkeys=[r['id'] for r in dp]
 nat={}
 for sp in ('fit','dev'):
  vv=[v for v in views if v['split']==sp and v['variant']=='full' and v['original']]
  nat[sp]=(np.array([ix[v['view_id']] for v in vv]),np.array([v['label'] for v in vv]))
 group={}
 for k,kinds in [('A',('A',)),('P',('P',)),('AP',('A','P')),('C',('C',)),('F',('F',)),('CF',('C','F')),('F0',('F0',))]:
  pp=[p for p in meta['pairs'] if p['split']=='dev' and p['kind'] in kinds]
  group[k]=(np.array([ix[p['left']] for p in pp]),np.array([ix[p['right']] for p in pp]),np.array([p['left_label'] for p in pp]),np.array([p['right_label'] for p in pp]))
 al,ar,_,_=group['AP'];rawap=ell[ar]-ell[al];den=float(np.sqrt(np.mean(rawap**2)))
 cl,cr,yl,yr=group['CF'];direction=yr-yl;rawresponse=float(np.mean(direction*(ell[cr]-ell[cl])))
 maxgold=max(len(c['gold_path']) for c in cases)
 gold_idx=np.zeros((len(cases),maxgold),int);gold_lab=np.zeros_like(gold_idx);gold_mask=np.zeros_like(gold_idx,dtype=bool)
 for j,c in enumerate(cases):
  for k,n in enumerate(c['gold_path']):
   gold_idx[j,k]=ix['path|'+c['case_id']+':'+n['gid']];gold_lab[j,k]=n['label'];gold_mask[j,k]=True
 def fit_b(c):
  nn,ny=nat['fit'];z=ell[nn]+c[nn]
  return float(brentq(lambda b:float(np.mean(expit(z+b)-ny)),-150,150))
 def metrics(c,b):
  ss=ell+c+b;g={}
  for k,(l,r,yl,yr) in group.items():
   raw=ell[r]-ell[l];dc=c[r]-c[l];dy=raw+dc;cor=((ss[l]>0)==yl)&((ss[r]>0)==yr)
   z=dict(n=len(l),difference_rms=float(np.sqrt(np.mean(dy**2))),correction_rms=float(np.sqrt(np.mean(dc**2))),paired_correct=int(cor.sum()),paired_accuracy=float(cor.mean()))
   z['correction_rms_over_ap_rms']=z['correction_rms']/den
   if k in ('A','P','AP','F0'):z['variance_reduction']=1-float(np.var(dy)/np.var(raw))
   else:
    di=yr-yl;z['mean_evidence_retention']=float(np.mean(di*dy)/np.mean(di*raw));z['response_direction_degraded']=int(np.sum((di*raw>0)&(di*dy<=0)))
   g[k]=z
  ni,ny=nat['dev'];g['natural']=dict(n=len(ni),correct=int(np.sum((ss[ni]>0)==ny)))
  return g
 # Input predictions are raw deltas; member-selected gain is carried in coefficients.
 cache={}
 def root_delta(r):
  if r['name'] not in cache:
   z=np.load(r['prediction_file'])
   if z['ids'].tolist()!=ids:raise RuntimeError('prediction ids differ')
   cache[r['name']]=z['delta'].astype('float64')
  return cache[r['name']]
 lp=dict(np.load(OUT/'path_coordinates.npz'));lidx={v:i for i,v in enumerate(lp['ids'].tolist())}
 oldids=np.load(OUT/'ids.npz')['ids'].tolist();oidx={v:i for i,v in enumerate(oldids)}
 linear=json.loads((OUT/'path_calibrated_selection.json').read_text())['tiers']
 linear_cache={}
 def linear_delta(r):
  cfg=r['config'];key=cfg['parent']
  if key not in linear_cache:
   model=np.load(cfg['model_file']);old=np.load(cfg['parent_score_file'])['delta']
   z=(lp[cfg['family']]-model['mu'])/model['scale'];pd=z@model['w']
   full=np.empty(len(ids),dtype='float64')
   for j,vid in enumerate(ids):
    full[j]=old[oidx[vid]] if vid in oidx else pd[lidx[vid]]
   linear_cache[key]=full
  return linear_cache[key]
 result=[]
 for tier in (.05,.1,.2,.4):
  pool=[r for r in rootc if abs(r['tier']-tier)<1e-9 and r['path'].get('eligible') and r['path']['correct']>=384 and r['dev']['CF']['mean_evidence_retention']>=.98-1e-8]
  pool.sort(key=lambda r:(r['dev']['AP']['variance_reduction'],r['path']['correct']),reverse=True)
  unique=[];seen=set()
  for r in pool:
   key=json.dumps(r['config'],sort_keys=True)
   if key not in seen:unique.append(r);seen.add(key)
  linearbest=linear[str(tier)]['highest_VR_preserve_path'][0]
  for K in (2,3,5,10):
   for hybrid in (False,True):
    members=[];deltas=[]
    n=K-1 if hybrid else K
    if len(unique)<n:continue
    for r in unique[:n]:
     coeff=r['gain']/K;deltas.append(root_delta(r)*coeff)
     members.append(dict(name=r['name'],kind=r['kind'],model_file=r['model_file'],prediction_file=r['prediction_file'],selected_gain=r['gain'],coefficient=coeff,config=r['config']))
    if hybrid:
     cfg=linearbest['config'];coeff=cfg['gain']/K;deltas.append(linear_delta(linearbest)*coeff)
     members.append(dict(name=cfg['name'],kind='linear',model_file=cfg['model_file'],selected_gain=cfg['gain'],coefficient=coeff,family=cfg['family'],config=cfg))
    raw=np.sum(deltas,axis=0)
    da=raw[ar]-raw[al];dc=raw[cr]-raw[cl]
    optimal=max(0.,-float(np.mean((rawap-rawap.mean())*(da-da.mean())))/max(float(np.var(da)),1e-12))
    cfrms=float(np.sqrt(np.mean(dc**2)));slope=float(np.mean(direction*dc)/rawresponse)
    erbound=.02/(-slope) if slope<0 else float('inf')
    gain=min(optimal,tier*den/max(cfrms,1e-12),erbound)
    final=raw*gain;bf=fit_b(final)
    scores=ell+final
    nn,ny=nat['dev']
    op=select_dev_bias(dict(zip(pathkeys,scores[pathidx])),cases,scores[nn],ny,bref=bf,min_natural_correct=195)
    op=stable_interval_bias(scores,scores[nn],ny,op,bf,gold_idx,gold_lab,gold_mask)
    if not op['eligible']:continue
    b=op['best_b'];g=metrics(final,b)
    # Execute autonomous paths to confirm the interval scorer after stable bias.
    bycase=defaultdict(dict)
    for j,r in enumerate(dp):bycase[r['case_id']][r['gid']]=float(scores[pathidx[j]]+b)
    correct=sum(execute(c,bycase[c['case_id']])['correct'] for c in cases)
    if correct!=op['correct']:raise RuntimeError((correct,op))
    name='ensemble_'+str(tier).replace('.','p')+'_K'+str(K)+('_hybrid' if hybrid else '_neural')
    file=dest/(name+'.npz')
    np.savez_compressed(file,ids=np.array(ids),delta=raw,final_delta=final,gain=gain,b=b)
    result.append(dict(name=name,tier=tier,kind='ensemble',member_count=K,hybrid=hybrid,members=members,coefficients=[v['coefficient'] for v in members],
      gain=gain,b=b,fit_ce_b=bf,path=op,dev=g,prediction_file=str(file),prediction_convention='delta is sum(member_raw_delta*coefficient); final_delta=delta*gain; score=ell+final_delta+b'))
 selection={}
 for tier in (.05,.1,.2,.4):
  rr=[r for r in result if r['tier']==tier and r['path']['correct']>=384]
  selection[str(tier)]=dict(best_vr=sorted(rr,key=lambda r:(r['dev']['AP']['variance_reduction'],r['path']['correct']),reverse=True)[:3],
    best_path=sorted(rr,key=lambda r:(r['path']['correct'],r['dev']['AP']['variance_reduction']),reverse=True)[:3])
 dump(OUT/'ensemble_candidates.json',result)
 dump(OUT/'ensemble_selection.json',dict(selection_split='dev',test_evaluated=False,source_catalog_rows=len(rootc),candidates=len(result),tiers=selection))
 dump(OUT/'ensemble_done.json',dict(state='complete',candidates=len(result),seconds=time.time()-start))
 print(json.dumps({tier:{k:[dict(name=r['name'],VR=r['dev']['AP']['variance_reduction'],ER=r['dev']['CF']['mean_evidence_retention'],path=r['path']['correct'],b=r['b']) for r in rr] for k,rr in val.items()} for tier,val in selection.items()}))
if __name__=='__main__':main()

