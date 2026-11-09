
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
import json,sys,time,importlib.util
from pathlib import Path
import numpy as np,torch
from scipy.special import expit
from scipy.optimize import brentq
OUT=Path('/home/gcx/aster_optimization_20261109/selection')
BASE=OUT.parent;SH=BASE/'shared';OLD=Path('/home/gcx/aster_cancel_story_20261109/meeting/real_completion')
OUT.mkdir(exist_ok=True);(OUT/'predictions').mkdir(exist_ok=True)
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def module(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
path_tool=module('path_tool',BASE/'diagnosis/path_tools.py')
neural=module('neural_models',BASE/'neural/search.py')
scorecond=module('scorecond_models',BASE/'score_conditioned/search.py')
def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def main():
 torch.set_num_threads(2);start=time.time()
 a=dict(np.load(SH/'features.npz'));allids=a['ids'].tolist();aix={v:i for i,v in enumerate(allids)}
 meta=json.loads((SH/'metadata.json').read_text())
 views=[v for v in meta['views'] if v['split'] in ('fit','dev')]
 dp=[r for r in meta['paths'] if r['split']=='dev']
 vid=[v['view_id'] for v in views];pids=['path|'+r['id'] for r in dp];ids=vid+pids;ix={v:i for i,v in enumerate(ids)}
 pos=np.array([aix[v] for v in ids]);features={k:v[pos] for k,v in a.items() if k!='ids'};features['ids']=np.array(ids)
 nr=len(vid);ell=features['ell']
 cases=[c for c in rows(Path('/home/gcx/aster_cr_repair_v2_run_20261108/data/path_cases.jsonl')) if c['split']=='dev']
 nat={sp:[v for v in views if v['split']==sp and v['variant']=='full' and v['original']] for sp in ('fit','dev')}
 ni={sp:np.array([ix[v['view_id']] for v in vv]) for sp,vv in nat.items()}
 ny={sp:np.array([v['label'] for v in vv]) for sp,vv in nat.items()}
 ps=[p for p in meta['pairs'] if p['split']=='dev']
 group={}
 for k,kinds in [('A',('A',)),('P',('P',)),('AP',('A','P')),('C',('C',)),('F',('F',)),('CF',('C','F')),('F0',('F0',))]:
  pp=[p for p in ps if p['kind'] in kinds]
  group[k]=(np.array([ix[p['left']] for p in pp]),np.array([ix[p['right']] for p in pp]),np.array([p['left_label'] for p in pp]),np.array([p['right_label'] for p in pp]))
 pl,pr,_,_=group['AP'];rawAP=ell[pr]-ell[pl];den=float(np.sqrt(np.mean(rawAP**2)))
 cl,cr,cyl,cyr=group['CF'];rawCF=ell[cr]-ell[cl];direction=cyr-cyl;rawresponse=float(np.mean(direction*rawCF))
 # Add AP coordinates for every fit/dev view and path; all fitted on fit.
 lin=np.load(BASE/'linear/feature_coordinates.npz');oldids=np.load(BASE/'linear/ids.npz')['ids'].tolist();oix={v:i for i,v in enumerate(oldids)}
 lp=dict(np.load(BASE/'linear/path_coordinates.npz'));lix={v:i for i,v in enumerate(lp['ids'].tolist())}
 ap=np.concatenate([lin['AP256'][[oix[v] for v in vid]],lp['AP256'][[lix[v] for v in pids]]])
 linfeat={'AP256':ap}
 def fit_b(c):
  z=ell[ni['fit']]+c[ni['fit']];fn=lambda b:float(np.mean(expit(z+b)-ny['fit']))
  return float(brentq(fn,-150,150))
 def metric(c,b):
  ss=ell+c+b;g={}
  for k,(l,r,yl,yr) in group.items():
   raw=ell[r]-ell[l];dc=c[r]-c[l];dy=raw+dc
   co=((ss[l]>0)==yl)&((ss[r]>0)==yr)
   z=dict(n=len(l),difference_rms=float(np.sqrt(np.mean(dy**2))),correction_rms=float(np.sqrt(np.mean(dc**2))),paired_correct=int(co.sum()),paired_accuracy=float(co.mean()))
   z['correction_rms_over_ap_rms']=z['correction_rms']/den
   if k in ('A','P','AP','F0'):z['variance_reduction']=1-float(np.var(dy)/np.var(raw))
   else:
    di=yr-yl;z['mean_evidence_retention']=float(np.mean(di*dy)/np.mean(di*raw))
    z['response_direction_degraded']=int(np.sum((di*raw>0)&(di*dy<=0)))
   g[k]=z
  g['natural']=dict(correct=int(np.sum((ss[ni['dev']]>0)==ny['dev'])),n=len(ni['dev']))
  return g
 def optimize(c,bref):
  nodes={(r['case_id'],r['gid']):float(ell[nr+j]+c[nr+j]) for j,r in enumerate(dp)}
  return path_tool.select_dev_bias(nodes,cases,ell[ni['dev']]+c[ni['dev']],ny['dev'],bref=bref,min_natural_correct=195)
 catalog=[]
 for base,kind in [(BASE/'neural','neural'),(BASE/'neural/extension','neural'),(BASE/'neural/allfit','neural'),(BASE/'neural/confirmation','neural'),(BASE/'score_conditioned/candidates','scorecond')]:
  if base.exists():
   for p in sorted(base.glob('*.pt')):
    # Ignore optimizer helpers or files that have no matching completed scores.
    if p.with_suffix('.npz').exists():catalog.append((kind,p))
 results=[]
 # Strong calibration control is selected by the same path objective.
 bz=fit_b(np.zeros(len(ids)));cal=optimize(np.zeros(len(ids)),bz)
 dump(OUT/'calibration_control.json',dict(b=cal['best_b'],selection=cal,dev=metric(np.zeros(len(ids)),cal['best_b'])))
 for j,(kind,p) in enumerate(catalog):
  name=kind+'_'+p.stem;cache=OUT/'predictions'/(name+'.npz')
  state=torch.load(p,map_location='cpu',weights_only=False)
  if cache.exists():pred=np.load(cache)['delta']
  else:
   if kind=='neural':
    cfg=state['config'];model=neural.Editor(len(state['mu']),cfg['width'],cfg['depth'])
    x=features[cfg['feature']]
   else:
    cfg=state['cfg'];model=scorecond.Editor(len(state['mu']),cfg['width'])
    x=scorecond.features(cfg['family'],features,linfeat)
   model.load_state_dict(state['state_dict']);model.eval()
   z=((x-state['mu'])/state['scale']).astype('float32')
   with torch.no_grad():pred=np.concatenate([model(torch.from_numpy(z[i:i+2048])).numpy() for i in range(0,len(z),2048)]).astype('float64')
   np.savez_compressed(cache,ids=np.array(ids),delta=pred)
  dcAP=pred[pr]-pred[pl];dcCF=pred[cr]-pred[cl]
  v=float(np.var(dcAP));optimal=max(0.,-float(np.mean((rawAP-rawAP.mean())*(dcAP-dcAP.mean())))/max(v,1e-12))
  cf_rms=float(np.sqrt(np.mean(dcCF**2)));er_slope=float(np.mean(direction*dcCF)/rawresponse)
  er_bound=.02/(-er_slope) if er_slope<0 else float('inf')
  for tier in (.05,.1,.2,.4):
   gain=min(optimal,tier*den/max(cf_rms,1e-12),er_bound)
   if gain<=0:continue
   c=pred*gain;bf=fit_b(c);op=optimize(c,bf)
   if not op['eligible']:continue
   b=op['best_b'];g=metric(c,b)
   results.append(dict(name=name,tier=tier,kind=kind,model_file=str(p),gain=gain,b=b,fit_ce_b=bf,path=op,dev=g,
    config=state.get('config',state.get('cfg')),seed=state.get('seed',11),steps=state.get('steps'),prediction_file=str(cache)))
  if j%20==0:dump(OUT/'status.json',dict(stage='path_selection',completed=j+1,total=len(catalog),seconds=time.time()-start))
 dump(OUT/'candidates.json',results)
 tiers={}
 for tier in (.05,.1,.2,.4):
  rr=[r for r in results if r['tier']==tier and r['path']['correct']>=384]
  byvr=sorted(rr,key=lambda r:(r['dev']['AP']['variance_reduction'],r['path']['correct']),reverse=True)[:5]
  bypath=sorted(rr,key=lambda r:(r['path']['correct'],r['dev']['AP']['variance_reduction']),reverse=True)[:5]
  tiers[str(tier)]=dict(best_vr=byvr,best_path=bypath)
 dump(OUT/'selection.json',dict(selection_split='dev',test_metrics_used=False,tiers=tiers))
 dump(OUT/'done.json',dict(state='complete',models=len(catalog),operating_points=len(results),seconds=time.time()-start))
 print(json.dumps({t:{k:[dict(name=x['name'],vr=x['dev']['AP']['variance_reduction'],path=x['path']['correct'],gain=x['gain']) for x in v[:2]] for k,v in z.items()} for t,z in tiers.items()}))
if __name__=='__main__':main()

