
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import json
from pathlib import Path
import numpy as np
from collections import defaultdict
OUT=Path(__file__).resolve().parent
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def ci(z):
 z=np.asarray(z);z=z[np.isfinite(z)]
 return np.quantile(z,[.025,.975]).tolist() if len(z) else None
def main():
 m=json.loads((OUT/'metrics.json').read_text());meta=json.loads((OUT/'metadata.json').read_text())
 a=np.load(OUT/'evaluated_scores.npz');ix={v:i for i,v in enumerate(a['ids'].tolist())}
 vs=[v for v in meta['views'] if v['split']=='test'];ps=[p for p in meta['pairs'] if p['split']=='test']
 sources=sorted({v['source_id'] for v in vs});sx={s:i for i,s in enumerate(sources)}
 boot=np.random.default_rng(11).multinomial(len(sources),np.full(len(sources),1/len(sources)),size=2000)
 def agg(values,ss):
  values=np.asarray(values,dtype=float)
  if values.ndim==1:values=values[:,None]
  out=np.zeros((len(sources),values.shape[1]))
  for i,s in enumerate(ss):out[sx[s]]+=values[i]
  return boot@out
 draws={};comp={}
 for group,kinds in [('A',('A',)),('P',('P',)),('AP',('A','P')),('C',('C',)),('F',('F',)),('CF',('C','F')),('F0',('F0',))]:
  pp=[p for p in ps if p['kind'] in kinds];li=np.array([ix[p['left']] for p in pp]);ri=np.array([ix[p['right']] for p in pp])
  yl=np.array([p['left_label'] for p in pp]);yr=np.array([p['right_label'] for p in pp]);ss=[p['source_id'] for p in pp]
  raw=a['raw'][ri]-a['raw'][li];direction=yr-yl
  for name in m['methods']:
   s=a[name];dy=s[ri]-s[li];dc=dy-raw;cor=((s[li]>0)==yl)&((s[ri]>0)==yr)
   z=agg(np.column_stack([np.ones(len(pp)),raw,raw**2,dy,dy**2,dc**2,direction*raw,direction*dy,cor]),ss)
   with np.errstate(divide='ignore',invalid='ignore'):
    d=dict(difference_rms=np.sqrt(z[:,4]/z[:,0]),correction_rms=np.sqrt(z[:,5]/z[:,0]),paired_accuracy=z[:,8]/z[:,0])
    if group in ('A','P','AP','F0'):d['variance_reduction']=1-(z[:,4]/z[:,0]-(z[:,3]/z[:,0])**2)/(z[:,2]/z[:,0]-(z[:,1]/z[:,0])**2)
    else:d['mean_evidence_retention']=z[:,7]/z[:,6]
   draws[name,group]=d
  for name in m['methods']:
   comp.setdefault(name,{})[group]={k:dict(difference=m['methods'][name]['groups'][group][k]-m['methods']['linear_output']['groups'][group][k],ci=ci(z-draws['linear_output',group][k])) for k,z in draws[name,group].items()}
 # Natural and path outcomes share each source draw across methods.
 path=defaultdict(dict)
 for line in (OUT/'path_results.jsonl').read_text().splitlines():
  r=json.loads(line);path[r['method']][r['case_id']]=r
 for name in m['methods']:
  vv=[v for v in vs if v['variant']=='full' and v['original']]
  ss=[v['source_id'] for v in vv];cor=[(a[name][ix[v['view_id']]]>0)==bool(v['label']) for v in vv]
  z=agg(np.column_stack([np.ones(len(vv)),cor]),ss)
  draws[name,'natural']={'accuracy':z[:,1]/z[:,0]}
  rr=list(path[name].values());z=agg(np.array([[1,int(r['correct'])] for r in rr]),[r['source_id'] for r in rr])
  draws[name,'path']={'accuracy':z[:,1]/z[:,0]}
 for name in m['methods']:
  for group in ('natural','path'):
   point=m['methods'][name]['path'] if group=='path' else m['methods'][name]['groups']['natural']
   ref=m['methods']['linear_output']['path'] if group=='path' else m['methods']['linear_output']['groups']['natural']
   comp[name][group]={'accuracy':dict(difference=point['accuracy']-ref['accuracy'],ci=ci(draws[name,group]['accuracy']-draws['linear_output',group]['accuracy']))}
 summary={}
 for kind in ('mlp_output','linear_representation','mlp_representation'):
  ns=[kind+'_seed'+str(s) for s in (11,22,33)];summary[kind]={}
  for group in ('A','P','AP','C','F','CF','F0','natural','path'):
   result={}
   for metric in draws[ns[0],group]:
    points=[]
    for n in ns:
     r=m['methods'][n]['path'] if group=='path' else m['methods'][n]['groups'][group]
     points.append(r[metric])
    mean_draw=np.mean([draws[n,group][metric] for n in ns],axis=0)
    ref=m['methods']['linear_output']['path'] if group=='path' else m['methods']['linear_output']['groups'][group]
    result[metric]=dict(mean=float(np.mean(points)),seed_sample_sd=float(np.std(points,ddof=1)),
     source_ci=ci(mean_draw),difference_vs_linear_output=float(np.mean(points)-ref[metric]),
     difference_vs_linear_output_source_ci=ci(mean_draw-draws['linear_output',group][metric]))
   summary[kind][group]=result
 dump(OUT/'comparisons.json',dict(reference='linear_output',replicates=2000,grouping='source_id',methods=comp))
 dump(OUT/'seed_summary.json',dict(explanation='Mean and sample SD over seeds 11/22/33; CIs resample sources jointly across fixed fitted seeds, not generation seeds or new training runs',methods=summary))
if __name__=='__main__':main()

