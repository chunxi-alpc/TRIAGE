
"""Append fixed-prompt development pair metrics; do not change its calibration."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from pathlib import Path
import json,time,csv,argparse
from collections import defaultdict
import numpy as np
BASE=Path('/home/gcx/aster_next_20261109')
OUT=BASE/'diagnosis';RESULTS=BASE/'prompt_results'
OLD=Path('/home/gcx/aster_cancel_story_20261109/meeting/real_completion')
ROOT=Path('/home/gcx/aster_cr_repair_v2_run_20261108')
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()] if Path(p).exists() else []
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def ci(a):
 a=np.asarray(a);a=a[np.isfinite(a)]
 return np.quantile(a,[.025,.975]).tolist() if len(a) else None
def run(wait=False):
 if wait:
  while not (RESULTS/'dev_pairs.done.json').exists() or not (OUT/'alignment_fit_calibration.json').exists():
   if (RESULTS/'dev_pairs.failed.json').exists():raise RuntimeError('dev pair scoring failed')
   time.sleep(10)
 if not (RESULTS/'dev_pairs.done.json').exists():raise RuntimeError('dev pair scoring has not completed')
 views=[v for v in rows(ROOT/'data/views.jsonl') if v['split']=='dev']
 pairs=[p for p in rows(ROOT/'data/pairs.jsonl') if p['split']=='dev']
 scores={}
 for r in rows(RESULTS/'dev_pairs.jsonl'):
  if r.get('status')=='ok':scores[r['view_id']]=float(r['ell'])
 missing=[v['view_id'] for v in views if v['view_id'] not in scores]
 if missing:raise RuntimeError('Missing development view scores: '+str(missing[:5]))
 ids=[v['view_id'] for v in views];ix={v:i for i,v in enumerate(ids)}
 a=np.load(OLD/'coordinates.npz');oldix={v:i for i,v in enumerate(a['ids'].tolist())}
 pos=np.array([oldix[v] for v in ids]);raw=a['ell'][pos].astype(float)
 prompt=np.array([scores[v] for v in ids]);b=json.loads((OUT/'alignment_fit_calibration.json').read_text())['b']
 oldw=np.load(OLD/'../evidence_preserving/weights.npz')
 linear=raw+a['r16'][pos]@oldw['bounded_0.1']+float(oldw['bounded_0.1_b'])
 methods=dict(raw=raw,old_linear=linear,prompt_zero=prompt,prompt_fit_ce=prompt+b)
 sourceids=sorted({v['source_id'] for v in views});sx={s:i for i,s in enumerate(sourceids)}
 boot=np.random.default_rng(11).multinomial(len(sourceids),np.full(len(sourceids),1/len(sourceids)),size=2000)
 ap=[p for p in pairs if p['kind'] in ('A','P')]
 ai=np.array([ix[p['left']] for p in ap]);aj=np.array([ix[p['right']] for p in ap]);ady=raw[aj]-raw[ai]
 apss=np.array([sx[p['source_id']] for p in ap])
 ag=np.column_stack([np.bincount(apss,minlength=len(sx)),np.bincount(apss,weights=ady**2,minlength=len(sx))])
 z=boot@ag
 with np.errstate(divide='ignore',invalid='ignore'):denboot=np.sqrt(z[:,1]/z[:,0])
 den=float(np.sqrt(np.mean(ady**2)))
 result={};replicates={};pairrows=[]
 for name,s in methods.items():
  groups={}
  for group,kinds in [('A',('A',)),('P',('P',)),('AP',('A','P')),('C',('C',)),('F',('F',)),('CF',('C','F')),('F0',('F0',))]:
   pp=[p for p in pairs if p['kind'] in kinds];li=np.array([ix[p['left']] for p in pp]);ri=np.array([ix[p['right']] for p in pp])
   yl=np.array([p['left_label'] for p in pp]);yr=np.array([p['right_label'] for p in pp]);ss=np.array([sx[p['source_id']] for p in pp])
   rawdy=raw[ri]-raw[li];dy=s[ri]-s[li];dc=dy-rawdy;direction=yr-yl
   co=((s[li]>0)==yl)&((s[ri]>0)==yr);deg=(direction*rawdy>0)&(direction*dy<=0)
   values=np.column_stack([np.ones(len(pp)),rawdy,rawdy**2,dy,dy**2,dc**2,direction*rawdy,direction*dy,co,deg])
   ag=np.vstack([np.bincount(ss,weights=values[:,j],minlength=len(sx)) for j in range(values.shape[1])]).T;z=boot@ag
   with np.errstate(divide='ignore',invalid='ignore'):
    drms=np.sqrt(z[:,4]/z[:,0]);crms=np.sqrt(z[:,5]/z[:,0]);vr=1-(z[:,4]/z[:,0]-(z[:,3]/z[:,0])**2)/(z[:,2]/z[:,0]-(z[:,1]/z[:,0])**2)
    er=z[:,7]/z[:,6];pa=z[:,8]/z[:,0];rat=crms/denboot
   reps=dict(difference_rms=drms,correction_rms=crms,correction_rms_over_ap_rms=rat,paired_accuracy=pa)
   r=dict(n=len(pp),sources=len(set(ss)),raw_difference_rms=float(np.sqrt(np.mean(rawdy**2))),difference_rms=float(np.sqrt(np.mean(dy**2))),
    correction_rms=float(np.sqrt(np.mean(dc**2))),correction_rms_over_ap_rms=float(np.sqrt(np.mean(dc**2))/den),
    paired_correct=int(co.sum()),paired_accuracy=float(co.mean()),response_direction_degraded=int(deg.sum()),
    response_direction_improved=int(np.sum((direction*rawdy<=0)&(direction*dy>0))))
   if group in ('A','P','AP','F0'):r['variance_reduction']=1-float(np.var(dy)/np.var(rawdy));reps['variance_reduction']=vr
   else:r['mean_evidence_retention']=float(np.mean(direction*dy)/np.mean(direction*rawdy));reps['mean_evidence_retention']=er
   for k,v in reps.items():r[k+'_ci']=ci(v)
   replicates[name,group]=reps;groups[group]=r
   if group not in ('AP','CF'):
    for j,p in enumerate(pp):pairrows.append(dict(method=name,pair_id=p['pair_id'],kind=p['kind'],source_id=p['source_id'],
     left_score=float(s[li[j]]),right_score=float(s[ri[j]]),raw_difference=float(rawdy[j]),difference=float(dy[j]),correction_difference=float(dc[j]),paired_correct=bool(co[j])))
  result[name]=groups
 comparisons={}
 for name in ('prompt_zero','prompt_fit_ce'):
  comparisons[name]={}
  for reference in ('raw','old_linear'):
   comparisons[name][reference]={}
   for group in result[name]:
    comparisons[name][reference][group]={k:dict(difference=result[name][group][k]-result[reference][group][k],ci=ci(v-replicates[reference,group][k])) for k,v in replicates[name,group].items()}
 summary=dict(scope='development only; one fixed prompt candidate, no old/official test read or tuning',
  methods=result,comparisons=comparisons,fit_ce_b=b,view_count=len(views),pair_count=len(pairs),source_count=len(sourceids),
  uncertainty='2000 paired source-cluster bootstrap replicates seed11; undefined sparse-group draws excluded',
  constraint_note='This is an evidence-alignment prompt, not a learned constrained editor. It has no original per-fit-C/F hard-budget guarantee. All evidence-response changes are measured outcomes.',
  no_reselection_or_calibration_change=True)
 dump(OUT/'alignment_pair_metrics.json',summary)
 with (OUT/'alignment_pair_results.csv').open('w',newline='',encoding='utf-8') as h:
  writer=csv.DictWriter(h,fieldnames=list(pairrows[0]));writer.writeheader();writer.writerows(pairrows)
 if wait:
  while not (OUT/'alignment_summary.json').exists():time.sleep(5)
 if (OUT/'alignment_summary.json').exists():
  full=json.loads((OUT/'alignment_summary.json').read_text());full['development_pair_metrics']=summary
  dump(OUT/'alignment_summary.json',full)
 dump(OUT/'alignment_pairs_evaluation.done.json',dict(state='complete',views=len(views),pairs=len(pairs)))
 print(json.dumps(dict(state='complete',views=len(views),AP=result['prompt_fit_ce']['AP'],CF=result['prompt_fit_ce']['CF'])))
 return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');run(wait=p.parse_args().wait)

