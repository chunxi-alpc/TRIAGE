"""Faithfulness on frozen-method official MEGA controlled node pairs; CPU only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
import argparse,json,time
from pathlib import Path
import numpy as np
from summarize_nli_pairs import score_cached_nodes,pair_stats,source_sums,ci,write_csv,dump
BASE=Path('/home/gcx/aster_next_20261109')
GROUPS=[('A',('A',)),('P',('P',)),('AP',('A','P')),('C',('C',)),('F',('F',)),('CF',('C','F')),('F0',('F0',))]
def rows(path):
 with Path(path).open() as f:
  for line in f:
   if line.strip():yield json.loads(line)
def ratio(x,y):
 return float(x/y) if y!=0 and np.isfinite(x/y) else None

def summarize(args):
 started=time.time();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 # Prompt text is not required for cached scoring and is discarded while streaming.
 cases=[{k:c[k] for k in ('case_id','gid','label','source_id') if k in c} for c in rows(args.cases)]
 ix={c['case_id']:i for i,c in enumerate(cases)}
 pairs=[]
 for p in rows(args.pairs):
  p=dict(p);p['family']=p['source_id']
  if p['left'] not in ix or p['right'] not in ix:raise ValueError('Pair endpoint missing from prepared cases')
  if cases[ix[p['left']]]['label']!=p['left_label'] or cases[ix[p['right']]]['label']!=p['right_label']:
   raise ValueError('Pair labels differ from prepared endpoint labels')
  pairs.append(p)
 scores,names,coverage=score_cached_nodes(cases,args.nodes,args.manifest,args.methods)
 raw=scores['raw'];source_ids=sorted({p['source_id'] for p in pairs});sx={s:i for i,s in enumerate(source_ids)}
 draws=np.random.default_rng(args.seed).multinomial(len(sx),np.full(len(sx),1/len(sx)),size=args.iterations) if sx else np.empty((0,0))
 ap=[p for p in pairs if p['kind'] in ('A','P') and np.isfinite(raw[ix[p['left']]]) and np.isfinite(raw[ix[p['right']]])]
 apraw=np.array([raw[ix[p['right']]]-raw[ix[p['left']]] for p in ap]);apscale=float(np.sqrt(np.mean(apraw**2))) if len(apraw) else None
 groups={};flat=[];pair_rows=[];rep_cache={}
 for group,kinds in GROUPS:
  pp=[p for p in pairs if p['kind'] in kinds]
  item={'expected_pairs':len(pp),'methods':{}}
  if not pp:item['unavailable_reason']='No prepared source-grounded pairs of this kind; not inferred from natural path accuracy.'
  for name,score in scores.items():
   point,basic_reps=pair_stats(score,raw,pp,ix,draws,sx)
   rep_cache[group,name]=dict(basic_reps)
   valid=[p for p in pp if all(np.isfinite(v) for v in (score[ix[p['left']]],score[ix[p['right']]],raw[ix[p['left']]],raw[ix[p['right']]]))]
   if valid:
    li=np.array([ix[p['left']] for p in valid]);ri=np.array([ix[p['right']] for p in valid]);dy0=raw[ri]-raw[li];dy=score[ri]-score[li];dc=dy-dy0
    sources=np.array([sx[p['source_id']] for p in valid]);rawvar=float(np.var(dy0));variance=float(np.var(dy))
    stable=((score[li]>0)==(score[ri]>0)).astype(float)
    point.update(sources=len(set(p['source_id'] for p in valid)),raw_difference_variance=rawvar,difference_variance=variance,
     correction_abs_max=float(np.max(np.abs(dc))),correction_rms_over_ap_rms=ratio(float(np.sqrt(np.mean(dc**2))),apscale) if apscale else None,
     prediction_unchanged_fraction=float(stable.mean()))
    z=draws@source_sums(np.column_stack([np.ones(len(valid)),dy0,dy0**2,dy,dy**2,stable]),sources,len(sx))
    with np.errstate(divide='ignore',invalid='ignore'):
     v0=z[:,2]/z[:,0]-(z[:,1]/z[:,0])**2;v=z[:,4]/z[:,0]-(z[:,3]/z[:,0])**2
     bvr=1-v/v0;br2=1-(z[:,4]/z[:,0])/v0
    if group in ('A','P','AP','F0'):
     point.update(variance_reduction=1-ratio(variance,rawvar) if rawvar else None,variance_reduction_ci=ci(bvr),
      paired_prediction_r2=1-ratio(float(np.mean(dy**2)),rawvar) if rawvar else None,paired_prediction_r2_ci=ci(br2))
    if group in ('A','P','AP','F0'):
     rep_cache[group,name].update(variance_reduction=bvr,paired_prediction_r2=br2)
    if group=='F0':
     with np.errstate(divide='ignore',invalid='ignore'):stable_rep=z[:,5]/z[:,0]
     point['prediction_unchanged_fraction_ci']=ci(stable_rep)
    if group in ('C','F','CF'):
     point['mean_evidence_retention']=point.get('mean_signed_response_retention')
     point['mean_evidence_retention_ci']=point.get('mean_signed_response_retention_ci')
     point['response_direction_degraded']=point.get('gold_direction_degraded')
     point['exceeds_original_fit_budget_count']=int(np.sum(np.abs(dc)>.2923349027297149+1e-7))
    if group in ('A','P','C','F','F0'):
     for j,pair in enumerate(valid):
      pair_rows.append(dict(method=name,pair_id=pair['pair_id'],source_id=pair['source_id'],kind=pair['kind'],
       left_score=float(score[li[j]]),right_score=float(score[ri[j]]),raw_difference=float(dy0[j]),difference=float(dy[j]),
       correction_difference=float(dc[j]),paired_correct=bool((score[li[j]]>0)==pair['left_label'] and (score[ri[j]]>0)==pair['right_label'])))
   item['methods'][name]=point;flat.append(dict(group=group,method=name,**point))
  groups[group]=item
 comparisons={}
 if 'old_linear' in names:
  for group,_ in GROUPS:
   comparisons[group]={}
   for name in names:
    if name in ('raw','old_linear'):continue
    a=groups[group]['methods'][name];b=groups[group]['methods']['old_linear'];items={}
    for metric,rep in rep_cache.get((group,name),{}).items():
     refrep=rep_cache.get((group,'old_linear'),{}).get(metric)
     if refrep is not None and a.get(metric) is not None and b.get(metric) is not None:
      with np.errstate(invalid='ignore'):difference_reps=rep-refrep
      items[metric]=dict(difference=float(a[metric]-b[metric]),ci=ci(difference_reps))
    comparisons[group][name]=items
 result=dict(comparisons_vs_old_linear=comparisons,dataset='MEGA-CDP',dataset_split='official_test',scope='First released gold decision node of every official reference path; controlled node faithfulness, not full-path counterfactual accuracy.',
  protocol='Repair-v2 action/background edits, frozen methods and calibration; no fitting or test-time selection.',
  cases_manifest=str(args.cases),pairs_manifest=str(args.pairs),frozen_manifest=str(args.manifest),
  methods=names,groups=groups,raw_ap_difference_rms=apscale,
  coverage=dict(expected_views=len(cases),scored_views={n:int(np.isfinite(s).sum()) for n,s in scores.items()},expected_pairs=len(pairs),cache=coverage),
  uncertainty=f'{args.iterations} joint source-cluster bootstrap samples, seed {args.seed}; undefined sparse-source replicates omitted.',
  no_additional_backbone_forward_in_postprocessing=True,no_training_or_calibration=True,seconds=time.time()-started)
 dump(out/'metrics.json',result);write_csv(out/'summary.csv',flat);write_csv(out/'pair_results.csv',pair_rows)
 dump(out/'done.json',dict(state='complete',methods=names,expected_views=len(cases),expected_pairs=len(pairs),seconds=time.time()-started))
 print(json.dumps(dict(state='complete',expected_views=len(cases),expected_pairs=len(pairs),out=str(out))))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cases',required=True);p.add_argument('--pairs',required=True);p.add_argument('--nodes',nargs='+',required=True)
 p.add_argument('--manifest',default=str(BASE/'frozen_methods.json'));p.add_argument('--out',required=True);p.add_argument('--methods',nargs='+')
 p.add_argument('--iterations',type=int,default=2000);p.add_argument('--seed',type=int,default=11);summarize(p.parse_args())
