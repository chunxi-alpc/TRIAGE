
"""Report fit/dev only and freeze both predeclared equal-weight ensembles."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
import sys,json,time,math
import numpy as np,torch
from scipy.special import expit
from scipy.optimize import brentq
from train import Editor,calibrate
BASE=Path('/home/gcx/aster_hard_optimization_20261109')
sys.path.insert(0,'/home/gcx/TRIAGE-worktrees/aster-cr-20261108')
from experiments.aster_closed_form.data import canonical_guide
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def rms(x):return float(np.sqrt(np.mean(x*x))) if len(x) else None
def tail_rms(x):return rms(np.sort(np.abs(x))[-max(1,int(math.ceil(len(x)*.2))):])
def main():
 torch.set_num_threads(2);a=dict(np.load(BASE/'data.npz'));meta=json.loads((BASE/'data.json').read_text())
 ids=a['ids'].tolist();ix={v:i for i,v in enumerate(ids)};ell=a['ell'];labels=a['labels'];natural=a['natural_fit']
 scores={'raw':ell.copy()};deltas={};models={}
 frozen=json.loads(Path('/home/gcx/aster_next_20261109/frozen_methods.json').read_text());frozen.pop('evidence_alignment',None)
 old=np.load(meta['original_weights']);linear_delta=a['x'][:,:16]@old['bounded_0.1']
 scores['old_linear']=ell+linear_delta+float(old['bounded_0.1_b']);scores['fit_bias_only']=ell+float(old['bias_only_b'])
 baseline=[]
 for member in meta['initial_balanced']['members']:
  s=torch.load(member['model_file'],map_location='cpu',weights_only=False);m=Editor(len(s['mu']),128);m.load_state_dict(s['state_dict']);m.eval()
  z=((a['x']-s['mu'])/s['scale']).astype('float32')
  with torch.no_grad():v=np.concatenate([m(torch.from_numpy(z[i:i+1024])).numpy() for i in range(0,len(z),1024)]).astype('float64')
  baseline.append(v)
 c0=np.mean(baseline,axis=0)*float(meta['initial_balanced']['gain'])
 scores['old_balanced']=ell+c0+float(meta['initial_balanced']['b'])
 entries={}
 for scheme in ('tail_pair','tail_robust'):
  records=json.loads((BASE/scheme/'models.json').read_text());dc=[]
  for record in records:
   aa=np.load(record['prediction_file']);assert aa['ids'].tolist()==ids
   delta=aa['delta'];name=record['name'];scores[name]=ell+delta+record['b'];deltas[name]=delta;dc.append(delta);models[name]=record
  correction=np.mean(dc,axis=0);b=calibrate(ell,correction,labels,natural);name='hard_'+scheme+'_fit_calibration'
  scores[name]=ell+correction+b;deltas[name]=correction
  entries[name]=dict(name=name,kind='ensemble',member_count=3,gain=1.,b=b,fit_ce_b=b,
   members=[dict(name=r['name'],kind='scorecond',model_file=r['model_file'],coefficient=1/3,seed=r['seed'],fit_scale_already_folded=r['fit_scale']) for r in records],
   coefficients=[1/3]*3,scheme=scheme,calibration='CE intercept on original 761 natural fit only',
   prediction_convention='state_dict output already includes fit-budget scale; equal mean member deltas; score=ell+mean_delta+b')
 frozen['methods'].update(entries);frozen['comparison_names']=['raw','fit_bias_only','old_linear']+list(frozen['methods'])
 frozen.update(frozen_before_error_retest=True,new_optimization_time=time.strftime('%Y-%m-%d %H:%M:%S'),
   optimization_data='Original fit AP/CF/natural + 1720 original-fit-only gold nodes; dev evaluation only; no official data loaded',
   new_method_names=list(entries),new_method_selection='Both predeclared schemes frozen as 3-seed equal ensembles; no winner or test-dependent gain/b selection')
 frozen.pop('frozen_before_expanded_evaluation',None)
 frozen.update(training=dict(new_training=True,runs=6,schemes=['tail_pair','tail_robust'],seeds=[11,22,33],steps_per_run=1200,
  fit_AP=4887,fit_CF=135,original_natural_fit=761,extra_fit_gold_nodes=1720,extra_fit_sources=12),
  evaluation='Fixed prior old_linear error-pool retest after new fit/dev optimization',
  model_selection='Retain both predeclared equal-seed ensembles; no official scheme/gain/b choice',
  freeze_scope='Frozen before this new error-pool retest only, not before earlier official full evaluation')
 dump(BASE/'frozen_methods.json',frozen);dump(BASE/'ensembles.json',entries)
 views=meta['views'];pairs=meta['pairs'];measurements={}
 for name,ss in scores.items():
  correction=ss-ell;by_split={}
  for split in ('fit','dev'):
   aps=[p for p in pairs if p['split']==split and p['kind'] in ('A','P')];apraw=np.array([ell[ix[p['right']]]-ell[ix[p['left']]] for p in aps]);den=rms(apraw)
   groups={}
   for group,kinds in [('A',('A',)),('P',('P',)),('AP',('A','P')),('C',('C',)),('F',('F',)),('CF',('C','F')),('F0',('F0',))]:
    pp=[p for p in pairs if p['split']==split and p['kind'] in kinds];li=np.array([ix[p['left']] for p in pp]);ri=np.array([ix[p['right']] for p in pp])
    raw=ell[ri]-ell[li];dy=ss[ri]-ss[li];dc=dy-raw;yl=np.array([p['left_label'] for p in pp]);yr=np.array([p['right_label'] for p in pp])
    correct=((ss[li]>0)==yl)&((ss[ri]>0)==yr)
    m=dict(n=len(pp),difference_rms=rms(dy),difference_tail20_rms=tail_rms(dy),difference_abs_p95=float(np.quantile(np.abs(dy),.95)),difference_abs_max=float(np.max(np.abs(dy))),
     correction_rms=rms(dc),correction_tail20_rms=tail_rms(dc),correction_abs_p95=float(np.quantile(np.abs(dc),.95)),correction_abs_max=float(np.max(np.abs(dc))),
     correction_rms_over_ap_rms=rms(dc)/den,paired_correct=int(correct.sum()),paired_accuracy=float(correct.mean()))
    if group in ('A','P','AP','F0'):m['variance_reduction']=1-float(np.var(dy)/np.var(raw))
    else:
     direction=yr-yl;m.update(mean_evidence_retention=float(np.mean(direction*dy)/np.mean(direction*raw)),
      response_direction_degraded=int(np.sum((direction*raw>0)&(direction*dy<=0))),exceeds_original_fit_budget=int(np.sum(np.abs(dc)>.2923349027297149+1e-6)))
    groups[group]=m
   vv=[v for v in views if v['split']==split and v['variant']=='full' and v['original']];ii=np.array([ix[v['view_id']] for v in vv])
   groups['natural']=dict(n=len(ii),correct=int(np.sum((ss[ii]>0)==labels[ii])))
   by_split[split]=groups
  measurements[name]=dict(splits=by_split)
 cases=meta['dev_cases'];case_trees={}
 for c in cases:
  t=canonical_guide(c['guideline']);case_trees[c['case_id']]=({n['id']:n for n in t['nodes']},t['nodes'][0]['id'])
 pathpos={(r['case_id'],r['gid']):ix['path|'+r['id']] for r in meta['dev_paths']}
 results={}
 for name,ss in scores.items():
  rr=[]
  for c in cases:
   nodes,gid=case_trees[c['case_id']];actual=[];error=None
   while gid in nodes:
    if len(actual)>len(nodes):error='cycle';break
    pos=pathpos.get((c['case_id'],gid))
    if pos is None:error='missing_dev_cache';break
    y=int(ss[pos]>0);actual.append(dict(gid=gid,label=y));gid=nodes[gid]['yes' if y else 'no']
   first=next((p['gid'] for i,p in enumerate(actual) if i>=len(c['gold_path']) or p!=c['gold_path'][i]),None)
   rr.append(dict(case_id=c['case_id'],source_id=c['source_id'],correct=not error and actual==c['gold_path'] and gid==c['gold_leaf'],first_error=first,path=actual,leaf=gid,error=error))
  results[name]=rr
 baseline_wrong={r['case_id'] for r in results['old_linear'] if not r['correct']};baseline_correct={r['case_id'] for r in results['old_linear'] if r['correct']}
 for name,rr in results.items():
  good={r['case_id'] for r in rr if r['correct']}
  measurements[name]['dev_path']=dict(n=len(rr),correct=len(good),accuracy=len(good)/len(rr),old_linear_error_pool=len(baseline_wrong),
   repaired_vs_old_linear=len(good&baseline_wrong),new_errors_vs_old_linear=len(baseline_correct-good),
   repaired_cases=sorted(good&baseline_wrong),new_error_cases=sorted(baseline_correct-good),missing=sum(r['error'] is not None for r in rr))
 with (BASE/'dev_path_results.jsonl').open('w') as f:
  for name,rs in results.items():
   for r in rs:f.write(json.dumps(dict(method=name,**r),ensure_ascii=False)+'\n')
 np.savez_compressed(BASE/'dev_scores.npz',ids=a['ids'],**scores)
 dump(BASE/'dev_metrics.json',dict(methods=measurements,models=models,new_ensembles=entries,
  training_scope='fit only',evaluation_scope='fit/dev only',official_data_loaded=False,original_fit_budget=.2923349027297149))
 # Verify helper inference numerically on a small fit/dev slice, without official cases.
 from predict import Predictor
 helper=Predictor(BASE/'frozen_methods.json');sub=slice(0,32);q=a['x'][sub,96:99]
 pred=helper.predict_from_coordinates(dict(ell=ell[sub],r32=a['x'][sub,:32],residual64=a['x'][sub,32:96],q=q))
 errors={n:float(np.max(np.abs(pred[n]-scores[n][sub]))) for n in entries}
 dump(BASE/'validation.json',dict(fit_budget_satisfied=all(r['fit_cf_max_after_scale']<=r['fit_budget']+1e-5 for r in models.values()),
  fit_ce_calibration_count=len(natural),helper_max_abs_error=errors,official_data_loaded=False,gpu_backbone_forwards=0))
 dump(BASE/'dev_done.json',dict(state='complete',fixed_runs=6,predeclared_ensembles=2,frozen_before_official_error_retest=True))
 print(json.dumps({name:dict(AP_VR=measurements[name]['splits']['dev']['AP']['variance_reduction'],AP_tail20=measurements[name]['splits']['dev']['AP']['difference_tail20_rms'],
  CF_ER=measurements[name]['splits']['dev']['CF']['mean_evidence_retention'],CF_rms=measurements[name]['splits']['dev']['CF']['correction_rms'],
  CF_p95=measurements[name]['splits']['dev']['CF']['correction_abs_p95'],CF_max=measurements[name]['splits']['dev']['CF']['correction_abs_max'],
  natural=measurements[name]['splits']['dev']['natural'],path=measurements[name]['dev_path']) for name in ['old_linear','old_balanced',*entries]}))
if __name__=='__main__':main()

