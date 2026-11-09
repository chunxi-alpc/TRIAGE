
"""Last scalar control: recalibrate old16-dimensional correction on NLI train."""
from pathlib import Path
import json,datetime
import numpy as np
from scipy.special import expit
from scipy.optimize import brentq
BASE=Path('/home/gcx/aster_next_20261109')
OUT=Path('/home/gcx/aster_hard_optimization_20261109/nli_intercept_calibration/old_linear')
def rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def dump(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def evaluate(s,y):return dict(n=len(y),correct=int(np.sum((s>0)==y)),accuracy=float(np.mean((s>0)==y)),CE=float(np.mean(np.logaddexp(0,s)-y*s)))
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 dump(OUT/'plan.json',dict(created_at=now(),method_name='NLI in-domain intercept calibration of original16-dimensional main method',
  fixed_function='ell+Nw from original constrained MEGA model; w/support/gain unchanged',
  fitting='One scalar b minimizes BCE on NLI train1700 only',
  dev_rule='Advance iff dev correct>=150, the original old_linear score; no dev threshold optimization',
  scalar_fits=1,no_further_search=True,test_scope='Original fixed old_linear error cases/pairs only if dev criterion holds'))
 frozen=json.loads((BASE/'frozen_methods.json').read_text())
 weights=np.load(frozen['original_linear_weights']);old_b=float(weights[frozen['original_linear_key']+'_b'])
 scores=np.load(BASE/'primary_results/nli4ct_all/pair_summary/case_scores.npz')
 ix={v:i for i,v in enumerate(scores['ids'].tolist())};data={}
 for split in ['train','dev']:
  cc=rows(BASE/('inventory/nli4ct_'+split+'_cases.jsonl'));ii=np.array([ix[c['case_id']] for c in cc])
  s=scores['old_linear'][ii].astype(float);data[split]=dict(base=s-old_b,old=s,y=np.array([c['label'] for c in cc]))
 fun=lambda b:float(np.mean(expit(data['train']['base']+b)-data['train']['y']))
 lo,hi=-1.,1.
 while fun(lo)>0:lo*=2
 while fun(hi)<0:hi*=2
 b=float(brentq(fun,lo,hi,xtol=1e-12))
 metrics={sp:dict(original=evaluate(v['old'],v['y']),intercept_calibrated=evaluate(v['base']+b,v['y'])) for sp,v in data.items()}
 advance=metrics['dev']['intercept_calibrated']['correct']>=150
 selected=dict(frozen=advance,saved_at=now(),method_name='NLI in-domain intercept calibration of old_linear',
  original_method='old_linear',b=b,original_b=old_b,intercept_shift=b-old_b,
  train_n=1700,dev_n=200,dev_criterion_met=advance,fit_dev_metrics=metrics,
  fitting_only_train_labels=True,no_test_labels_for_fitting_or_selection=True,
  correction_function_unchanged=True,pair_differences_unchanged=True,plan=str(OUT/'plan.json'))
 dump(OUT/'selected.json',selected);dump(OUT/'fit_dev_summary.json',metrics)
 np.savez(OUT/'calibration.npz',b=b,original_b=old_b,shift=b-old_b)
 print(json.dumps(selected))
if __name__=='__main__':main()

