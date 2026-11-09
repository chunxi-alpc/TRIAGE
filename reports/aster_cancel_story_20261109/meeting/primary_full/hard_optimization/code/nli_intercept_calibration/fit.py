
"""One scalar NLI intercept calibration of the unchanged frozen balanced edit."""
from pathlib import Path
import json,datetime,time
import numpy as np
from scipy.special import expit
from scipy.optimize import brentq
BASE=Path('/home/gcx/aster_next_20261109')
OUT=Path('/home/gcx/aster_hard_optimization_20261109/nli_intercept_calibration')
def rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def dump(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def metrics(s,y):return dict(n=len(y),correct=int(np.sum((s>0)==y)),accuracy=float(np.mean((s>0)==y)),CE=float(np.mean(np.logaddexp(0,s)-y*s)))
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 plan=dict(created_at=now(),method_name='NLI in-domain intercept calibration',
  frozen_correction='optimized_balanced_fit_calibration correction weights, feature standardization, gain and input template unchanged',
  training='NLI train1700 native binary labels only',selection='Evaluate once on NLI dev200; advance only if correct >=148 (the previously frozen balanced dev score). No threshold search.',
  objective='mean BCE(ell+fixed_balanced_delta+b,y), fit b only; replace the original MEGA intercept',
  scalar_fits=1,official_or_practice_used_for_calibration=False,
  difference_metric_note='Pair differences, variance reductions, response retention, correction differences and direction are algebraically unchanged by this constant intercept. This control cannot claim improved faithfulness.',
  retest_scope='Only existing old_linear error cases and fixed error native pairs/necessary reference endpoints if the predetermined dev criterion holds; no old-correct full-population assessment.')
 dump(OUT/'plan.json',plan)
 manifest=json.loads((BASE/'frozen_methods.json').read_text())
 old_b=float(manifest['methods']['optimized_balanced_fit_calibration']['b'])
 a=np.load(BASE/'primary_results/nli4ct_all/pair_summary/case_scores.npz')
 ix={v:i for i,v in enumerate(a['ids'].tolist())}
 cases={}
 for split in ('train','dev'):cases[split]=rows(BASE/('inventory/nli4ct_'+split+'_cases.jsonl'))
 data={}
 for split,cc in cases.items():
  ii=np.array([ix[c['case_id']] for c in cc]);y=np.array([c['label'] for c in cc])
  score=a['optimized_balanced_fit_calibration'][ii].astype(float)
  data[split]=dict(ids=[c['case_id'] for c in cc],y=y,score=score,base=score-old_b)
 fn=lambda b:float(np.mean(expit(data['train']['base']+b)-data['train']['y']))
 lo,hi=-1.,1.
 while fn(lo)>0:lo*=2
 while fn(hi)<0:hi*=2
 b=float(brentq(fn,lo,hi,xtol=1e-12))
 result={}
 for split,d in data.items():result[split]=dict(original_balanced=metrics(d['score'],d['y']),intercept_calibrated=metrics(d['base']+b,d['y']))
 advance=result['dev']['intercept_calibrated']['correct']>=148
 fixed=dict(frozen=advance,saved_at=now(),method_name=plan['method_name'],b=b,original_balanced_b=old_b,
  intercept_shift=b-old_b,correction_unchanged=True,train_n=1700,dev_n=200,dev_criterion_met=advance,
  fit_dev_metrics=result,plan=str(OUT/'plan.json'),official_or_practice_used_for_fit_or_criterion=False,
  prior_official_results_previously_viewed=True,scope=plan['difference_metric_note'])
 dump(OUT/'selected.json',fixed);dump(OUT/'fit_dev_summary.json',result)
 np.savez(OUT/'calibration.npz',b=b,original_b=old_b,shift=b-old_b)
 print(json.dumps(fixed))
if __name__=='__main__':main()

