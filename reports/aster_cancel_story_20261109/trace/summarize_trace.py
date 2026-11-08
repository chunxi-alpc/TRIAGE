"""Summarize the same repair-v2 held-out pairs: intervention transmission and trace predictability."""
import csv,json,math
from pathlib import Path
from collections import Counter
import numpy as np
ROOT=Path('/home/gcx/aster_cr_repair_v2_run_20261108')
REPO=Path('/home/gcx/TRIAGE-worktrees/aster-cr-20261108')
OUT=Path('/home/gcx/aster_cancel_story_20261109/trace')
OUT.mkdir(parents=True,exist_ok=True)
patch=[json.loads(s) for s in (ROOT/'qwen/mechanisms/h2_dev.jsonl').read_text().splitlines() if s]
cached=list(csv.DictReader((REPO/'reports/aster_h2_alignment_20261108/pair_results.csv').open()))
pred={(r['method'],r['pair_id']):r for r in cached if r['split']=='dev'}
orig=json.loads((ROOT/'qwen/h2.json').read_text())
records=[]; summaries=[]
def rms(x):return float(np.sqrt(np.mean(np.square(x)))) if len(x) else None
for rank in ('4','8','16','full'):
 for group,kinds in [('AP',('A','P')),('A',('A',)),('P',('P',)),('CF',('C','F')),('C',('C',)),('F',('F',))]:
  subset=[r for r in patch if r['kind'] in kinds and ('ridge_r'+rank,r['pair_id']) in pred]
  if not subset:continue
  y=np.array([r['total_effect'] for r in subset],float)
  m=np.array([r['leakage_by_rank'][rank] for r in subset],float)
  yp=np.array([float(pred['ridge_r'+rank,r['pair_id']]['predicted_nuisance_difference']) for r in subset],float)
  denom=float(y@y); centered=float(np.sum((y-y.mean())**2))
  out=dict(rank=rank,group=group,n=len(subset),sources=len({r['source_id'] for r in subset}),
    transmission_aligned_slope=float(y@m/denom),transmission_abs_ratio=float(np.abs(m).sum()/np.abs(y).sum()),
    transmission_rms_ratio=rms(m)/rms(y),transmission_mean_logit=float(m.mean()),
    transmission_rms_logit=rms(m),total_rms_logit=rms(y),
    transmission_r2=1-float(np.sum((y-m)**2))/centered if centered else None,
    trace_r2=1-float(np.sum((y-yp)**2))/centered if centered else None,
    trace_residual_rms=rms(y-yp),trace_prediction_rms=rms(yp),target_mean=float(y.mean()),
    full_head_abs_ratio=float(sum(abs(r['full_head']) for r in subset)/sum(abs(r['total_effect']) for r in subset)),
    full_head_aligned_slope=float(sum(r['full_head']*r['total_effect'] for r in subset)/denom),
    random_head_abs_ratio=float(sum(abs(r['random_heads']) for r in subset)/sum(abs(r['total_effect']) for r in subset)),
    random_head_aligned_slope=float(sum(r['random_heads']*r['total_effect'] for r in subset)/denom),
    old_head_abs_ratio=float(sum(abs(r['old_heads']) for r in subset)/sum(abs(r['total_effect']) for r in subset)))
  summaries.append(out)
  for r,p in zip(subset,yp):
   records.append(dict(rank=rank,group=group,pair_id=r['pair_id'],kind=r['kind'],source_id=r['source_id'],
      total_effect=r['total_effect'],trace_patch_effect=r['leakage_by_rank'][rank],
      trace_predicted_difference=float(p),trace_residual_difference=float(r['total_effect']-p)))
with (OUT/'summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
with (OUT/'pairs.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
summary=dict(data_version='repair-v2',model='qwen',intervention='one-way left-to-right patch at decision position; not old bidirectional protocol',
 coefficients='existing fit-only A/P Ridge(alpha=.01, fit_intercept=False); same N as patch rank',
 source_patch=str(ROOT/'qwen/mechanisms/h2_dev.jsonl'),
 source_prediction=str(REPO/'reports/aster_h2_alignment_20261108/pair_results.csv'),
 matched_counts=dict(Counter(r['kind'] for r in patch)),
 note='A/P predictability R2 and intervention transmission are distinct quantities. Report numerator and denominator definitions explicitly. No pairwise division or effect threshold filtering.',
 original_h2_ap_r2_with_pair_intercept=orig['by_rank'],results=summaries)
(OUT/'metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
lines=['# repair-v2 ??????????','',
'?????????????????????? left?right????????? repair-v2?',
'?????? fit A/P ?????????????????????',
'',
'| rank | ?? | n | ???? ?|IE|/?|TE| | ???? ?IE?TE/?TE? | ?? RMS ? | ?? R? |',
'|---|---|---:|---:|---:|---:|---:|']
for r in summaries:
 if r['group'] in ('AP','A','P'):
  lines.append('| {rank} | {group} | {n} | {transmission_abs_ratio:.4f} | {transmission_aligned_slope:.4f} | {transmission_rms_ratio:.4f} | {trace_r2:.4f} |'.format(**r))
lines+=['','?|IE|/?|TE| ??????????????????????????????RMS ???????',
'???????????????????????????? R? ????????????????????',
'????????? pairs.csv?????????????']
(OUT/'REPORT.zh.md').write_text('\n'.join(lines)+'\n')
print(json.dumps([r for r in summaries if r['rank']=='16'],ensure_ascii=False))

