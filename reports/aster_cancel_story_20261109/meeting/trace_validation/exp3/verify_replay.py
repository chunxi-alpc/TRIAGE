"""Compare saved full CV and CPU replay predictions/heads; no GPU required."""
from pathlib import Path
import csv,json
HERE=Path(__file__).resolve().parent
rr=[]
for fold in range(5):
 a=HERE/f'full/fold{fold}';b=HERE/f'full/replay/fold{fold}'
 with (a/'pair_predictions.csv').open() as f:x=list(csv.DictReader(f))
 with (b/'pair_predictions.csv').open() as f:y=list(csv.DictReader(f))
 error=max(abs(float(u['prediction'])-float(v['prediction'])) for u,v in zip(x,y))
 ac=json.loads((a/'configuration.json').read_text());bc=json.loads((b/'configuration.json').read_text());bm=json.loads((b/'metrics.json').read_text())
 rr.append(dict(fold=fold,pairs=len(x),pair_order_equal=[u['pair_id'] for u in x]==[u['pair_id'] for u in y],heads_equal=ac['heads']==bc['heads'],maximum_prediction_difference=error,replay_new_final_forwards=bm['new_final_forwards']))
passed=all(x['pair_order_equal'] and x['heads_equal'] and x['maximum_prediction_difference']==0 and x['replay_new_final_forwards']==0 for x in rr)
(HERE/'replay_verification.json').write_text(json.dumps(dict(folds=rr,passed=passed,evidence_type='cached_cpu_replay_no_model_forwards'),indent=2)+'\n')
print(json.dumps(dict(passed=passed,folds=len(rr))))
if not passed:raise SystemExit(1)

import sys
import numpy as np
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent))
from shared import Data,ROOT,rows
from run_cv import weighted_subset
d=Data();a=np.load(HERE/'full/attribution_by_source.npz');attrs=a['attr'].sum(0)/a['den'].sum(0)[:,None];sources=set(a['sources'].tolist())
original=json.loads((ROOT/'qwen/selection.json').read_text());exact={(r['pair_id'],r['head']):r for r in rows(ROOT/'qwen/mechanisms/fit_exact.jsonl')}
ranks={}
for ki,kind in enumerate(('A','P','C','F')):
 candidate=np.argsort(-attrs[ki])[:64].tolist();wp=weighted_subset(d.pairs,d.ell,d.ix,sources,kind)
 sums={h:sum(w*.5*(abs(exact[(p['pair_id'],h)]['ie_forward'])+abs(exact[(p['pair_id'],h)]['ie_reverse'])) for p,w in wp) for h in candidate}
 ranks[kind]=sorted(candidate,key=lambda h:(-sums[h],h))
nsort=sorted(set(ranks['A']+ranks['P']),key=lambda h:sum(ranks[t].index(h) if h in ranks[t] else 64 for t in ('A','P')))
ssort=sorted(set(ranks['C']+ranks['F']),key=lambda h:sum(ranks[t].index(h) if h in ranks[t] else 64 for t in ('C','F')))
heads=[]
for i in range(max(len(nsort),len(ssort))):
 for ranking in (nsort,ssort):
  if i<len(ranking) and ranking[i] not in heads:heads.append(ranking[i])
 if len(heads)>=51:break
result=dict(all80_candidates_match_original={k:set(ranks[k])==set(original['ranks'][k]) for k in ranks},all80_exact_ranks_match_original={k:ranks[k]==original['ranks'][k] for k in ranks},selected_head_order_equal=heads[:51]==original['heads'],reconstructed_heads=heads[:51],evidence_type='real_cache_recomputation')
(HERE/'full/attribution_reconstruction.json').write_text(json.dumps(result,indent=2)+'\n')
if not result['selected_head_order_equal']:raise SystemExit('Original localization reconstruction differs')
