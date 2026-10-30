"""Supplementary clinical statistics and corrected root readout, with no re-generation."""
import argparse,json,random
from pathlib import Path
from collections import defaultdict,Counter
import torch
from .data import read,write
from .analyze import load,retention
from .engine import parse_prediction
from .report import ci
from src.aster.manuscript_core import parse_head
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=Path(a.run);out=run/'clinical_statistics';out.mkdir(exist_ok=False);torch.set_num_threads(4)
 expert=json.loads((run/'analysis/expert.json').read_text());stats=json.loads((run/'analysis/statistics.json').read_text());held=[r for r in load(run) if r['row']['split']=='confirm'];rng=torch.Generator().manual_seed(20261030);randoms=[]
 for name,c in expert['candidates'].items():
  h=parse_head(name);rank=len(c['basis'][0])
  for _ in range(3):randoms.append(retention(held,h,torch.linalg.qr(torch.randn(128,rank,dtype=torch.float64,generator=rng)).Q))
 selected={}
 for name in expert['selected_heads']:
  s=stats[name];cs=[x['cosine'] for x in s['unseen_rule_cosines'] if x['cosine'] is not None];r=[x for states in s['heldout_query_responses'].values() for mat in states.values() for q in mat for x in q]
  selected[name]={k:s[k] for k in ['training_energy_retention','heldout_all_difference_energy','mean_norm_retention','guideline_half_overlap','random_overlap']}
  selected[name].update(unseen_rule_cosine_mean=sum(cs)/len(cs),unseen_rule_positive_fraction=sum(v>0 for v in cs)/len(cs),heldout_query_positive_fraction=sum(v>0 for v in r)/len(r))
 write(out/'structure.json',{'selected_heads':selected,'same_rank_random_heldout_energy':sum(randoms)/len(randoms),'random_repeats':3,'clinical_truth_scope':'current-age predicates only; no physician adjudication'})
 rows=[]
 for shard in [0,1]:rows+=read(run/('mechanism'+str(shard))/'results_private.jsonl')
 result=[]
 for route in ['gold_root_diagnostic','predicted_root_diagnostic']:
  for method in ['baseline','zero','spectral','mean','random','reverse','score_bias','enhance_only','suppress_only']:
   rs=[r for r in rows if r['route']==route and r['method']==method];correct=[];invalid=0;mass=[];missing=0
   for r in rs:
    if r['status']!='complete':correct.append(False);continue
    leaf,path,bad=parse_prediction(r['result']['text']);root=path[0] if path else 'NONE' if leaf=='NONE' else None
    correct.append(root==r['expected_root']);missing+=root is None
    if r['expected_root'] not in ['R0Y','R0N']:continue
    other='R0N' if r['expected_root']=='R0Y' else 'R0Y';seen=set()
    for entry in r['result']['attention']:
     key=entry['layer'],entry['kv_head']
     if key in seen:continue
     seen.add(key);num=sum(entry['mass'].get(r['expected_root'],[0]));den=sum(entry['mass'].get(other,[0]))
     if den>0:mass.append(num/den)
   result.append({'route':route,'method':method,'denominator':len(rs),'root_correct':sum(correct),'root_accuracy':sum(correct)/len(rs),'missing_root_readout':missing,'first_query_mass_ratio':sum(mass)/len(mass) if mass else None,'technical_failures':sum(r['status']=='failed' for r in rs)})
 write(out/'mechanism_root_readout_v2.json',{'scoring_version':'v2_missing_output_is_not_semantic_NONE','legacy_root_correct_field_preserved':True,'units':len(rows),'comparisons':result})
 print(json.dumps({'clinical_units':len(rows),'selected_heads':expert['selected_heads'],'random_energy':sum(randoms)/len(randoms)}))
if __name__=='__main__':main()

