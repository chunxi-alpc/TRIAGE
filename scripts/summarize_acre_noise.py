#!/usr/bin/env python3
"""Full-denominator, deterministic corruption summary; reuses unchanged oracle runs."""
import copy,gzip,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_acre_noise import plans,BASE,OUT
from src.acre_pilot.evaluator import evaluate_cases
from scripts.run_acre_pilot import write
with gzip.open(OUT/'case_results.jsonl.gz','rt') as f:original=[json.loads(x) for x in f]
refs=json.loads((BASE/'development_references.json').read_text());rows=[]
for job in plans():
 p=OUT/'noise'/(job['name']+'_'+job['item']['case_id']+'.json')
 if not p.exists():raise ValueError('Incomplete noise run: '+str(p))
 row=json.loads(p.read_text())
 if 'nonzero_added_wrong_tokens' in row:row['added_tokens_vs_clean_oracle']=row.pop('nonzero_added_wrong_tokens')
 rows.append(row)
summary={}
for method in ['predicted','reference','reference_state_only','flip_0.1','flip_0.2','flip_0.3','span_noise_25pct']:
 cases=copy.deepcopy(original);changes=[r for r in rows if r['name']==method];lookup={r['case_id']:r for r in changes}
 for c in cases:
  c['variants']={'B0':c['variants']['B0'],method:lookup[c['case_id']]['variant'] if c['case_id'] in lookup else c['variants']['B3@2.0' if method=='predicted' else 'B6@2.0']}
 metrics,_,_=evaluate_cases(cases,refs);summary[method]={'metrics':metrics[method],'selected_cases':len(changes),'changed_target_union':sum(r['target_set_changed'] for r in changes),'actual_nonzero_wrong_enhancement_cases':sum(r['added_tokens_vs_clean_oracle']>0 and r['variant']['attention']['enabled'] for r in changes) if method.startswith('flip_') or method=='span_noise_25pct' else None,'no_op_corruption':sum(not r['target_set_changed'] for r in changes),'removed_target_without_new_wrong_tokens':sum(r['removed_tokens']>0 and not r['added_tokens_vs_clean_oracle'] for r in changes)}
write(OUT/'noise_metrics.json',{'seed':20261015,'beta':2.,'scope':'Offline oracle corruption diagnostic; no training, no validation','all_80_cases_in_denominator':True,'reference_state_only_protocol':'Only rule states replaced by reference, predicted relevance/evidence retained; normal conflict gate retained. Extra tokens versus oracle are not classified as wrong evidence.','corruption_conflict_gate':'Artificial flips are executed as operator-level corruption even if they introduce a contradictory predicted applicable rule; not production gate robustness.','flip_protocol':'Nested seeded random patient target-rule states only, applicable/inapplicable exchanged; existing evidence roles unchanged. Other rule annotations remain oracle. Not a population robustness claim.','span_protocol':'Replace target active evidence with disjoint original patient window of same character length as decisive quote; unchanged-condition context, not injected text. 1/20 cannot supply such a window and is a recorded no-op. Token counts retained in per-case records.','results':summary})
with gzip.open(OUT/'noise_results.jsonl.gz','wt',encoding='utf-8') as f:
 for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
names=['reference','flip_0.1','flip_0.2','flip_0.3'];x=[0,10,20,30]
fig,axes=plt.subplots(1,2,figsize=(8,3.2))
for ax,key,title in zip(axes,['accuracy','oar'],['Action accuracy','Rule over-application rate']):
 y=[100*summary[n]['metrics'][key]['value'] for n in names];ax.plot(x,y,marker='o');ax.set(xlabel='Target-state corruption (%)',ylabel='Percent',title=title);ax.set_xticks(x);ax.grid(alpha=.25)
fig.tight_layout();fig.savefig(OUT/'noise_curve.png',dpi=180);plt.close(fig)
print(json.dumps({k:{'accuracy':v['metrics']['accuracy']['numerator'],'oar':v['metrics']['oar']['numerator'],'wrong_nonzero':v['actual_nonzero_wrong_enhancement_cases'],'union_changed':v['changed_target_union']} for k,v in summary.items()}))
