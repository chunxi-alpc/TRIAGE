"""Matched CAD128 unfiltered/filter comparison, with separate choice-constrained controls."""
import argparse,json,collections
from pathlib import Path
from analyze import metrics,paired
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--model',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
root=a.root;model=a.model;all_new={};report={'model':model,'cases':500,'filter_beta':.1,'max_new_tokens':128,'comparisons':{},'option_order':{},'notes':['New filtered free-generation results do not replace either original CAD8 or unfiltered CAD128.','The first-token logits retain unfiltered CAD scores; candidate mask changes greedy token selection.','Filter threshold was fixed after a 3-failure-case diagnostic, not an independently held-out hyperparameter validation.','Choice-constrained controls are reported separately and are not free-generation accuracy.']}
def key(r):return r['id'],r['variant'],r['method'],r.get('assigned_wrong',False)
for order in ['original','left','right']:
 source=root/f'{model}_full.jsonl' if order=='original' else root/'option_order'/f'{model}_{order}.jsonl'
 old={key(r):r for r in map(json.loads,source.read_text().splitlines())}
 path=root/'cad_fix'/f'{model}_{order}_filtered.jsonl';manifest=json.loads(path.with_suffix('.manifest.json').read_text());new=list(map(json.loads,path.read_text().splitlines()));assert manifest['complete'] and len(new)==4500 and len({key(r) for r in new})==4500
 all_new[order]={key(r):r for r in new};groups=collections.defaultdict(list)
 for r in new:groups[(r['method'],r['variant'],r.get('assigned_wrong',False))].append(r)
 for k,rs in groups.items():
  before=[old[key(r)] for r in rs]
  report['comparisons']['/'.join(map(str,[order,*k]))]={'unfiltered':metrics(before),'filtered':metrics(rs),'paired':paired(before,rs),'recovered_parse':sum(b['prediction'] is None and r['prediction'] is not None for b,r in zip(before,rs)),'lost_parse':sum(b['prediction'] is not None and r['prediction'] is None for b,r in zip(before,rs)),'hit_length_limit':sum(r.get('stop_reason')=='length' for r in rs)}
design=json.loads((root/'option_order/design.json').read_text())
for order in ['left','right']:
 groups=collections.defaultdict(list)
 for k,r in all_new[order].items():
  original=all_new['original'][k];back=design['inverse_label_maps'][order][r['id']];pred=back.get(r['prediction']);groups[k[1:]].append((original['prediction'],pred))
 for k,pairs in groups.items():
  valid=[(a,b) for a,b in pairs if a is not None and b is not None];report['option_order']['/'.join(map(str,[order,*k]))]={'n':len(pairs),'valid_pairs':len(valid),'action_agreement_valid':sum(a==b for a,b in valid)/len(valid) if valid else None}
pairs=sorted([v['paired'] for v in report['comparisons'].values()],key=lambda x:x['mcnemar_exact_p']);previous=0
for i,r in enumerate(pairs):previous=max(previous,min(1,r['mcnemar_exact_p']*(len(pairs)-i)));r['mcnemar_holm_p']=previous
a.out.mkdir(parents=True,exist_ok=True);(a.out/'summary.json').write_text(json.dumps(report,indent=2))
with (a.out/'scores.jsonl').open('w') as f:
 for order,rr in all_new.items():
  for r in rr.values():f.write(json.dumps(dict(order=order,**{k:v for k,v in r.items() if k not in ('completion','attention','unconstrained_first_token')}))+'\n')
print('CAD correction analysis complete')
