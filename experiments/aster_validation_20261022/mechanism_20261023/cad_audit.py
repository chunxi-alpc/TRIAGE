"""Read-only audit after current CAD completes; never repeat inference."""
import json,collections,argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--model',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();result={}
for revision,path in [('8token',a.root/f'{a.model}_full_8token_original.jsonl'),('128_unfiltered',a.root/f'{a.model}_full.jsonl'),('128_filtered',a.root/'cad_fix'/f'{a.model}_original_filtered.jsonl')]:
 groups=collections.defaultdict(list)
 for r in map(json.loads,path.read_text().splitlines()):
  if r['method'] in ['cad','prune_cad']:groups[(r['method'],r['variant'],r.get('assigned_wrong',False))].append(r)
 for key,rows in groups.items():
  n=len(rows);invalid=sum(r['prediction'] is None for r in rows);correct=sum(r['correct'] for r in rows);valid=n-invalid;result['/'.join(map(str,[revision,*key]))]={'n':n,'parse_failures':invalid,'valid_choice_errors':n-invalid-correct,'correct':correct,'accuracy_all':correct/n,'choice_error_rate_given_valid':(valid-correct)/valid if valid else None,'length_limit':sum(r.get('stop_reason')=='length' for r in rows)}
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps({'read_only':True,'model':a.model,'cells':result,'note':'Invalid output and a valid but incorrect choice are disjoint categories; valid-only accuracy is supplementary, not a replacement for all-case accuracy.'},indent=2))
