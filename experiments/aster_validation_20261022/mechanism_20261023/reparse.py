"""Correct explicit leading-answer parsing using preserved completions, no inference."""
import argparse,json,hashlib
from pathlib import Path
from run import parse
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert not a.out.exists();cases={r['id']:r for r in map(json.loads,a.dataset.read_text().splitlines())};rows=[];changed=0
for r in map(json.loads,a.input.read_text().splitlines()):
 c=cases[r['id']];labels=[chr(65+i) for i in range(len(c['branches']))] if r['kind']=='branch' else c['options'];old=r['prediction']
 # Existing ABC baseline observations are retained exactly as originally parsed.
 if r['kind']=='branch' or (not r.get('reused') and not r.get('upstream_branch_parse_failure')):r['prediction']=parse(r['completion'],labels)
 r['parser_revision']='explicit_first_line_v2';r['previous_prediction']=old;r['correct']=r['prediction']==r['gold'];r['parse_failure']=r['prediction'] is None
 if r['kind']=='action':
  r['original_action_selected']=r['prediction']==c['original_answer'];rules=c['variant_rules'][r['variant']];r['branch_permitted_action']=r['prediction'] in [k for k,t in c['options'].items() if t in [rules[i]['action'] for i in c['branches'][c['gold_branch'][r['variant']]]['rule_indices']]];r['sibling_action_selected']=r['prediction'] in [k for k,t in c['options'].items() if t in [rules[i]['action'] for i in c['branches'][c['sib_branch']]['rule_indices']]]
 changed+=old!=r['prediction'];rows.append(r)
a.out.write_text(''.join(json.dumps(r)+'\n' for r in rows));m=json.loads(a.input.with_suffix('.manifest.json').read_text());m.update(parser_revision='explicit_first_line_v2',parser_changes=changed,source_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest());a.out.with_suffix('.manifest.json').write_text(json.dumps(m,indent=2));print(json.dumps({'rows':len(rows),'changed':changed}))
