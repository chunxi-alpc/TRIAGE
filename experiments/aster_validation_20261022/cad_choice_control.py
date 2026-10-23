"""Exact one-token legal-choice decoding from saved first-step logits; not free generation."""
import argparse,json,collections,hashlib
from pathlib import Path
from analyze import metrics,paired
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
source=Path(a.input);rows=[json.loads(l) for l in source.read_text().splitlines()];chosen=[];buckets=collections.defaultdict(lambda:[[],[]])
for row in rows:
 if row['method'] not in ('baseline','prune','cad','prune_cad'):continue
 scores=row['option_logits'];pred=max(scores,key=scores.get);r=dict(row);r.update(prediction=pred,parsed_prediction=pred,correct=pred==row['gold_answer'],completion=pred,decoding_revision='one-token legal-choice mask',execution='exact deterministic argmax from saved first-position logits; no new model forward',free_generation=False)
 chosen.append(r);k=(r['method'],r['variant'],r.get('assigned_wrong',False));buckets[k][0].append(row);buckets[k][1].append(r)
report={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'cases':len({r['id'] for r in chosen}),'rows':len(chosen),'notes':['Constrained option-label decoding, NOT free generation.','All legal option labels allowed equally; no gold label enters decoding.','Baseline and prune receive the same constraint as CAD and prune_cad.','Uses exact saved decision-token scores; computation is reused, not claimed as new GPU inference.'],'metrics':{}}
for k,(old,new) in buckets.items():report['metrics']['/'.join(map(str,k))]={'free':metrics(old),'constrained':metrics(new),'paired':paired(old,new)}
out=Path(a.output);out.mkdir(parents=True,exist_ok=True);(out/'summary.json').write_text(json.dumps(report,indent=2));(out/'predictions.jsonl').write_text(''.join(json.dumps({k:v for k,v in r.items() if k not in ('completion','attention','unconstrained_first_token')})+'\n' for r in chosen));print('Legal-choice decoding complete',len(chosen))
