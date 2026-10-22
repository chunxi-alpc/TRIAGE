"""Prespecified paired option-order sensitivity; two rotations, 200 held-out cases."""
import argparse,collections,hashlib,json
from pathlib import Path

def rotate(row,shift):
 r=dict(row);old=row['options'];labels=list(old);n=len(labels)
 mapping={label:labels[(i+shift)%n] for i,label in enumerate(labels)}
 new={label:old[labels[(i-shift)%n]] for i,label in enumerate(labels)}
 oldblock='Options:\n'+'\n'.join(f'{k}: {v}' for k,v in old.items())
 newblock='Options:\n'+'\n'.join(f'{k}: {v}' for k,v in new.items())
 for field in ('prompt_text','closed_book_prompt','direct_rule_prompt'):
  if field not in r:continue
  assert r[field].count(oldblock)==1,(r['id'],field)
  r[field]=r[field].replace(oldblock,newblock)
 for field in ('gold_answer','original_answer','sibling_action_label','original_sibling_action_label','replacement_action_label'):
  if field in r:r[field]=mapping[r[field]]
 r['options']=new
 # Rules precede the options block; text spans must remain unchanged.
 for span,text in [('applicable_char_span','applicable_rule_text'),('sibling_char_span','sibling_rule_text')]:
  lo,hi=r[span];assert r['prompt_text'][lo:hi]==r[text]
 return r,{v:k for k,v in mapping.items()}

def prepare(root):
 manifest=json.loads((root/'dataset_manifest.json').read_text());ids=sorted(manifest['splits']['test']['ids'],key=lambda x:hashlib.sha256(('option-order-20261023:'+x).encode()).hexdigest())[:200]
 rows=[json.loads(l) for l in (root/'all.jsonl').read_text().splitlines()];rows=[r for r in rows if r['id'] in set(ids)]
 dest=root/'option_order';dest.mkdir(exist_ok=True);maps={}
 for rotation in ('left','right'):
  prepared=[];maps[rotation]={}
  for row in rows:
   r,back=rotate(row,1 if rotation=='left' else len(row['options'])-1);prepared.append(r);maps[rotation][row['id']]=back
  path=dest/(rotation+'.jsonl');payload=''.join(json.dumps(r)+'\n' for r in prepared)
  if path.exists():assert path.read_text()==payload
  else:path.write_text(payload)
 (dest/'design.json').write_text(json.dumps({'ids':ids,'n':len(ids),'inverse_label_maps':maps,'selection':'SHA256 rank on heldout IDs, independent of outcomes','rotations':['left','right'],'source_sha256':hashlib.sha256((root/'all.jsonl').read_bytes()).hexdigest()},indent=2))
 print('Prepared',len(ids),'held-out cases x two rotations')

def analyze(root,model,out):
 from scipy.stats import binomtest
 design=json.loads((root/'option_order/design.json').read_text());ids=set(design['ids'])
 def key(r):return r['id'],r['variant'],r['method'],r.get('assigned_wrong',False)
 original={key(r):r for r in map(json.loads,(root/f'{model}_complete.jsonl').read_text().splitlines()) if r['id'] in ids}
 report={'model':model,'n':len(ids),'design':{k:v for k,v in design.items() if k!='inverse_label_maps'},'comparisons':{},'notes':['Same action compared by inverse label mapping, not by printed letter.','Each rotation analyzed separately; rotations are not independent extra cases.','K/U is fixed from original-order closed-book output for subgroup comparisons.','Judgment yes/no/unknown options are unchanged and excluded from order comparisons.','Exploratory paired tests; no sequential significance claim.']}
 scores=[]
 for rotation in design['rotations']:
  path=root/'option_order'/f'{model}_{rotation}_complete.jsonl';rr=[json.loads(l) for l in path.read_text().splitlines()]
  assert len(rr)==len(ids)*58 and len({key(r) for r in rr})==len(rr)
  buckets=collections.defaultdict(list)
  for r in rr:
   if r['method'].startswith('judgment'):continue
   b=original[key(r)];back=design['inverse_label_maps'][rotation][r['id']];pred=back.get(r['prediction']);valid=pred is not None and b['prediction'] is not None
   pair={'id':r['id'],'original_correct':bool(b['correct']),'reordered_correct':bool(r['correct']),'original_prediction':b['prediction'],'reordered_action_as_original_label':pred,'valid_pair':valid,'same_action':pred==b['prediction'] if valid else None,'original_invalid':b['prediction'] is None,'reordered_invalid':r['prediction'] is None,'disease':r['disease']}
   for stratum in ['all',b['stratum']]:buckets[(r['method'],r['variant'],r.get('assigned_wrong',False),stratum)].append(pair)
   scores.append(dict(pair,rotation=rotation,method=r['method'],variant=r['variant'],assigned_wrong=r.get('assigned_wrong',False)))
  for (method,variant,wrong,stratum),pairs in buckets.items():
   n=len(pairs);wins=sum(not p['original_correct'] and p['reordered_correct'] for p in pairs);losses=sum(p['original_correct'] and not p['reordered_correct'] for p in pairs);valid=[p for p in pairs if p['valid_pair']]
   report['comparisons']['/'.join(map(str,[rotation,method,variant,wrong,stratum]))]={'n':n,'original_accuracy':sum(p['original_correct'] for p in pairs)/n,'reordered_accuracy':sum(p['reordered_correct'] for p in pairs)/n,'delta_accuracy':(wins-losses)/n,'action_agreement_valid':sum(p['same_action'] for p in valid)/len(valid) if valid else None,'valid_pairs':len(valid),'original_invalid':sum(p['original_invalid'] for p in pairs),'reordered_invalid':sum(p['reordered_invalid'] for p in pairs),'wins':wins,'losses':losses,'mcnemar_p':binomtest(wins,wins+losses,.5).pvalue if wins+losses else 1}
 # Holm within this model across all reported comparisons.
 ordered=sorted(report['comparisons'].values(),key=lambda x:x['mcnemar_p']);previous=0
 for i,r in enumerate(ordered):previous=max(previous,min(1,r['mcnemar_p']*(len(ordered)-i)));r['mcnemar_p_holm']=previous
 out.mkdir(parents=True,exist_ok=True);(out/'summary.json').write_text(json.dumps(report,indent=2));(out/'paired_scores.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in scores))
 print('Option-order analysis complete',model)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','analyze']);p.add_argument('--root',type=Path,required=True);p.add_argument('--model');p.add_argument('--out',type=Path);a=p.parse_args()
 if a.mode=='prepare':prepare(a.root)
 else:analyze(a.root,a.model,a.out)
