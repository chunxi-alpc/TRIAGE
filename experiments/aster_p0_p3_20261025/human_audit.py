"""Prepare private independent human-review forms; never fill human verdicts."""
import collections,csv,hashlib,json
from pathlib import Path
from analysis import BASE,ROOT,OUT,read,key,save

def main():
 cases=[c for c in read(BASE/'cases.jsonl') if c['split']=='test' and c['seed']==0];selected=[]
 for variant in ('NC','AC','IC','SW','CS','CS-null'):
  groups=collections.defaultdict(list)
  for c in cases:
   if c['variant']==variant:groups[c['tree']].append(c)
  for g in groups.values():g.sort(key=lambda c:hashlib.sha256(key(c).encode()).hexdigest())
  chosen=[]
  while len(chosen)<25:
   progress=False
   for tree in sorted(groups):
    if groups[tree] and len(chosen)<25:chosen.append(groups[tree].pop());progress=True
   if not progress:break
  selected+=chosen
 out=ROOT/'human_audit';out.mkdir(exist_ok=True)
 fields=['review_id','patient','rules','options','applicable_rule','correct_option','construction_valid','evidence_quote','notes']
 for reviewer in ('A','B'):
  p=out/f'reviewer_{reviewer}.csv'
  if p.exists():continue
  with p.open('w',encoding='utf-8-sig') as f:
   w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
   for c in selected:w.writerow({'review_id':hashlib.sha256(key(c).encode()).hexdigest()[:16],'patient':c['patient'],'rules':json.dumps(c['rules'],ensure_ascii=False),'options':json.dumps(c['options'],ensure_ascii=False)})
 manifest={'n':len(selected),'design':'25 per variant; seed0 only; round-robin trees; deterministic hash within tree; descriptive audit, not prevalence estimate','review_ids':{hashlib.sha256(key(c).encode()).hexdigest()[:16]:key(c) for c in selected}}
 (out/'private_manifest.json').write_text(json.dumps(manifest,indent=2))
 status={'requested_rows_per_reviewer':len(selected),'sample_design':manifest['design'],'completed':{},'agreement':None,'status':'pending'};answers={}
 for reviewer in ('A','B'):
  rr=list(csv.DictReader((out/f'reviewer_{reviewer}.csv').open(encoding='utf-8-sig')))
  good={r['review_id']:r for r in rr if all(r[k].strip() for k in ('applicable_rule','correct_option','construction_valid','evidence_quote'))};status['completed'][reviewer]=len(good);answers[reviewer]=good
 common=set(answers['A'])&set(answers['B'])
 if common:
  status['agreement']={k:sum(answers['A'][i][k]==answers['B'][i][k] for i in common)/len(common) for k in ('applicable_rule','correct_option','construction_valid')};status['paired_rows']=len(common)
 if len(common)==len(selected):status['status']='double_review_complete_requires_disagreement_adjudication'
 save('human_audit_status.json',status)
if __name__=='__main__':main()
