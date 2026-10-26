"""Check paired construction and scoring units; these are not P4 attention tests."""
from pathlib import Path
import collections,json,hashlib,re
from run_p1 import render
BASE=Path(__file__).resolve().parent
rows=[json.loads(s) for s in (BASE/'private/cases.jsonl').read_text().splitlines()]
groups=collections.defaultdict(dict)
for r in rows:groups[r['id']][r['variant']]=r
for cid,vs in groups.items():
 assert set(vs)>={'NC','IC','AC','SW'}
 n=vs['NC'];g=int(n['gold'])-1;s=1-g
 assert len({q['condition'] for q in n['rules']})==2
 assert len({render(vs[v],'blind') for v in ['NC','IC','AC','SW']})==1
 for v in ['NC','IC','AC','SW']:
  r=vs[v];assert r['gold']==n['gold'] and r['patient']==n['patient'] and r['options']==n['options']
 assert vs['AC']['rules'][g]['action']!=n['rules'][g]['action'] and vs['AC']['rules'][s]['action']==n['rules'][s]['action']
 assert vs['IC']['rules'][s]['action']!=n['rules'][s]['action'] and vs['IC']['rules'][g]['action']==n['rules'][g]['action']
 assert vs['SW']['rules'][g]['action']==n['rules'][s]['action'] and vs['SW']['rules'][s]['action']==n['rules'][g]['action']
 if 'CS' in vs:
  c=vs['CS'];p=c['threshold_proof'];assert c['gold']!=n['gold'] and c['patient']==n['patient'] and c['options']==n['options']
  truth=[]
  for q,old in zip(c['rules'],n['rules']):
   assert q['action']==old['action'];m=re.search(r'([<≥])([0-9.]+) mo from completion of primary therapy',q['condition']);t=float(m[2]);assert t==p['new_threshold']
   truth.append(p['patient_value']<t if m[1]=='<' else p['patient_value']>=t)
  assert sum(truth)==1 and truth[int(c['gold'])-1]
a={'status':'PASS','cases':len(groups),'rows':len(rows),'checks':['same-parent distinct branch conditions','fixed branch order and gold across four volumes','blind text identical across action-only variants','AC/IC/SW exact intended action changes','CS unique scalar branch flip with fixed patient/actions/options'],'not_P4':True}
(BASE/'results/construction_checks.json').write_text(json.dumps(a,indent=2)+'\n')
print(json.dumps(a))
