"""Independent, CPU-only verification. Does not change frozen code or predictions."""
import collections,gzip,hashlib,json,math,random,time
from pathlib import Path
H=Path(__file__).resolve().parent;R=H/'results';O=R/'verification_20261025';O.mkdir(exist_ok=True)
B=Path('/home/gcx/aster_e0_e8_20261024/private');P=Path('/home/gcx/aster_p0_p3_20261025')
def load(p):return json.loads(p.read_text())
def lines(p):return [json.loads(x) for x in p.read_text().splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def k(c):return f"{c['id']}|{c['seed']}|{c['variant']}"
def winner(x):return max(range(len(x)),key=x.__getitem__)
def summary(rr,method):
 n=len(rr);correct=sum(r[method]==r['gold'] for r in rr);harm=sum(r['none']==r['gold'] and r[method]!=r['gold'] for r in rr);corrected=sum(r['none']!=r['gold'] and r[method]==r['gold'] for r in rr);base=sum(r['none']==r['gold'] for r in rr)
 return {'n':n,'correct':correct,'accuracy':correct/n,'harmed':harm,'harm_rate':harm/n,'harm_given_baseline_correct':harm/base if base else None,'corrected':corrected,'coverage':sum(r[method+'_use'] for r in rr)/n}
def paired(rr,other):
 groups=collections.defaultdict(list)
 for r in rr:
  ac=int(r['candidate']==r['gold'])-int(r[other]==r['gold']);base=r['none']==r['gold'];hc=int(base and r['candidate']!=r['gold'])-int(base and r[other]!=r['gold']);groups[r['tree']].append((ac,hc))
 totals=[(len(g),sum(x[0] for x in g),sum(x[1] for x in g)) for t,g in sorted(groups.items())];rng=random.Random(20261025);aa=[];hh=[]
 for _ in range(10000):
  draws=rng.choices(totals,k=len(totals));n=sum(x[0] for x in draws);aa.append(sum(x[1] for x in draws)/n);hh.append(sum(x[2] for x in draws)/n)
 def ci(x):x.sort();return [x[250],x[9750]]
 return {'clusters':len(totals),'replicates':10000,'accuracy_delta':sum(x[1] for x in totals)/len(rr),'accuracy_delta_ci95':ci(aa),'harm_delta':sum(x[2] for x in totals)/len(rr),'harm_delta_ci95':ci(hh),'note':'candidate minus comparator; harm for BOTH relative to original no-intervention'}
lock=load(R/'frozen.json');allcases=lines(B/'cases.jsonl');cases={k(c):c for c in allcases if c['split']=='test'}
assert len(cases)==12288
assert not set(lock['dev_trees']) & {c['tree'] for c in cases.values()}
assert sha(H/'analysis.py')==lock['analysis_sha256'] and sha(H/'infer.py')==lock['infer_sha256']
assert sha(R/'p2_cross_validation.json')==lock['selection_report_sha256'] and sha(R/'provenance.json')==lock['provenance_sha256']
config=lock['candidate'];assert not config['agreement']
methods=['none','candidate','threshold','direct','blind_direct'];public={}
for method in methods:
 with gzip.open(R/f'p3_{method}.jsonl.gz','rt') as f:rr=[json.loads(x) for x in f]
 public[method]={(r['model'],r['key']):r for r in rr};assert len(rr)==len(public[method])==24576
old={(r['model'],r['method']):r for r in load(R/'p3_metrics.json')['results']}
audit={'verified_unix':time.time(),'frozen_sha256':sha(R/'frozen.json'),'checks':['exact expected key sets, no duplicates','all seeds/variants/tree split','frozen code, selection and data fingerprints','fresh record provenance and explicit exact-input memoization','independent selector replay agrees with published predictions','independent aggregate counts agree with published metrics'],'models':{}}
for model in ('qwen','llama'):
 d=P/'test'/model;raw=lines(d/'predictions.jsonl');manifest=load(d/'manifest.json');status=load(d/'status.json');cache=lines(d/'condition_cache.jsonl');lookup={x['condition_key']:x for x in cache}
 assert len(raw)==len({r['key'] for r in raw})==len(cases) and {r['key'] for r in raw}==set(cases)
 assert status['complete'] and status['completed']==status['total']==12288
 assert manifest['data_sha256']==sha(B/'cases.jsonl') and manifest['code_sha256']==lock['infer_sha256'] and manifest['frozen_sha256']==sha(R/'frozen.json')
 assert lock['created_unix']<status['updated_unix']-status['elapsed_seconds']
 checked=[]
 for r in raw:
  c=cases[r['key']];assert r['provenance']=='fresh_gpu_forward'
  assert all(r[x]==c[x] for x in ('id','tree','seed','variant','split'))
  payload={'patient':c['patient'],'conditions':[{'rid':x['rid'],'condition':x['condition']} for x in c['rules']]}
  ck=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest();assert ck==r['condition_key']
  assert lookup[ck]['provenance']=='fresh_gpu_forward' and lookup[ck]['p']==r['p'] and lookup[ck]['assessments']==r['assessments']
  assert winner(r['none_scores'])==r['none_prediction']
  masses=[sum(r['p'][x['rid']] for x in c['rules'] if x['action']==opt) for opt in c['options']];total=sum(masses);masses=[x/total for x in masses];direct=winner(masses)
  members=[x['rid'] for x in c['rules'] if x['action']==c['options'][direct]];best=max(members,key=lambda j:r['assessments'][j][0]);support,contra,unknown=r['assessments'][best];other=max((r['assessments'][x['rid']][0] for x in c['rules'] if x['rid'] not in members),default=0)
  use=support>=config['support'] and max(contra,unknown)<=config['uncertainty'] and support-other>=config['margin'];tu=max(masses)>=lock['simple_threshold']['threshold'];none=r['none_prediction']
  blind=winner([sum(r['blind'][x['rid']] for x in c['rules'] if x['action']==opt) for opt in c['options']])
  out={'tree':c['tree'],'seed':c['seed'],'variant':c['variant'],'gold':c['gold_option'],'none':none,'candidate':direct if use else none,'threshold':direct if tu else none,'direct':direct,'blind_direct':blind,'none_use':False,'candidate_use':use,'threshold_use':tu,'direct_use':True,'blind_direct_use':True}
  for method in methods:
   pub=public[method][model,r['key']];assert pub['pred']==out[method] and pub['gold']==out['gold'] and pub['none']==none
   if 'use' in pub:assert pub['use']==out[method+'_use']
  checked.append(out)
 subsets={name:[r for r in checked if (r['variant'] in ('NC','AC','IC','SW') if name=='core' else True if name=='all' else r['variant']==name)] for name in ('all','core','NC','AC','IC','SW','CS','CS-null')}
 summaries={s:{m:summary(rr,m) for m in methods} for s,rr in subsets.items()}
 for method in methods:
  new=summaries['all'][method];before=old[model,method]
  for a,b in [('accuracy','accuracy'),('harmed','harmed'),('corrected','corrected'),('coverage','coverage')]:assert abs(new[a]-before[b])<1e-12
 audit['models'][model]={'raw_sha256':sha(d/'predictions.jsonl'),'cache_sha256':sha(d/'condition_cache.jsonl'),'rows':len(raw),'unique_condition_inputs':len(cache),'condition_reused_rows':sum(r['condition_reused_exact_input'] for r in raw),'counts':dict(collections.Counter(f"{r['seed']}|{r['variant']}" for r in checked)),'status':status,'metrics':summaries,'paired':{s:{other:paired(rr,other) for other in ('none','threshold','direct')} for s,rr in subsets.items() if s in ('all','core','CS','CS-null')},'core_seeds':{str(seed):{m:summary([r for r in subsets['core'] if r['seed']==seed],m) for m in methods} for seed in (0,1,2)}}
(O/'verified.json').write_text(json.dumps(audit,indent=2))
for m,v in audit['models'].items():
 print(m,'core',json.dumps(v['metrics']['core']));print('comparisons',json.dumps(v['paired']['core']));print('CS',json.dumps({s:v['metrics'][s]['candidate'] for s in ('CS','CS-null')}))
