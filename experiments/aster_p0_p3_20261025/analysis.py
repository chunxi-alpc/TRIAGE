"""CPU reanalysis of recorded probabilities, with explicit inference provenance."""
import argparse,collections,csv,gzip,hashlib,itertools,json,math,random,statistics,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;OUT=HERE/'results'
BASE=Path('/home/gcx/aster_e0_e8_20261024/private');ROOT=Path('/home/gcx/aster_p0_p3_20261025')
CORE=('NC','AC','IC','SW');TEMPS=(.25,.5,1.,2.,4.)
def read(p):return [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
def key(c):return f"{c['id']}|{c['seed']}|{c['variant']}"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));tmp.replace(p)
def csvout(name,rows):
 if not rows:return
 with (OUT/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def argmax(p):return max(range(len(p)),key=lambda i:p[i])
def norm(p):
 s=sum(p);return [x/s for x in p] if s else [1/len(p)]*len(p)
def temper(p,t):
 z=[math.log(max(x,1e-30))/t for x in p];m=max(z);return norm([math.exp(x-m) for x in z])
def action(c,p,mode='sum',dedup=False,duplicate=None):
 groups=[[] for _ in c['options']];seen=set()
 for r in c['rules']:
  sig=(r['condition'],r['action'])
  if dedup and sig in seen:continue
  seen.add(sig);j=c['options'].index(r['action']);groups[j].append(p[r['rid']])
  if duplicate==r['rid']:groups[j].append(p[r['rid']])
 scores=[sum(g) if mode=='sum' else max(g,default=0) if mode=='max' else sum(g)/len(g) if g else 0 for g in groups]
 return norm(scores)
def select(c,p,assess,blind,none,cfg):
 """Deployment API accepts no target, variant, tree or split; no oracle fallback."""
 q=action(c,p);winner=argmax(q)
 if cfg['kind']=='none':return none,False
 if cfg['kind']=='direct':return winner,True
 confidence=max(q)
 if cfg['kind']=='threshold':use=confidence>=cfg['threshold']
 else:
  chosen=[r['rid'] for r in c['rules'] if c['options'][winner]==r['action']]
  best=max(chosen,key=lambda i:assess[i][0]);support,contra,unknown=assess[best]
  alternatives=[assess[r['rid']][0] for r in c['rules'] if r['rid'] not in chosen]
  use=(support>=cfg['support'] and max(contra,unknown)<=cfg['uncertainty'] and support-max(alternatives,default=0)>=cfg['margin'])
  if cfg['agreement']:use=use and argmax(action(c,blind))==winner
 return (winner if use else none),use
EVIDENCE=[{'kind':'evidence','support':s,'uncertainty':u,'margin':m,'agreement':a} for s,u,m,a in itertools.product((.7,.9),(.1,.3),(.05,.2),(False,True))]
THRESHOLDS=[{'kind':'threshold','threshold':t} for t in (.4,.5,.6,.7,.8,.9,1.01)]
def metric(rows):
 n=len(rows)
 if not n:return {'n':0}
 cell=collections.Counter((r['none']==r['gold'],r['pred']==r['gold']) for r in rows)
 good=cell[True,True]+cell[True,False];bad=cell[False,True]+cell[False,False];changed=sum(r['pred']!=r['none'] for r in rows)
 probs=[r for r in rows if 'q' in r]
 brier=sum(sum((p-(j==r['gold']))**2 for j,p in enumerate(r['q'])) for r in probs)/len(probs) if probs else None
 ece=0.
 for b in range(10):
  group=[r for r in probs if min(9,int(max(r['q'])*10))==b]
  if group:ece+=len(group)/len(probs)*abs(sum(max(r['q']) for r in group)/len(group)-sum(argmax(r['q'])==r['gold'] for r in group)/len(group))
 return {'n':n,'accuracy':(cell[True,True]+cell[False,True])/n,'baseline_accuracy':good/n,'both_correct':cell[True,True],'corrected':cell[False,True],'harmed':cell[True,False],'both_wrong':cell[False,False],'harm_all':cell[True,False]/n,'harm_given_baseline_correct':cell[True,False]/good if good else None,'correction_given_baseline_wrong':cell[False,True]/bad if bad else None,'changed_fraction':changed/n,'harm_given_changed':cell[True,False]/changed if changed else None,'coverage':sum(r.get('use',True) for r in rows)/n,'brier':brier,'ece_10_equal_width':ece if probs else None}
def bootstrap(rows,rep=10000):
 groups=collections.defaultdict(list)
 for r in rows:groups[r['tree']].append((int(r['pred']==r['gold'])-int(r['none']==r['gold']),int(r['none']==r['gold'] and r['pred']!=r['gold'])))
 totals={t:(len(g),sum(x[0] for x in g),sum(x[1] for x in g)) for t,g in groups.items()};trees=sorted(groups);rng=random.Random(20261025);deltas=[];harms=[]
 for _ in range(rep):
  picked=[totals[rng.choice(trees)] for _ in trees];n=sum(g[0] for g in picked)
  deltas.append(sum(g[1] for g in picked)/n);harms.append(sum(g[2] for g in picked)/n)
 def ci(x):x.sort();return [x[int(.025*rep)],x[min(rep-1,int(.975*rep))]]
 return {'clusters':len(trees),'replicates':rep,'delta_ci95':ci(deltas),'harm_all_ci95':ci(harms),'unit':'tree; all patients, variants and seeds retained together','warning':'Only four development trees; intervals unstable' if len(trees)<=4 else ''}
def apply(rows,cfg):
 out=[]
 for r in rows:
  pred,use=select(r['input'],r['p'],r['assess'],r['blind'],r['none'],cfg)
  out.append({**r,'pred':pred,'use':use,'q':action(r['input'],r['p']) if use else r['none_q']})
 return out
def objective(rows,cfg):
 rr=apply(rows,cfg);groups=collections.defaultdict(list)
 for r in rr:groups[r['model'],r['tree']].append(r)
 acc=statistics.mean(metric(g)['accuracy'] for g in groups.values());harm=statistics.mean(metric(g)['harm_all'] for g in groups.values());coverage=statistics.mean(metric(g)['coverage'] for g in groups.values())
 return (acc,-harm,-coverage)
def choose(rows,configs):return max(configs,key=lambda c:objective(rows,c))
def load_dev(cases):
 allrows=[];sources={}
 for model in ('qwen','llama'):
  d=BASE/'improvement_seed0_v1'/model
  tables={n:{r['key']:r for r in read(d/p)} for n,p in [('blind','branches/blind.jsonl'),('condition','branches/condition.jsonl'),('none','answers/none.jsonl')]}
  for p in [d/'branches/blind.jsonl',d/'branches/condition.jsonl',d/'answers/none.jsonl']:sources[str(p)]=sha(p)
  for c in cases:
   if c['split']!='dev' or c['seed']!=0 or c['variant'] not in CORE:continue
   k=key(c);b=tables['blind'][k];cond=tables['condition'][k];n=tables['none'][k]
   assert len(cond['assessments'])==len(c['rules'])
   allrows.append({'key':k,'model':model,'id':c['id'],'tree':c['tree'],'variant':c['variant'],'seed':c['seed'],'input':{k:c[k] for k in ('rules','options')},'gold':c['gold_option'],'gold_rule':c['gold_rule'],'none':n['prediction'],'p':cond['p'],'assess':cond['assessments'],'blind':b['p'],'none_q':temper_logits(n['scores']),'provenance':'offline_reanalysis_of_previous_real_inference'})
 return allrows,sources
def temper_logits(z):
 m=max(z);return norm([math.exp(x-m) for x in z])
def summarize_groups(rows):
 result=[]
 for model in ('qwen','llama'):
  rr=[r for r in rows if r['model']==model]
  dims={'all':lambda r:'all','tree':lambda r:r['tree'],'variant':lambda r:r['variant'],'seed':lambda r:r['seed'],'confidence':lambda r:('0-.5' if max(r['q'])<.5 else '.5-.7' if max(r['q'])<.7 else '.7-.9' if max(r['q'])<.9 else '.9-1'),'rule_count':lambda r:len(r['p'])}
  for dim,fn in dims.items():
   groups=collections.defaultdict(list)
   for r in rr:groups[fn(r)].append(r)
   for value,group in groups.items():result.append({'model':model,'stratum':dim,'value':value,**metric(group)})
 return result
def public_records(name,rows):
 fields=['key','model','id','tree','variant','seed','gold','none','pred','use','q','provenance']
 with gzip.open(OUT/name,'wt') as f:
  for r in rows:f.write(json.dumps({k:r[k] for k in fields if k in r})+'\n')
def p0(rows,cases):
 table=[];details=[]
 for model in ('qwen','llama'):
  rr=[r for r in rows if r['model']==model]
  methods=[('uniform_options',None,1,'sum'),('uniform_rules',None,1,'sum'),('action_frequency',None,1,'sum')]
  methods += [(s,s,t,m) for s in ('blind','condition','condition_ternary') for t in TEMPS for m in ('sum','max','mean')]
  methods += [('condition_deduplicated','condition_deduplicated',1,m) for m in ('sum','max','mean')]
  for label,source,t,mode in methods:
   predictions=[];ties=0;expected=0.;branch=0;flips=0
   for r in rr:
    c=r['input']
    if source=='blind':p=temper(r['blind'],t)
    elif source=='condition_ternary':p=norm([temper(x,t)[0] for x in r['assess']])
    elif source:p=temper(r['p'],t)
    else:p=[1/len(r['p'])]*len(r['p'])
    q=[1/len(c['options'])]*len(c['options']) if label=='uniform_options' else action(c,p,mode,dedup=source=='condition_deduplicated')
    best=max(q);wins=[i for i,x in enumerate(q) if abs(x-best)<1e-12];ties+=len(wins)>1;expected+=(r['gold'] in wins)/len(wins);branch+=argmax(p)==r['gold_rule']
    pred=argmax(q);predictions.append({**r,'pred':pred,'q':q})
    if source=='condition' and t==1:flips+=sum(argmax(action(c,p,mode,duplicate=i))!=pred for i in range(len(p)))/len(p)
   for variant in ('ALL',*CORE):
    group=predictions if variant=='ALL' else [r for r in predictions if r['variant']==variant]
    table.append({'model':model,'method':label,'temperature':t,'aggregation':mode,'variant':variant,**metric(group),'all_tie_rate':ties/len(rr),'all_random_tie_expected_accuracy':expected/len(rr),'all_branch_accuracy':branch/len(rr),'all_single_rule_duplication_flip_rate':flips/len(rr) if source=='condition' and t==1 else None})
  mismatch={r['id']:r for r in read(ROOT/'mismatch'/model/'predictions.jsonl')}
  for source in ('condition','blind'):
   for mode in ('sum','max','mean'):
    original=[];wrong=[]
    for r in rr:
     if r['id'] not in mismatch or (source=='blind' and r['variant']!='NC'):continue
     mm=mismatch[r['id']];p=mm['p'] if source=='condition' else mm['blind'];matched=r['p'] if source=='condition' else r['blind']
     for dst,pp in ((original,matched),(wrong,p)):
      q=action(r['input'],pp,mode);dst.append({**r,'pred':argmax(q),'q':q})
    if wrong:details.append({'model':model,'source':source,'aggregation':mode,'unique_patients':len(mismatch),'matched':metric(original),'mismatched':metric(wrong),'delta_accuracy':metric(wrong)['accuracy']-metric(original)['accuracy'],'note':'Fresh within-tree different-patient inference on NC conditions; condition probabilities are remapped across four action variants (not fresh variant inference). Blind mismatch comparison restricted to NC to hold actions/options fixed.'})
 csvout('p0_controls.csv',table);save('p0_mismatch.json',details)
 save('p0_definition.json',{'temperature':'Post-hoc probability temperature, not generation sampling. Same T applied to branch distributions for blind/condition. condition_ternary separately rescales A/B/C before normalizing support. Floored probabilities cannot recover underflowed logits.','uniform':'uniform branch mass + sum equals action frequency exactly; deterministic ties pick first option, random tie expectation also reported.','counts':'sum/max/mean, exact condition+action deduplication, single-rule duplication sensitivity. Uniform option expected chance is 1/number_options; no model inference.','mismatch':'Within-tree donor chosen by hash among different patient texts, without gold labels. Same disease tree avoids trivial cross-disease mismatch. Targets retained for diagnostic degradation only; not asserted to be donor truth.'})
def portable_inputs(name,rows):
 # Integer action identities and hashes of conditions preserve aggregation/deduplication.
 with gzip.open(OUT/name,'wt') as f:
  for r in rows:
   c=r['input'];rr={k:v for k,v in r.items() if k!='input'}
   rr['input']={'options':list(range(len(c['options']))),'rules':[{'rid':q['rid'],'action':c['options'].index(q['action']),'condition':hashlib.sha256(q['condition'].encode()).hexdigest()} for q in c['rules']]}
   f.write(json.dumps(rr)+'\n')
def develop(freeze=False):
 OUT.mkdir(exist_ok=True);cases=read(BASE/'cases.jsonl');rows,sources=load_dev(cases)
 portable_inputs('development_inputs.jsonl.gz',rows)
 p0(rows,cases);mapped=apply(rows,{'kind':'direct'});csvout('p1_strata.csv',summarize_groups(mapped));public_records('p1_cases.jsonl.gz',mapped)
 trees=sorted({r['tree'] for r in rows});folds=[];oof={'evidence':[],'threshold':[]}
 for held in trees:
  train=[r for r in rows if r['tree']!=held];test=[r for r in rows if r['tree']==held]
  f={'heldout_tree':held,'training_trees':sorted({r['tree'] for r in train})}
  for name,grid in [('evidence',EVIDENCE),('threshold',THRESHOLDS)]:
   cfg=choose(train,grid);f[name]=cfg;oof[name]+=apply(test,cfg)
  folds.append(f)
 configs={'evidence':choose(rows,EVIDENCE),'threshold':choose(rows,THRESHOLDS)}
 results=[]
 arms={'evidence_oof':oof['evidence'],'threshold_oof':oof['threshold'],'direct':mapped,'none':apply(rows,{'kind':'none'})}
 # Recorded development baselines remain reanalyses, never fresh model runs.
 for name in ('blind_direct','condition_crop','condition_soft_all','condition_hard_all'):
  arm=[]
  for model in ('qwen','llama'):
   p=BASE/'improvement_seed0_v1'/model/'answers'/f'{name}.jsonl';lookup={r['key']:r for r in read(p)}
   for r in rows:
    if r['model']==model and r['key'] in lookup:arm.append({**r,'pred':lookup[r['key']]['prediction']})
  arms[name]=arm
 for name,rr in arms.items():
  for model in ('qwen','llama'):
   group=[r for r in rr if r['model']==model]
   results.append({'method':name,'model':model,**metric(group),'cluster_inference':bootstrap(group)})
  public_records('p2_'+name+'.jsonl.gz',rr)
 save('p2_cross_validation.json',{'folds':folds,'selected_on_all_development':configs,'results':results,'selection':'Leave one tree out jointly across both models; maximize equally weighted model/tree accuracy, ties prefer less harm then less coverage. Labels used for training folds only. Inference receives rules/options/probabilities, not gold or variant.','evidence_definition':'Three-way condition judgments supply support, contradiction and insufficient-information probabilities. Require a supported rule for mapped action, low contradiction/unknown, separation from competing-action rules, optionally blind agreement. No claim of human-verified evidence spans.','candidate_grid':EVIDENCE,'threshold_grid':THRESHOLDS})
 save('provenance.json',{'base_commit':'2a4054015b2fff80242ab3a8c23d1c94a2bc75bd','data_sha256':sha(BASE/'cases.jsonl'),'source_sha256':sources,'development_rows':len(rows),'dev_trees':trees,'test_trees_overlap':bool(set(trees)&{r['tree'] for r in cases if r['split']=='test'}),'created_unix':time.time(),'old_predictions_preserved':True})
 if freeze:
  for model in ('qwen','llama'):
   status=json.loads((ROOT/'mismatch'/model/'status.json').read_text());assert status['complete'] and status['completed']==200
  lock={'created_unix':time.time(),'candidate':configs['evidence'],'simple_threshold':configs['threshold'],'mapping':'condition_T1_sum','unique_candidate':True,'dev_trees':trees,'test_labels_used_for_selection':False,'selection_report_sha256':sha(OUT/'p2_cross_validation.json'),'analysis_sha256':sha(Path(__file__)),'infer_sha256':sha(HERE/'infer.py'),'test_variants':['NC','AC','IC','SW','CS','CS-null'],'seeds':[0,1,2],'expected_per_model':12288,'provenance_sha256':sha(OUT/'provenance.json'),'mismatch_manifest_sha256':{m:sha(ROOT/'mismatch'/m/'manifest.json') for m in ('qwen','llama')}}
  if (OUT/'frozen.json').exists():raise RuntimeError('Freeze already exists; do not overwrite')
  save('frozen.json',lock)
 print(json.dumps({'development_rows':len(rows),'P2':[{'method':r['method'],'model':r['model'],'accuracy':r['accuracy'],'harm_all':r['harm_all'],'coverage':r['coverage']} for r in results[:8]]}))
def test_report():
 lock=json.loads((OUT/'frozen.json').read_text());assert sha(Path(__file__))==lock['analysis_sha256']
 cases={key(c):c for c in read(BASE/'cases.jsonl') if c['split']=='test'};rows=[];status={}
 for model in ('qwen','llama'):
  records=read(ROOT/'test'/model/'predictions.jsonl');status[model]={'completed':len(records),'expected':len(cases),'complete':len(records)==len(cases)}
  assert len({r['key'] for r in records})==len(records)
  for rec in records:
   c=cases[rec['key']];rows.append({'key':key(c),'model':model,'id':c['id'],'tree':c['tree'],'variant':c['variant'],'seed':c['seed'],'input':{k:c[k] for k in ('rules','options')},'gold':c['gold_option'],'none':rec['none_prediction'],'p':rec['p'],'assess':rec['assessments'],'blind':rec['blind'],'none_q':temper_logits(rec['none_scores']),'provenance':'fresh_gpu_forward_with_explicit_exact_input_condition_reuse'})
 save('p3_status.json',status)
 if not rows:return
 portable_inputs('test_inputs.jsonl.gz',rows)
 results=[]
 for name,cfg in [('candidate',lock['candidate']),('threshold',lock['simple_threshold']),('direct',{'kind':'direct'}),('none',{'kind':'none'})]:
  rr=apply(rows,cfg);csvout('p3_'+name+'_strata.csv',summarize_groups(rr));public_records('p3_'+name+'.jsonl.gz',rr)
  for model in ('qwen','llama'):
   group=[r for r in rr if r['model']==model]
   if group:results.append({'method':name,'model':model,**metric(group),'cluster_inference':bootstrap(group)})
 # Blind direct is a fresh shared-baseline mapping, not an additional model call.
 rr=[]
 for r in rows:
  q=action(r['input'],r['blind']);rr.append({**r,'pred':argmax(q),'q':q})
 csvout('p3_blind_direct_strata.csv',summarize_groups(rr));public_records('p3_blind_direct.jsonl.gz',rr)
 for model in ('qwen','llama'):
  group=[r for r in rr if r['model']==model]
  if group:results.append({'method':'blind_direct','model':model,**metric(group),'cluster_inference':bootstrap(group)})
 comparisons=[]
 candidate=apply(rows,lock['candidate'])
 for other,cfg in [('threshold',lock['simple_threshold']),('direct',{'kind':'direct'}),('none',{'kind':'none'})]:
  comparator=apply(rows,cfg);lookup={(r['model'],r['key']):r['pred'] for r in comparator}
  for model in ('qwen','llama'):
   for subset in ('core','CS','CS-null'):
    group=[{**r,'none':lookup[r['model'],r['key']]} for r in candidate if r['model']==model and (r['variant'] in CORE if subset=='core' else r['variant']==subset)]
    if group:comparisons.append({'candidate_vs':other,'model':model,'subset':subset,**metric(group),'cluster_inference':bootstrap(group)})
 save('p3_pairwise.json',comparisons)
 save('p3_metrics.json',{'complete':all(s['complete'] for s in status.values()),'results':results,'human_audit':'pending; do not equate automated evaluation with human review','seed_semantics':'Data construction seeds 0/1/2; deterministic model inference, not three independent model training runs.'})
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--freeze',action='store_true');ap.add_argument('--test',action='store_true');a=ap.parse_args()
 if a.test:test_report()
 else:develop(a.freeze)
