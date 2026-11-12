"""Calibration, frozen evaluation, grouped intervals, and per-record recomputation."""
import json,collections,itertools,time,math
from pathlib import Path
import numpy as np
from scipy.special import expit,logsumexp
from scipy.optimize import minimize
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
import run as r
from run import ROOT,H,B,CFG,ev,read,write,rows,lines,model,Store
REG=CFG['mega_m4_regularization']
def logistic_fit(scores,labels,mode,ids,splits):
 assert set(splits)=={'dev'},'Calibration requires development inputs'
 x=np.asarray(scores,float);y=np.asarray(labels,float)
 if mode==0:return dict(alpha=0.,beta=1.,mode=mode,training_ids=ids,split='dev')
 def unpack(z):
  if mode==1:return z[0],1.
  if mode==2:return 0.,float(np.exp(z[0]))
  return z[0],float(np.exp(z[1]))
 def objective(z):
  a,b=unpack(z);v=a+b*x
  return float(np.mean(np.logaddexp(0,v)-y*v)+REG*np.sum(z*z)/2)
 opt=minimize(objective,np.zeros(2 if mode==3 else 1),method='BFGS',options={'gtol':1e-7})
 if not np.isfinite(opt.fun):raise RuntimeError('Nonfinite calibration fit')
 a,b=unpack(opt.x);assert b>0
 return dict(alpha=float(a),beta=float(b),mode=mode,training_ids=ids,split='dev',regularization=REG,objective=float(opt.fun),optimizer_success=bool(opt.success),message=str(opt.message))
def full_models(f='full'):
 mm={g:model(g,f) for g in ['E1','E2','E3']}
 zero=dict(mm['E1'],group='E0',alpha=np.zeros_like(mm['E1']['alpha']),offset=0.,directory='raw:'+f)
 return dict(E0=zero,**mm)
def candidate_sets(item):
 letters=list(item['options']);none=[k for k in letters if 'none of' in item['options'][k].lower()]
 if none:
  other=[k for k in letters if k not in none]
  return [none]+[list(c) for size in range(1,len(other)+1) for c in itertools.combinations(other,size)],none
 return [list(c) for size in range(len(letters)+1) for c in itertools.combinations(letters,size)],[]
def candidate_data(item,ss):
 sets,none=candidate_sets(item);maxsize=len(item['options'])
 # gamma_1=0 removes the common-energy redundancy; None gets eta only.
 sums=np.array([sum(ss[k] for k in c) for c in sets])
 design=np.zeros((len(sets),7)) # eta, gamma_0, gamma_2 ... gamma_6
 for i,c in enumerate(sets):
  if none and c==none:design[i,0]=1
  elif len(c)!=1:
   k=len(c);assert k<=6
   design[i,1 if k==0 else k]=1
 gold=next(i for i,c in enumerate(sets) if set(c)==set(item['answer']))
 return dict(sets=sets,none=none,sums=sums,design=design,gold=gold)
def energy(d,z):return d['sums']*np.exp(-z[0])+d['design']@z[1:]
def set_fit(data,reg):
 def obj(z):
  loss=0.;grad=np.zeros(8)
  for d in data:
   en=energy(d,z);p=np.exp(en-logsumexp(en));p[d['gold']]-=1
   loss+=logsumexp(en)-en[d['gold']]
   grad[0]+=p@(-d['sums']*np.exp(-z[0]));grad[1:]+=p@d['design']
  return loss/len(data)+reg*(z@z)/2,grad/len(data)+reg*z
 opt=minimize(obj,np.zeros(8),jac=True,method='L-BFGS-B',bounds=[(-5,5)]+[(None,None)]*7,options={'ftol':1e-12,'maxiter':1000})
 if not opt.success:raise RuntimeError('MedPIC calibration: '+str(opt.message))
 return opt.x
def medpic_fit(store):
 spec=r.target_specs('medpic','dev');vv=spec['views'];ss=ev.score_views(store,vv,{'E3':model('E3')})['E3']
 items={i['id']:i for i in rows(ev.TARGET/'medpic/items.jsonl')}
 grouped=collections.defaultdict(dict)
 for v in vv:
  if v.get('original') and v.get('variant')=='full':grouped[v['item_id']][v['option_letter']]=v
 assert len(grouped)==93
 data=[];ids=[];groups=[]
 for q,options in sorted(grouped.items()):
  d=candidate_data(items[q],{k:ss[ev.uid(v)] for k,v in options.items()})
  d['id']=q;data.append(d);ids.append(q);groups.append(next(iter(options.values()))['source_id'])
 records=[];folds=list(GroupKFold(5).split(ids,groups=groups))
 for reg in CFG['medpic_regularization_grid']:
  losses=[]
  for f,(train,valid) in enumerate(folds):
   z=set_fit([data[i] for i in train],reg)
   ll=[float(logsumexp(energy(data[i],z))-energy(data[i],z)[data[i]['gold']]) for i in valid];losses+=ll
   records.append(dict(regularization=reg,fold=f+1,train_ids=[ids[i] for i in train],validation_ids=[ids[i] for i in valid],nll=float(np.mean(ll)),parameters=z.tolist()))
  records.append(dict(regularization=reg,fold='pooled',nll=float(np.mean(losses))))
 chosen=min([v for v in records if v['fold']=='pooled'],key=lambda v:(v['nll'],-v['regularization']))['regularization']
 z=set_fit(data,chosen)
 result=dict(z=z.tolist(),T=float(np.exp(z[0])),eta=float(z[1]),gamma={'0':float(z[2]),'1':0.,**{str(k):float(z[k+1]) for k in range(2,7)}},training_ids=ids,split='dev',selection='source-grouped five-fold set NLL',regularization=chosen,folds=records)
 write(ROOT/'calibration/medpic.json',result)
 lines(ROOT/'calibration/medpic_training_scores.jsonl',[dict(item_id=d['id'],candidate_sets=d['sets'],sums=d['sums'].tolist(),gold=d['gold']) for d in data])
 return result
ORIGINAL_SCORE=ev.score_views;ORIGINAL_DECISIONS=ev.decisions_for;ORIGINAL_OBSERVATIONS=ev.observations
CAL={};MED=None;CURRENT='';BASES={}
def score(store,views,models):
 ss=ORIGINAL_SCORE(store,views,models)
 if 'E3' in ss:
  for name,param in CAL.items():ss[name]={k:param['alpha']+param['beta']*v for k,v in ss['E3'].items()}
 BASES.clear();BASES.update(ss)
 return ss
def decisions(views,scores,task=None):
 ds,qq=ORIGINAL_DECISIONS(views,scores,task)
 if task!='medpic' or 'E4' not in scores:return ds,qq
 qq=[v for v in qq if v['method']!='E4'];items={i['id']:i for i in rows(ev.TARGET/'medpic/items.jsonl')}
 grouped=collections.defaultdict(dict)
 for v in views:grouped[(v['item_id'],v.get('variant','full'))][v['option_letter']]=v
 records=[]
 for (qid,var),opts in grouped.items():
  ss={k:scores['E3'][ev.uid(v)] for k,v in opts.items()}
  d=candidate_data(items[qid],ss);en=energy(d,np.array(MED['z']));lp=en-logsumexp(en);idx=int(np.argmax(en));chosen=d['sets'][idx]
  for k,v in opts.items():ds['E4'][ev.uid(v)]=int(k in chosen)
  v=next(iter(opts.values()))
  qq.append(dict(method='E4',item_id=qid,variant=var,source_id=v['source_id'],split=v.get('split'),predicted=chosen,gold=items[qid]['answer'],correct=set(chosen)==set(items[qid]['answer'])))
  records.append(dict(item_id=qid,variant=var,source_id=v['source_id'],option_scores=ss,candidate_sets=d['sets'],energies=en.tolist(),probabilities=np.exp(lp).tolist(),gold_index=d['gold'],prediction_index=idx,set_nll=float(-lp[d['gold']])))
 lines(ROOT/'medpic_candidate_scores'/(CURRENT+'.jsonl'),records)
 return ds,qq
def observations(views,pairs,scores,decisions,native=False,budgets=None):
 # M4 decisions are evaluated, while mechanism metrics remain measurements of the frozen M3 scores.
 mech=dict(scores)
 for k in CAL:
  if k in mech:mech[k]=scores['E3']
 return ORIGINAL_OBSERVATIONS(views,pairs,mech,decisions,native,budgets)
def mega_calibrate(store,f='full'):
 vv=[v for v in r.mega_specs('dev')['views'] if v.get('original') and v.get('label') is not None]
 ss=ORIGINAL_SCORE(store,vv,{'E3':model('E3',f)})['E3']
 out={f'M4-{m}':logistic_fit([ss[ev.uid(v)] for v in vv],[v['label'] for v in vv],m,[ev.uid(v) for v in vv],[v['split'] for v in vv]) for m in range(4)}
 write(ROOT/'calibration'/('mega_'+f+'.json'),out)
 lines(ROOT/'calibration'/('mega_'+f+'_training_scores.jsonl'),[dict(view_id=ev.uid(v),source_id=v['source_id'],split='dev',label=v['label'],score=ss[ev.uid(v)]) for v in vv])
 return out
def evaluate(scope,spec,models,cal=None):
 global CURRENT,CAL
 CURRENT=scope;CAL=cal or {}
 # Model refs only; calibration does not save into or mutate M3 files.
 for k in CAL:models[k]=dict(models['E3'],directory=models['E3']['directory'])
 result=ev.scope_evaluate(scope,store=STORE,models=models,out=ROOT,**spec)
 # Save path-node scores as well, enabling exact path replay without inference.
 lines(ROOT/'predictions'/(scope+'_scoring.jsonl'),[dict(view_id=k,scores={m:ss[k] for m,ss in BASES.items()}) for k in BASES['raw']])
 lines(ROOT/'predictions'/(scope+'_path_cases.jsonl'),spec.get('cases',[]))
 lines(ROOT/'predictions'/(scope+'_path_views.jsonl'),spec.get('path_views',[]))
 return result
def metrics_from_records(scope):
 folder=ROOT/'evaluation'/scope
 pred=rows(folder/'predictions.jsonl');pairs=rows(folder/'pair_predictions.jsonl');paths=rows(folder/'path_predictions.jsonl');qs=rows(folder/'question_predictions.jsonl')
 sources=[v['source_id'] for v in pred]+[v['source_id'] for v in paths]
 boot=ev.SourceBootstrap(sources)
 methods=sorted(pred[0]['scores']) if pred else []
 methods=[m for m in methods if m!='raw']
 series={};table=[];denoms=[]
 native='medpic' in scope or 'nli4ct' in scope
 for m in methods:
  pp=[p for p in pairs if p['method']==m];mm,counts,_=ev.pair_statistics(pp,boot,native=native)
  nn=[p for p in pred if p['original'] and p['label'] is not None]
  if nn:
   mm['natural_node_accuracy']=boot.mean([p['predictions'][m]==p['label'] for p in nn],[p['source_id'] for p in nn]);counts['natural_node_count']=len(nn)
  for interference in [False,True]:
   pp=[p for p in paths if p['method']==m and p['interference']==interference]
   if pp:
    key=('interference' if interference else 'natural')+'_path_accuracy'
    mm[key]=boot.mean([p['correct'] for p in pp],[p['source_id'] for p in pp]);counts[key+'_denominator']=len(pp)
  if 'medpic' in scope:
   qq=[p for p in qs if p['method']==m and p['variant'] in ['full','original']]
   for name,part in [('question_accuracy',qq),('single_choice_accuracy',[p for p in qq if len(p['gold'])==1]),('multiple_choice_accuracy',[p for p in qq if len(p['gold'])>1]),('none_accuracy',[p for p in qq if any('none of' in ITEMS[p['item_id']]['options'][k].lower() for k in p['gold'])])]:
    mm[name]=boot.mean([p['correct'] for p in part],[p['source_id'] for p in part]);counts[name+'_denominator']=len(part)
   mm['answer_set_size_mae']=boot.mean([abs(len(p['predicted'])-len(p['gold'])) for p in qq],[p['source_id'] for p in qq])
  if 'nli4ct' in scope:
   if nn:mm['original_macro_f1']=macro_boot(nn,m,boot)
   labeled=[p for p in pred if p['label'] is not None]
   mm['formal_test_accuracy']=boot.mean([p['predictions'][m]==p['label'] for p in labeled],[p['source_id'] for p in labeled]);counts['formal_test_denominator']=len(labeled)
   changing=[p for p in pairs if p['method']==m and p['group']=='native_changing']
   src=[p['source_id'] for p in changing]
   eligible=[int((p['left_score']>0)==p['left_label']) for p in changing]
   both=[int(p['paired_correct']) for p in changing]
   mm['conditional_flip_faithfulness']=ev.ratio(boot.sums(both,src),boot.sums(eligible,src))
   counts['conditional_flip_eligible']=sum(eligible)
   counts['conditional_flip_success']=sum(both)

  if not native and nn:
   y=np.array([p['label'] for p in nn]);x=np.array([p['scores'][m] for p in nn]);prob=expit(x);src=[p['source_id'] for p in nn]
   mm['brier']=boot.mean((prob-y)**2,src);mm['log_loss']=boot.mean(np.logaddexp(0,x)-y*x,src)
   # Fixed 10 equal-width bins, never threshold tuning.
   bins=np.minimum((prob*10).astype(int),9);ece=np.zeros(boot.draws+1);n=boot.sums(np.ones(len(y)),src);curves=[]
   for b in range(10):
    mask=bins==b
    if mask.any():
     sb=[s for s,keep in zip(src,mask) if keep];nr=boot.sums(np.ones(sum(mask)),sb)
     ece+=np.abs(boot.sums(prob[mask]-y[mask],sb))
     curves.append(dict(bin=b,n=int(sum(mask)),mean_probability=float(prob[mask].mean()),fraction_positive=float(y[mask].mean())))
   mm['ece']=ev.ratio(ece,n)
   write(ROOT/'calibration'/(scope+'_'+m+'_curve.json'),curves)
   # Diagnostic intercept/slope on evaluation labels: reporting only, never used for predictions.
   fit=minimize(lambda z:np.mean(np.logaddexp(0,z[0]+z[1]*x)-y*(z[0]+z[1]*x)),[0.,1.],method='BFGS')
   counts['calibration_intercept']=float(fit.x[0]);counts['calibration_slope']=float(fit.x[1])
  series[m]=mm
  for metric,v in mm.items():table.append(dict(scope=scope,method=m,metric=metric,value=ev.finite(v[0])))
  for metric,v in counts.items():table.append(dict(scope=scope,method=m,metric=metric,value=v))
 contrasts=[('E2','E1'),('E3','E2'),('E4','E3'),('E3','E1'),('E4','E1')]+[(m,'E0') for m in methods if m!='E0']
 bb=[]
 for m,mm in series.items():
  for metric,v in mm.items():bb.append(dict(scope=scope,comparison=m,metric=metric,**boot.summary(v),draws=2000))
 for a,b in contrasts:
  if a not in series or b not in series:continue
  for metric in series[a].keys()&series[b].keys():bb.append(dict(scope=scope,comparison=a+' minus '+b,metric=metric,**boot.summary(series[a][metric]-series[b][metric]),draws=2000))
 transitions=[]
 for a,b in contrasts:
  if a not in series or b not in series:continue
  for unit,records in [('node',[dict(id=p['view_id'],source_id=p['source_id'],a=p['predictions'][a]==p['label'],b=p['predictions'][b]==p['label']) for p in pred if p['label'] is not None])]:
   for p in records:transitions.append(dict(scope=scope,comparison=a+' minus '+b,unit=unit,**p,corrected=int(p['a'] and not p['b']),new_error=int(p['b'] and not p['a'])))
  for unit,rr,keys in [('path',paths,['case_id','variant']),('question',qs,['item_id','variant'])]:
   left={tuple(p[k] for k in keys):p for p in rr if p['method']==a};right={tuple(p[k] for k in keys):p for p in rr if p['method']==b}
   for key in left.keys()&right.keys():
    aa,bbb=left[key]['correct'],right[key]['correct']
    transitions.append(dict(scope=scope,comparison=a+' minus '+b,unit=unit,id=str(key),source_id=left[key]['source_id'],a=aa,b=bbb,corrected=int(aa and not bbb),new_error=int(bbb and not aa)))
 return table,bb,transitions
def macro_boot(nn,m,boot):
 y=np.array([p['label'] for p in nn]);pr=np.array([p['predictions'][m] for p in nn]);src=[p['source_id'] for p in nn]
 vals=[]
 for label in [0,1]:
  tp=boot.sums((y==label)&(pr==label),src);fp=boot.sums((y!=label)&(pr==label),src);fn=boot.sums((y==label)&(pr!=label),src)
  den=2*tp+fp+fn;vals.append(np.divide(2*tp,den,out=np.zeros_like(tp),where=den>0))
 return np.mean(vals,axis=0)
def pool_cv():
 dest=ROOT/'evaluation/cv_pooled_common';dest.mkdir(parents=True,exist_ok=True)
 for name in ['predictions','pair_predictions','path_predictions','question_predictions']:
  lines(dest/(name+'.jsonl'),[p for i in range(1,6) for p in rows(ROOT/f'evaluation/cv_fold{i}'/(name+'.jsonl'))])
def recompute():
 tables=[];boots=[];trans=[]
 for folder in sorted((ROOT/'evaluation').iterdir()):
  if not (folder/'predictions.jsonl').exists():continue
  t,b,c=metrics_from_records(folder.name);tables+=t;boots+=b;trans+=c
 ev.write_csv(ROOT/'metrics.csv',tables);ev.write_csv(ROOT/'source_bootstrap.csv',boots);ev.write_csv(ROOT/'transitions.csv',trans)
 return tables,boots,trans
def verify():
 tests=[]
 def check(name,ok,**details):
  tests.append(dict(name=name,passed=bool(ok),**details))
  if not ok:raise AssertionError(name)
 for f in ['full']+['fold'+str(i) for i in range(1,6)]:
  p=r.fdir('E3',f);z=dict(np.load(p/'m3.npz'));meta=read(p/'model.json')
  check('M3_no_intercept_'+f,'b' not in z and not meta['global_intercept'])
  check('M3_pairs_unchanged_'+f,max(meta['validation']['pair_difference_max_error'].values())<1e-12)
  check('M3_CF_constraints_'+f,meta['validation']['cf_max_violation']<=1e-7)
  check('M3_train_center_'+f,abs(meta['validation']['centering_mean'])<1e-12 and set(z['centering_ids']).issubset({p[k] for p in r.train_pairs() if p['source_id'] in meta['training_sources'] for k in ['left','right']}))
  if f!='full':
   fold=next(v for v in r.folds() if 'fold'+str(v['fold'])==f)
   check('independent_fold_'+f,not set(meta['training_sources'])&set(fold['validation_sources']) and read(r.fdir('E2',f)/'m1_plan.json')['training_sources']==meta['training_sources'])
  check('M4_M3_unchanged_'+f,BEFORE[f]==(p/'m3.npz').read_bytes())
 for path in (ROOT/'calibration').glob('mega_*.json'):
  if '_training_' in path.name:continue
  d=read(path)
  if not isinstance(d,dict):continue
  check('MEGA_dev_only_'+path.stem,all(v['split']=='dev' and v['beta']>0 for v in d.values()))
 check('MedPIC_dev_only',set(MED['training_ids'])=={v['item_id'] for v in r.target_specs('medpic','dev')['views']})
 for path in (ROOT/'medpic_candidate_scores').glob('*.jsonl'):
  for row in rows(path):
   d=candidate_data(ITEMS[row['item_id']],row['option_scores']);en=energy(d,np.array(MED['z']))
   assert d['sets']==row['candidate_sets']
   assert np.allclose(en,row['energies'],atol=1e-10)
   assert row['prediction_index']==int(np.argmax(en))
  check('MedPIC_legal_sets_and_recomputed_energy_'+path.stem,True)
 check('NLI4CT_no_calibration',CFG['nli4ct_calibration'] is False)
 # Replay path traversal from stored node scores and frozen case records.
 for path in (ROOT/'predictions').glob('*_path_cases.jsonl'):
  scope=path.name[:-len('_path_cases.jsonl')];cases=rows(path)
  if not cases:continue
  sc=rows(ROOT/'predictions'/(scope+'_scoring.jsonl'));ss={m:{v['view_id']:v['scores'][m] for v in sc} for m in sc[0]['scores']}
  replay=ev.path_predictions(cases,rows(ROOT/'predictions'/(scope+'_path_views.jsonl')),ss)
  saved=rows(ROOT/'evaluation'/scope/'path_predictions.jsonl')
  check('path_replay_'+scope,replay==saved)

 # Check saved decisions against their declared scores and structured candidate model.
 for folder in sorted((ROOT/'evaluation').iterdir()):
  if folder.name=='cv_pooled_common':continue
  pred=rows(folder/'predictions.jsonl')
  if not pred:continue
  if 'medpic' in folder.name:
   questions=rows(folder/'question_predictions.jsonl')
   score_by_id={p['view_id']:p['scores'] for p in pred}
   vv=r.target_specs('medpic')['views'];options=collections.defaultdict(dict)
   for v in vv:
    if ev.uid(v) in score_by_id:options[(v['item_id'],v.get('variant','full'))][v['option_letter']]=v
   for q in questions:
    item=ITEMS[q['item_id']];oo=options[(q['item_id'],q['variant'])]
    ss={k:score_by_id[ev.uid(v)]['E3' if q['method']=='E4' else q['method']] for k,v in oo.items()}
    if q['method']=='E4':
     d=candidate_data(item,ss);chosen=d['sets'][int(np.argmax(energy(d,np.array(MED['z']))))]
    else:chosen=ev.decode_medpic(item,ss)
    assert set(chosen)==set(q['predicted'])
    assert q['correct']==(set(chosen)==set(q['gold']))
  else:
   for row in pred:
    assert all(row['predictions'][m]==int(value>0) for m,value in row['scores'].items())
  # Independent core-metric equality with the original evaluator's saved result.
  saved=read(folder/'metrics.json').get('models',{})
  calculated,_,_=metrics_from_records(folder.name)
  for row in calculated:
   expected=saved.get(row['method'],{}).get(row['metric'])
   if isinstance(expected,(int,float)) and row['value'] is not None:
    assert np.isclose(expected,row['value'],atol=1e-10), (folder.name,row,expected)
  check('predictions_and_metrics_recomputed_'+folder.name,True)
 write(ROOT/'status/tests.json',dict(passed=True,tests=tests))

def figures():
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 for scope in ['test','cv_pooled_common']:
  fig,ax=plt.subplots(figsize=(5,4));ax.plot([0,1],[0,1],color='gray',ls='--')
  for m in ['E0','E1','E2','E3','E4']:
   data=read(ROOT/'calibration'/(scope+'_'+m+'_curve.json'),[])
   if data:ax.plot([v['mean_probability'] for v in data],[v['fraction_positive'] for v in data],marker='o',label=m)
  ax.set(xlabel='Mean predicted probability',ylabel='Observed positive fraction');ax.legend();fig.tight_layout()
  for ext in ['png','pdf']:fig.savefig(ROOT/'figures'/(scope+'_calibration.'+ext),dpi=200)
  plt.close(fig)
def run():
 global STORE,MED,ITEMS,BEFORE,CAL
 STORE=Store();ITEMS={v['id']:v for v in rows(ev.TARGET/'medpic/items.jsonl')}
 BEFORE={f:(r.fdir('E3',f)/'m3.npz').read_bytes() for f in ['full']+['fold'+str(i) for i in range(1,6)]}
 MED=medpic_fit(STORE);mega={f:mega_calibrate(STORE,f) for f in BEFORE}
 ev.score_views=score;ev.decisions_for=decisions;ev.observations=observations
 for scope in ['dev','test','new60']:
  r.state('evaluation_'+scope)
  spec=r.mega_specs(scope) if scope!='new60' else dict(views=ev.data_rows(H,'new60_views.jsonl'),pairs=ev.data_rows(H,'new60_pairs.jsonl'),path_views=ev.data_rows(H,'new60_views.jsonl'),cases=[c for c in ev.case_rows(ev.TRACE/'data/cases_final.jsonl') if c.get('variant')=='original'])
  cal=dict(mega['full'],E4=mega['full']['M4-3'])
  evaluate(scope,spec,full_models(),cal)
 for f in r.folds():
  name='fold'+str(f['fold']);r.state('evaluation_'+name)
  evaluate('cv_'+name,r.cv_specs(f),full_models(name),dict(mega[name],E4=mega[name]['M4-3']))
 pool_cv()
 for split in ['dev','test']:
  mm=full_models()
  for g in ['rank8','rank32','without_M1','without_M2_projection','without_constraints']:
   if g.startswith('rank') and split=='test':continue
   mm[g]=model(g)
  unc=dict(mm['E3']);unc['offset']+=float(np.load(r.fdir('E3')/'m3.npz')['mu_c']);unc['directory']+=':uncentered';mm['without_centering']=unc
  mm['without_M3']=dict(mm['E0'])
  evaluate('ablation_'+split,r.mega_specs(split),mm)
 for task,split,scope in [('nli4ct','test','nli4ct_test'),('medpic','dev','medpic_dev'),('medpic','test','medpic_test'),('medpic',None,'medpic_official467')]:
  r.state('evaluation_'+scope)
  evaluate(scope,r.target_specs(task,split),full_models(),{'E4':{'alpha':0.,'beta':1.}})
 # E4 for NLI is identically E3: no target calibration.
 t,b,c=recompute();verify();figures()
 import report
 report.finish(t,b,c)
if __name__=='__main__':
 STORE=Store();ITEMS={v['id']:v for v in rows(ev.TARGET/'medpic/items.jsonl')}
 recompute()
