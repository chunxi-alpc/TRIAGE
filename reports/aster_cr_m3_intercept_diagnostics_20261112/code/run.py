import os,sys,json,gzip,itertools,time,shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit,logsumexp
REPO=Path(__file__).resolve().parents[3];sys.path.insert(0,str(REPO))
from experiments.aster_cr_diagnostics_e3cv_20261111 import evaluate as old
from experiments.aster_cr_diagnostics_e3cv_20261111.common import BASE
from experiments.aster_cr_retrain_paired_v2_20261111 import evaluation as ev
from experiments.aster_cr_retrain_paired_v2_20261111.common import Store
OUT=Path(__file__).resolve().parents[1];OLD=old.OUT
read=ev.read;rows=ev.rows;write=ev.write;uid=ev.uid
CONFIG=read(OUT/'configs/preregistered.json');SCALES=CONFIG['scales']
ALL_METRICS=[];ALL_BOOT=[];ALL_TRANS=[]
def save(name,records):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
 with gzip.open(str(p)+'.gz','wt',encoding='utf-8') as f:
  for r in records:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
def load(name):
 with gzip.open(OUT/(name+'.gz'),'rt',encoding='utf-8') as f:return [json.loads(x) for x in f]
def fit(x,y):
 x=np.asarray(x);y=np.asarray(y);fun=lambda b:float(np.mean(expit(x+b)-y));lo=-1.;hi=1.
 while fun(lo)>0:lo*=2
 while fun(hi)<0:hi*=2
 return float(brentq(fun,lo,hi,xtol=1e-12))
def nll(x,y):return float(np.mean(np.logaddexp(0,x)-np.asarray(y)*x))
def pairs_to_compare(names):
 c=[('pair_only','raw'),('pair_only','independent_intercept'),('full','independent_intercept'),('full','pair_only')]
 c += [(n,b) for n in names for b in ['raw','pair_only','full','independent_intercept'] if n!=b]
 return list(dict.fromkeys((a,b) for a,b in c if a in names and b in names))
def outcomes(nodes,pairs,paths,questions):
 o=defaultdict(dict)
 def add(level,key,source,val,method):
  if val is not None:o[level].setdefault(method,{})[key]=(source,bool(val))
 for n in nodes:
  if n['natural'] and n['label'] is not None:
   for m,v in n['predictions'].items():add('natural_node_accuracy',n['view_id'],n['source_id'],v==n['label'],m)
 for p in paths:add('interference_path_accuracy' if p['interference'] else 'natural_path_accuracy',p['case_id']+'|'+p['variant'],p['source_id'],p['correct'],p['method'])
 for p in pairs:
  for field in ['paired_correct','relation_correct']:add(p['group'].lower()+'_'+field,p['pair_id'],p['source_id'],p[field],p['method'])
 for q in questions:
  if q['variant'] not in ['full','original']:continue
  strata=['all','gold_size_'+str(len(q['gold'])),'single' if len(q['gold'])==1 else 'multiple','gold_none' if q['gold_none'] else 'gold_substantive','has_none' if q['none_options'] else 'no_none']
  for s in strata:add('question_accuracy/'+s,q['item_id'],q['source_id'],q['correct'],q['method'])
 return o

def summarize(scope,nodes,pairs,paths,questions):
 names=list(nodes[0]['scores']);boot=ev.SourceBootstrap([n['source_id'] for n in nodes],draws=2000)
 metrics={};tables=[];by=defaultdict(list)
 for r in pairs:by[r['method']].append(r)
 for m in names:
  mm,cc,_=ev.pair_statistics(by[m],boot,native=scope.startswith('medpic'))
  metrics[m]=mm;tables.append(dict(scope=scope,method=m,**cc))
 oo=outcomes(nodes,pairs,paths,questions)
 for level,methods in oo.items():
  for m,rec in methods.items():
   rr=list(rec.values());metrics[m][level]=boot.mean([x[1] for x in rr],[x[0] for x in rr])
   table=next(t for t in tables if t['method']==m);table[level+'_n']=len(rr);table[level+'_correct']=sum(x[1] for x in rr)
 for t in tables:t.update({k:ev.finite(v[0]) for k,v in metrics[t['method']].items()})
 br=[];tr=[]
 def badd(comp,metric,vals):br.append(dict(scope=scope,comparison=comp,metric=metric,**boot.summary(vals),sources=len(boot.sources),draws=2000))
 for m,mm in metrics.items():
  for k,v in mm.items():badd(m,k,v)
 for a,b in pairs_to_compare(names):
  comp=a+' minus '+b
  for k in metrics[a].keys()&metrics[b].keys():badd(comp,k,metrics[a][k]-metrics[b][k])
  for level,methods in oo.items():
   ar,bb=methods.get(a,{}),methods.get(b,{})
   ids=sorted(ar.keys()&bb.keys())
   if not ids:continue
   sources=[ar[k][0] for k in ids];wr=[not bb[k][1] and ar[k][1] for k in ids];rw=[bb[k][1] and not ar[k][1] for k in ids]
   tr.append(dict(scope=scope,comparison=comp,level=level,valid_denominator=len(ids),wrong_to_right=sum(wr),right_to_wrong=sum(rw),kept_right=sum(ar[k][1] and bb[k][1] for k in ids),kept_wrong=sum(not ar[k][1] and not bb[k][1] for k in ids)))
   for tag,values in [('wrong_to_right',wr),('right_to_wrong',rw)]:badd(comp,level+'/'+tag,boot.sums(values,sources))
 ev.write_csv(OUT/'predictions'/scope/'metrics.csv',tables)
 ev.write_csv(OUT/'predictions'/scope/'source_bootstrap.csv',br)
 ev.write_csv(OUT/'predictions'/scope/'transitions.csv',tr)
 ALL_METRICS.extend(tables);ALL_BOOT.extend(br);ALL_TRANS.extend(tr)

def save_scope(scope,nodes,pairs,paths,questions):
 for name,rs in [('nodes',nodes),('pairs',pairs),('paths',paths),('questions',questions)]:save('predictions/'+scope+'/'+name+'.jsonl',rs)
 summarize(scope,nodes,pairs,paths,questions)
 print('complete',scope,flush=True)

def calibrate():
 store=Store(BASE);intercepts=read(OLD/'independent_intercepts.json');devnodes=rows(OLD/'evaluation/dev/predictions.jsonl');dn=[n for n in devnodes if n['natural'] and n['label'] is not None]
 dv,_,_,_,_=old.scope_data('dev');byid={uid(v):v for v in dv};fitviews=[byid[n['view_id']] for n in dn];y=np.array([n['label'] for n in dn]);fits={};grid=[]
 for fold in ['full']+['fold'+str(i) for i in range(1,6)]:
  model=ev.model_parameters(BASE,'E1',fold);b=model['b']
  if fold=='full':x=np.array([n['scores']['E1_pair_only'] for n in dn])
  else:
   sc=ev.score_views(store,fitviews,{'E1':model});x=np.array([sc['E1'][n['view_id']]-b for n in dn])
  losses=[nll(x+s*b,y) for s in SCALES];s=SCALES[int(np.argmin(losses))];bd=fit(x,y)
  fits[fold]=dict(b=b,b0=intercepts[fold]['b0'],selected_scale=s,b_dev=bd,development_sources=sorted(set(n['source_id'] for n in dn)),development_ids=[n['view_id'] for n in dn],independent_intercept_training_sources=intercepts[fold]['fit_sources'])
  save('configs/'+fold+'_development_scores.jsonl',[dict(view_id=n['view_id'],source_id=n['source_id'],label=int(n['label']),pair_score=float(xx)) for n,xx in zip(dn,x)])
  for scale,loss in zip(SCALES,losses):grid.append(dict(fold=fold,scale=scale,intercept=scale*b,natural_node_nll=loss,natural_node_accuracy=float(np.mean((x+scale*b>0)==y)),selected=scale==s))
 write(OUT/'configs/intercepts.json',fits);ev.write_csv(OUT/'configs/intercept_scale_selection.csv',grid)
 # Original MedPIC development split only. No target model training.
 dv=[v for v in rows(BASE/'data/medpic_views.jsonl') if v['split']=='dev'];model=ev.model_parameters(BASE,'E1');sc=ev.score_views(store,dv,{'E1':model})
 save('configs/medpic_dev_option_scores.jsonl',[dict(view_id=uid(v),raw=sc['raw'][uid(v)],pair_only=sc['E1'][uid(v)]-model['b']) for v in dv])
 shutil.copyfile(OLD/'independent_intercepts.json',OUT/'configs/independent_intercepts_reused.json');shutil.copyfile(BASE/'folds.json',OUT/'configs/folds.json')
 return fits

def node_variants(oldnode,cal):
 ss=oldnode['scores'];raw=ss['raw'];p=ss['E1_pair_only'];out=dict(raw=raw,pair_only=p,original_intercept_only=ss['E1_original_intercept_only'],independent_intercept=ss['independent_intercept'],full=ss['E1_full'])
 for s in SCALES:out['scale_'+str(s)]=p+s*cal['b']
 out.update(pair_plus_b0=p+cal['b0'],dev_selected_scale=p+cal['selected_scale']*cal['b'],dev_refit=p+cal['b_dev'])
 return out

def source_eval(scope,cal,fold=None):
 oldscope=scope
 data=old.scope_data('common',fold) if fold else old.scope_data(scope)
 vv,pp,pv,cases,_=data
 orig=rows(OLD/'evaluation'/oldscope/'predictions.jsonl');nodes=[]
 for n in orig:
  ss=node_variants(n,cal);nodes.append(dict(n,scores=ss,predictions={k:int(v>0) for k,v in ss.items()}))
 scores={m:{n['view_id']:n['scores'][m] for n in nodes} for m in nodes[0]['scores']}
 dec={m:{n['view_id']:n['predictions'][m] for n in nodes} for m in scores}
 prs=ev.observations(vv,pp,scores,dec);paths=ev.path_predictions(cases,pv,scores)
 if any(p['error'] for p in paths):raise ValueError('Path replay missing score')
 save_scope(scope,nodes,prs,paths,[])
 return nodes,prs,paths

def candidates(item):
 letters=list(item['options']);none=[k for k in letters if 'none of' in item['options'][k].lower()]
 if none:
  other=[k for k in letters if k not in none];cc=[none]+[list(c) for n in range(1,len(other)+1) for c in itertools.combinations(other,n)]
 else:cc=[list(c) for n in range(len(letters)+1) for c in itertools.combinations(letters,n)]
 return cc,none

def med_data(split):
 vv=[v for v in rows(BASE/'data/medpic_views.jsonl') if v['split']==split];pp=[p for p in rows(BASE/'data/medpic_pairs.jsonl') if p['split']==split]
 if split=='dev':sr={r['view_id']:r for r in load('configs/medpic_dev_option_scores.jsonl')}
 else:sr={n['view_id']:dict(raw=n['scores']['raw'],pair_only=n['scores']['E1_pair_only']) for n in rows(OLD/'evaluation/medpic/predictions.jsonl')}
 grouped=defaultdict(dict)
 for v in vv:grouped[(v['item_id'],v['variant'])][v['option_letter']]=v
 items={r['id']:r for r in rows(ev.TARGET/'medpic/items.jsonl')};qs=[]
 for (qid,variant),opts in grouped.items():
  item=items[qid];cc,none=candidates(item);sample=next(iter(opts.values()));raw=np.array([sum(sr[uid(opts[l])]['raw'] for l in c) for c in cc]);pair=np.array([sum(sr[uid(opts[l])]['pair_only'] for l in c) for c in cc]);ks=np.array([len(c) for c in cc]);isnone=np.array([bool(none and set(c)==set(none)) for c in cc],float)
  qs.append(dict(item_id=qid,variant=variant,source_id=sample['source_id'],gold=item['answer'],candidates=cc,none=none,raw=raw,pair=pair,k=ks,isnone=isnone,options=opts))
 return vv,pp,sr,qs

def fit_set(qs,feature):
 qq=[q for q in qs if q['variant'] in ['full','original']]
 def objective_derivative(t):
  terms=[]
  for q in qq:
   f=q[feature];ss=q['pair']+t*f;prob=np.exp(ss-logsumexp(ss));gold=next(i for i,c in enumerate(q['candidates']) if set(c)==set(q['gold']));terms.append(float(prob@f-f[gold]))
  return float(np.mean(terms))
 lo=-1.;hi=1.
 while objective_derivative(lo)>0 and abs(lo)<1e6:lo*=2
 while objective_derivative(hi)<0 and hi<1e6:hi*=2
 return float(brentq(objective_derivative,lo,hi,xtol=1e-12))

def set_scores(q,cal,dc):
 k=q['k'];p=q['pair'];r=q['raw'];full=p+k*cal['b']
 return dict(raw=r,pair_only=p,original_intercept_only=r+k*cal['b'],independent_intercept=r+k*cal['b0'],full=full,C_mean_full=np.divide(full,k,out=np.zeros(len(k)),where=k>0),D_none_calibrated=p+dc['eta']*q['isnone'],E_size_calibrated=p+dc['tau']*k)

def med_calibrate(cal):
 _,_,_,qs=med_data('dev');dc=dict(eta=fit_set(qs,'isnone'),tau=fit_set(qs,'k'),dev_item_ids=[q['item_id'] for q in qs if q['variant']=='full'],dev_sources=sorted(set(q['source_id'] for q in qs)))
 names=['pair_only','C_mean_full','D_none_calibrated','E_size_calibrated','full'];acc={};loss={}
 for m in names:
  yy=[];ll=[]
  for q in qs:
   if q['variant']!='full':continue
   ss=set_scores(q,cal,dc)[m];pred=q['candidates'][int(np.argmax(ss))];g=next(i for i,c in enumerate(q['candidates']) if set(c)==set(q['gold']));yy.append(set(pred)==set(q['gold']));ll.append(float(logsumexp(ss)-ss[g]))
  acc[m]=float(np.mean(yy));loss[m]=float(np.mean(ll))
 dc.update(development_accuracy=acc,development_set_nll=loss,selected_decoder=max(names,key=lambda m:acc[m]))
 write(OUT/'configs/medpic_decoder_calibration.json',dc);return dc

def med_eval(split,cal,dc):
 vv,pp,sr,qs=med_data(split);scope='medpic_'+split;names=list(set_scores(qs[0],cal,dc));questions=[];candidate_rows=[];changes=[];dec={m:{} for m in names};qsby={}
 for q in qs:
  qid=q['item_id'];variant=q['variant'];ss=set_scores(q,cal,dc);ranks={m:np.argsort(np.argsort(-s,kind='stable'),kind='stable')+1 for m,s in ss.items()};pred={m:q['candidates'][int(np.argmax(s))] for m,s in ss.items()};qsby[(qid,variant)]=q
  for m in names:
   for l,v in q['options'].items():dec[m][uid(v)]=int(l in pred[m])
   questions.append(dict(method=m,item_id=qid,variant=variant,source_id=q['source_id'],split=split,predicted=pred[m],gold=q['gold'],correct=set(pred[m])==set(q['gold']),gold_none=bool(set(q['gold'])&set(q['none'])),none_options=q['none'],none_selected=bool(set(pred[m])&set(q['none'])),selected_option_count=len(pred[m])))
  for i,c in enumerate(q['candidates']):candidate_rows.append(dict(item_id=qid,variant=variant,source_id=q['source_id'],candidate=c,k=int(q['k'][i]),none=bool(q['isnone'][i]),gold=set(c)==set(q['gold']),scores={m:float(s[i]) for m,s in ss.items()},ranks={m:int(rr[i]) for m,rr in ranks.items()},original_intercept_shift=float(q['k'][i]*cal['b']),common_set_intercept_score=float(q['pair'][i]+cal['b'])))
  a,b=pred['pair_only'],pred['full'];changes.append(dict(item_id=qid,variant=variant,source_id=q['source_id'],changed=a!=b,before=a,after=b,k_before=len(a),k_after=len(b),none_before=bool(set(a)&set(q['none'])),none_after=bool(set(b)&set(q['none'])),rank_changes=sum(ranks['pair_only']!=ranks['full']),same_size_order_unchanged=all(np.sign(q['pair'][i]-q['pair'][j])==np.sign(ss['full'][i]-ss['full'][j]) for i in range(len(q['k'])) for j in range(i) if q['k'][i]==q['k'][j])))
 scores={m:{} for m in names};nodes=[]
 for v in vv:
  k=uid(v);r=sr[k]['raw'];p=sr[k]['pair_only'];vals=dict(raw=r,pair_only=p,original_intercept_only=r+cal['b'],independent_intercept=r+cal['b0'],full=p+cal['b'],C_mean_full=p+cal['b'],D_none_calibrated=p,E_size_calibrated=p)
  for m,x in vals.items():scores[m][k]=x
  nodes.append(dict(view_id=k,source_id=v['source_id'],label=v['label'],natural=bool(v.get('original')),variant=v['variant'],item_id=v['item_id'],option_letter=v['option_letter'],scores=vals,predictions={m:dec[m][k] for m in names}))
 prs=ev.observations(vv,pp,scores,dec,native='medpic');vby={uid(v):v for v in vv};qby={(q['method'],q['item_id'],q['variant']):q for q in questions};qp=[];seen=set();prchanges=[]
 for p in pp:
  if p['kind']=='P':continue
  l=vby[p['left']];r=vby[p['right']];key=(l['item_id'],r['item_id']);g='preserving' if p['left_label']==p['right_label'] else 'changing'
  prchanges.append(dict(pair_id=p['pair_id'],source_id=p['source_id'],option_relation=g,left_item=l['item_id'],right_item=r['item_id'],left_answer_changed=qby[('pair_only',l['item_id'],l['variant'])]['predicted']!=qby[('full',l['item_id'],l['variant'])]['predicted'],right_answer_changed=qby[('pair_only',r['item_id'],r['variant'])]['predicted']!=qby[('full',r['item_id'],r['variant'])]['predicted'],left_option_changed=dec['pair_only'][p['left']]!=dec['full'][p['left']],right_option_changed=dec['pair_only'][p['right']]!=dec['full'][p['right']]))
  if key in seen:continue
  seen.add(key)
  for m in names:
   a=qby[(m,l['item_id'],l['variant'])];b=qby[(m,r['item_id'],r['variant'])];preserves=set(a['gold'])==set(b['gold'])
   qp.append(dict(method=m,pair_id='|'.join(key),source_id=p['source_id'],group='question_preserving' if preserves else 'question_changing',paired_correct=a['correct'] and b['correct'],relation_correct=(set(a['predicted'])==set(b['predicted']))==preserves))
 save('medpic_candidate_scores/'+split+'.jsonl',candidate_rows);ev.write_csv(OUT/'medpic_candidate_scores'/('ranking_changes_'+split+'.csv'),changes);ev.write_csv(OUT/'medpic_candidate_scores'/('native_pair_changes_'+split+'.csv'),prchanges)
 # Additional question-level pairs are separate from score-difference pair statistics.
 save_scope(scope,nodes,prs,[],questions);save('predictions/'+scope+'/question_pairs.jsonl',qp)
 boot=ev.SourceBootstrap([q['source_id'] for q in questions]);extra={}
 for group in sorted(set(p['group'] for p in qp)):
  for field in ['paired_correct','relation_correct']:
   level=group+'_'+field;extra[level]={}
   for m in names:
    rr=[p for p in qp if p['group']==group and p['method']==m];extra[level][m]={p['pair_id']:(p['source_id'],p[field]) for p in rr}
    vv=boot.mean([p[field] for p in rr],[p['source_id'] for p in rr]);ALL_BOOT.append(dict(scope=scope,comparison=m,metric=level,**boot.summary(vv),sources=len(boot.sources),draws=2000))
    next(t for t in ALL_METRICS if t['scope']==scope and t['method']==m).update({level:ev.finite(vv[0]),level+'_n':len(rr)})
   for a,b in pairs_to_compare(names):
    ar=extra[level][a];br=extra[level][b];keys=sorted(ar);sources=[ar[k][0] for k in keys];dif=[int(ar[k][1])-int(br[k][1]) for k in keys]
    ALL_BOOT.append(dict(scope=scope,comparison=a+' minus '+b,metric=level,**boot.summary(boot.mean(dif,sources)),sources=len(boot.sources),draws=2000))
    ALL_TRANS.append(dict(scope=scope,comparison=a+' minus '+b,level=level,valid_denominator=len(keys),wrong_to_right=sum(ar[k][1] and not br[k][1] for k in keys),right_to_wrong=sum(br[k][1] and not ar[k][1] for k in keys),kept_right=sum(ar[k][1] and br[k][1] for k in keys),kept_wrong=sum(not ar[k][1] and not br[k][1] for k in keys)))
 # Explicit cardinality margins, selection distribution and native-pair endpoint strata.
 size_rows=[]
 for m in names:
  for k in range(6):
   qq=[q for q in questions if q['method']==m and q['variant']=='full'];chosen=[q for q in qq if len(q['predicted'])==k]
   size_rows.append(dict(split=split,method=m,k=k,chosen_questions=len(chosen),chosen_correct=sum(q['correct'] for q in chosen),candidate_count=sum(c['k']==k for c in candidate_rows),natural_questions=len(qq)))
 ev.write_csv(OUT/'medpic_candidate_scores'/('size_distribution_'+split+'.csv'),size_rows)
 return questions

def main():
 start=time.time();fits=calibrate();dc=med_calibrate(fits['full']);write(OUT/'configs/calibration_frozen.json',dict(frozen=True,selected_decoder=dc['selected_decoder'],before_test_evaluation=True))
 for scope in ['dev','test','new60']:source_eval(scope,fits['full'])
 pooled=[[],[],[]]
 for f in read(BASE/'folds.json')['folds']:
  name='fold'+str(f['fold']);out=source_eval('cv_'+name+'_common',fits[name],f)
  for target,rs in zip(pooled,out):target.extend(rs)
 save_scope('cv_pooled_common',*pooled,[])
 med_eval('dev',fits['full'],dc);med_eval('test',fits['full'],dc)
 ev.write_csv(OUT/'metrics.csv',ALL_METRICS);ev.write_csv(OUT/'source_bootstrap.csv',ALL_BOOT);ev.write_csv(OUT/'transitions.csv',ALL_TRANS)
 write(OUT/'completion.json',dict(complete=True,seconds=time.time()-start,gpu_used=[],metric_rows=len(ALL_METRICS),bootstrap_rows=len(ALL_BOOT),transition_rows=len(ALL_TRANS)))
 print('ALL COMPLETE',time.time()-start,flush=True)
if __name__=='__main__':main()
