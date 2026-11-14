"""Descriptive source probes and score diagnostics using read-only cached features."""
from pathlib import Path
import os,sys,json,collections,csv,re
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['OMP_NUM_THREADS']='4'
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score,roc_auc_score,accuracy_score
SRC=Path('/home/gcx/aster-cr-retrain-paired-v2-20261111')
BAL=SRC/'reports/aster_cr_multisource_20261112/balanced_10006'
OUT=Path(__file__).resolve().parents[1]
PRIVATE=Path('/home/gcx/aster-cr-medeinst-alignment-audit-private-20261112')
sys.path.insert(0,str(SRC));sys.path.insert(0,str(BAL/'code'))
from worker import CombinedStore
def rows(p):return [json.loads(s) for s in Path(p).open() if s.strip()]
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvwrite(p,rs):
 with Path(p).open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,list(rs[0]));w.writeheader();w.writerows(rs)
def stats(a):
 a=np.asarray(a,float)
 return dict(n=len(a),mean=float(a.mean()),std=float(a.std()),median=float(np.median(a)),q05=float(np.quantile(a,.05)),q95=float(np.quantile(a,.95)),min=float(a.min()),max=float(a.max()))
store=CombinedStore()
fit=rows(BAL/'data/fit_pairs.jsonl')
mf=[dict(p,dataset_id='MEGA-CDP') for p in rows(SRC/'reports/aster_cr_retrain_paired_v2_20261111/data/E2_pairs.jsonl') if p['kind']=='F']
pairs=fit+mf
ids=sorted({p[k] for p in pairs for k in ['left','right']});idx={uid:i for i,uid in enumerate(ids)}
model=BAL/'training/E-Multi-Balanced/full'
m2=np.load(model/'m2.npz');m3=np.load(model/'m3.npz');m1=json.loads((model/'m1.json').read_text())
heads=m2['heads'];mu=m2['mu'];scale=m2['scale'];bases={k:m2[k] for k in ['S','U','N']}
coords={k:np.empty((len(ids),bases[k].shape[1]),np.float32) for k in bases}
ell=np.array([store.ell(uid) for uid in ids])
tokens=np.array([store.records[uid]['tokens'] if uid in store.records else store.base.records[uid]['tokens'] for uid in ids])
for start in range(0,len(ids),256):
 block=ids[start:start+256]
 x=np.stack([store.heads(uid,heads).reshape(-1) for uid in block]).astype(np.float32)
 x=(x-mu)/scale
 for k in bases:coords[k][start:start+len(block)]=x@bases[k]
print('coordinates_complete',len(ids),flush=True)
correction=coords['N']@m3['w'];b=float(m3['b']);delta=float(m3['delta'])
# Match the original single-vector FP32 projection near the hard boundary.
boundary=set()
for p in pairs:
 l,r=idx[p['left']],idx[p['right']]
 if abs(abs(float(correction[r]-correction[l]))-delta)<5e-6:boundary.update([p['left'],p['right']])
for uid in boundary:
 raw=store.heads(uid,heads).reshape(-1).astype(np.float32)
 exact=(((raw-mu)/scale)@bases['N']).astype(np.float64)
 correction[idx[uid]]=exact@m3['w']
score_rows=[]
for p in pairs:
 l,r=idx[p['left']],idx[p['right']];d=float(ell[r]-ell[l]);dc=float(correction[r]-correction[l]);z=p['right_label']-p['left_label']
 score_rows.append(dict(pair_id=p['pair_id'],dataset_id=p.get('dataset_id','MEGA-CDP'),kind=p['kind'],case_id=p['case_id'],source_id=p['source_id'],
   population='MEGA_F_pool' if p in mf else 'frozen_training',original_delta=d,correction_delta=dc,corrected_delta=d+dc,
   delta_budget=delta,relative_budget=delta/abs(d) if abs(d)>1e-12 else None,
   original_direction_correct=z*d>0 if z else None,corrected_direction_correct=z*(d+dc)>0 if z else None,
   original_both_correct=bool((ell[l]>0)==bool(p['left_label']) and (ell[r]>0)==bool(p['right_label'])),
   corrected_both_correct=bool((ell[l]+correction[l]+b>0)==bool(p['left_label']) and (ell[r]+correction[r]+b>0)==bool(p['right_label'])),
   active_constraint=abs(abs(dc)-delta)<=1e-6,over_budget=abs(dc)>delta+1e-7,left_tokens=int(tokens[l]),right_tokens=int(tokens[r])))
csvwrite(OUT/'pair_score_diagnostics.csv',score_rows)
summary={}
for name,select in [('MEGA_AP',lambda r:r['kind'] in ['A','P']),('MEGA_C',lambda r:r['kind']=='C'),('MedEinst_F',lambda r:r['dataset_id']=='MedEinst'),('MEGA_F_666',lambda r:r['population']=='MEGA_F_pool')]:
 rr=[r for r in score_rows if select(r)];d=np.array([r['original_delta'] for r in rr]);c=np.array([r['correction_delta'] for r in rr])
 ps=[p for p in pairs if select(dict(kind=p['kind'],dataset_id=p.get('dataset_id','MEGA-CDP'),population='MEGA_F_pool' if p in mf else 'frozen_training'))]
 z=np.array([p['right_label']-p['left_label'] for p in ps])
 summary[name]=dict(n=len(rr),raw_delta=stats(d),raw_abs_delta=stats(abs(d)),raw_delta_RMS=float(np.sqrt(np.mean(d*d))),correction_RMS=float(np.sqrt(np.mean(c*c))),
  relative_budget=stats([r['relative_budget'] for r in rr if r['relative_budget'] is not None]),zero_raw_delta=int(np.sum(d==0)),
  raw_delta_below_budget=int(np.sum(abs(d)<delta)),active_constraints=sum(r['active_constraint'] for r in rr) if np.any(z) else None,over_budget=sum(r['over_budget'] for r in rr) if np.any(z) else None,max_budget_excess=float(np.max(abs(c)-delta)) if np.any(z) else None,
  original_direction_correct_rate=float(np.mean(z*d>0)) if np.any(z) else None,
  corrected_direction_correct_rate=float(np.mean(z*(d+c)>0)) if np.any(z) else None,
  original_both_correct_rate=float(np.mean([r['original_both_correct'] for r in rr])),
  corrected_both_correct_rate=float(np.mean([r['corrected_both_correct'] for r in rr])),
  evidence_response_retention=float(np.mean(z*(d+c))/np.mean(z*d)) if abs(np.mean(z*d))>1e-12 else None,
  tokens=stats([r['left_tokens'] for r in rr]),interpretation='unweighted descriptive training/pool diagnostics; model not refitted; MEGA F 666 are outside current frozen training but some shared cases')
# One original endpoint per case for source probes; folds group MEGA by guideline source, MedEinst by case.
representatives={}
for p in sorted(fit,key=lambda p:p['pair_id']):representatives.setdefault((p.get('dataset_id','MEGA-CDP'),p['case_id']),p)
ps=list(representatives.values());y=np.array([int(p.get('dataset_id')=='MedEinst') for p in ps])
groups=np.array([p['source_id'] if p.get('dataset_id')!='MedEinst' else p['case_id'] for p in ps])
ii=np.array([idx[p['left']] for p in ps]);jj=np.array([idx[p['right']] for p in ps])
texts=[store.input(p['left'])['prompt'] for p in ps]
def structural(t):
 return [len(t),t.count('\n'),t.count('- '),len(re.findall(r'[\u4e00-\u9fff]',t))/max(1,len(t)),t.count(' -> '),t.count('Clinical vignette:'),t.count('Differential diagnosis task.'),t.count('当前条件')]
xs=np.array([structural(t) for t in texts],float)
folds=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=20261112).split(xs,y,groups))
def probe(x):
 pred=np.zeros(len(y));fold_metrics=[]
 for tr,te in folds:
  clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=500,C=1.0,class_weight='balanced',random_state=20261112))
  clf.fit(x[tr],y[tr]);pr=clf.predict_proba(x[te])[:,1];pred[te]=pr
  fold_metrics.append(dict(test_n=len(te),balanced_accuracy=float(balanced_accuracy_score(y[te],pr>=.5)),auc=float(roc_auc_score(y[te],pr))))
 return dict(n=len(y),balanced_accuracy=float(balanced_accuracy_score(y,pred>=.5)),accuracy=float(accuracy_score(y,pred>=.5)),auc=float(roc_auc_score(y,pred)),folds=fold_metrics)
probes={'structure':probe(xs)}
for k in ['S','U','N']:
 probes[k+'_original_coordinates']=probe(coords[k][ii])
 probes[k+'_pair_difference']=probe(coords[k][jj]-coords[k][ii])
rule=np.array(['Differential diagnosis task. Choose the better-supported diagnosis from the two candidates.' in t for t in texts],int)
probes['format_rule']=dict(n=len(y),accuracy=float(np.mean(rule==y)),balanced_accuracy=float(balanced_accuracy_score(y,rule)),rule='presence of the differential-diagnosis instruction; input text only')
def cos(a,b):
 return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))) if np.linalg.norm(a)*np.linalg.norm(b)>0 else None
selected=set(m1['heads'])
m1stats=dict(selected_heads=len(selected),selected_in_top64_by_class={k:len(selected&set(m1['ranks'][k])) for k in ['A','P','C','F']},
 selected_F_candidates_only=len(selected&set(m1['ranks']['F'])-set(m1['ranks']['A'])-set(m1['ranks']['P'])-set(m1['ranks']['C'])),
 F_C_top64_overlap=len(set(m1['ranks']['F'])&set(m1['ranks']['C'])),
 attribution_cosine_F_C=cos(np.asarray(m1['attribution']['F']),np.asarray(m1['attribution']['C'])),
 exact_F_representatives=len(m1['exact_representatives']['F']),
 note='Different within-pair interventions; no cross-source pair subtraction. Candidate-rank differences alone do not show source causation.')
report=dict(score_diagnostics=summary,source_probes=probes,M1=m1stats,
 source_probe_protocol='5-fold StratifiedGroupKFold, seed 20261112, one original endpoint per original case, MEGA grouped by guideline source and MedEinst by released case ID. Logistic regression C=1 balanced; train-fold StandardScaler. Read-only learned S/U/N used; those representations were fitted using all original training data, so probe is descriptive, not an unbiased nested generalization estimate. No test labels used.',
 limitations=['MedEinst underlying DDXPlus case mapping absent; distinct released case IDs may share synthetic templates.','Perfect source separability does not establish test leakage or causation.','Projection probe cannot separate source from task with this confounded design.','Source-only additive constant cancels within a pair; task-dependent gradients, scaling, subspace geometry and evidence constraints need not cancel.'])
write(OUT/'source_diagnostics.json',report)
np.savez_compressed(PRIVATE/'probe_coordinates.npz',ids=np.array(ids),ell=ell,correction=correction,**coords)
print(json.dumps({'score_diagnostics':summary,'probes':{k:{q:v for q,v in val.items() if q!='folds'} for k,val in probes.items()},'M1':m1stats},ensure_ascii=False))
