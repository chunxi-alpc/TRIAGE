import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from collections import Counter
from common import lp,rate,bootstrap,mean_ci

def fuse(f,m,kind,param):
 f=np.asarray(f);m=np.asarray(m)
 if kind=='probability_average':
  w=float(param)
  if w==1:return f.copy()
  if w==0:return m.copy()
  return np.logaddexp(np.log(w)+f,np.log1p(-w)+m)
 if kind=='CAD_full':return (1+param)*f-param*m
 if kind=='CAD_masked':return (1+param)*m-param*f
 if kind=='learned':return param[0]*f+param[1]*m
 if kind=='Full':return f.copy()
 if kind=='Masked':return m.copy()
 raise ValueError(kind)
def balance_weights(data):
 counts=Counter(r['id'] for r in data)
 return np.array([1/counts[r['id']] for r in data])/len(counts)
def accuracy(data,kind,param):
 wt=balance_weights(data)
 return float(sum(w*(int(np.argmax(fuse(r['f'],r['m'],kind,param)))==r['gold_idx']) for w,r in zip(wt,data)))
def fit(data):
 assert data
 selected={};grid={}
 for kind,params in [('probability_average',[1,.75,.5,.25,0]),('CAD_full',[0,.25,.5,1,2,4]),('CAD_masked',[0,.25,.5,1,2,4])]:
  trials=[dict(parameter=p,development_accuracy=accuracy(data,kind,p)) for p in params]
  selected[kind]=max(trials,key=lambda x:x['development_accuracy'])['parameter'];grid[kind]=trials
 wt=balance_weights(data)
 def objective(w):
  loss=.01*np.dot(w,w);grad=.02*w
  for weight,r in zip(wt,data):
   x=np.column_stack([r['f'],r['m']]);z=x@w;l=lp(z);p=np.exp(l);y=r['gold_idx']
   loss-=weight*l[y];grad+=weight*(p@x-x[y])
  return loss,grad
 result=minimize(objective,np.array([.5,.5]),jac=True,method='BFGS',options={'gtol':1e-8})
 assert result.success or np.linalg.norm(result.jac)<1e-6, result.message
 selected['learned']=result.x.tolist()
 return dict(parameters=selected,grid=grid,learned_objective=float(result.fun),gradient_norm=float(np.linalg.norm(result.jac)),development_cases=len(set(r['id'] for r in data)),development_variants=len(data),regularization=.01,selection='case-balanced accuracy for grids; case-balanced NLL plus L2 for learned')
def evaluate(data,kind,param):
 groups=[r.get('tree',r['id']) for r in data]
 pf=np.array([int(np.argmax(r['f'])) for r in data]);pm=np.array([int(np.argmax(r['m'])) for r in data])
 gold=np.array([r['gold_idx'] for r in data]);pred=np.array([int(np.argmax(fuse(r['f'],r['m'],kind,param))) for r in data])
 fc=pf==gold;mc=pm==gold;ok=pred==gold;bad=~fc&~mc;either=fc|mc;same=bad&(pf==pm);different=bad&(pf!=pm)
 result=dict(n=len(data),accuracy=rate(ok,groups),full_accuracy=rate(fc,groups),masked_accuracy=rate(mc,groups),B=rate(bad,groups),
  corrected=bootstrap(bad&ok,bad,groups),harm=bootstrap(either&~ok,either,groups),harm_overall=rate(either&~ok,groups),
  same_wrong=rate(same,groups),different_wrong=rate(different,groups),
  same_wrong_corrected=bootstrap(same&ok,same,groups),different_wrong_corrected=bootstrap(different&ok,different,groups),
  accuracy_minus_full=mean_ci(ok.astype(float)-fc,groups),accuracy_minus_masked=mean_ci(ok.astype(float)-mc,groups))
 records=[dict(id=r['id'],variant=r.get('variant'),split=r['split'],method=kind,parameter=param,labels=r['labels'],gold=r['labels'][r['gold_idx']],
  prediction=r['labels'][int(pred[i])],log_probabilities=lp(fuse(r['f'],r['m'],kind,param)).tolist(),
  full_prediction=r['labels'][int(pf[i])],masked_prediction=r['labels'][int(pm[i])],both_wrong=bool(bad[i]),corrected_both_wrong=bool(bad[i]&ok[i]),harmed=bool(either[i]&~ok[i])) for i,r in enumerate(data)]
 return result,records
