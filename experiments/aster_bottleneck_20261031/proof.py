"""Patient-first sufficiency and rank-one LEACE necessity on held-out age pairs."""
import argparse,json,time
from pathlib import Path
from collections import defaultdict
import torch
from .run import OUT,BASE,read,write

def fit_erasers():
 accepted={r['pair_id'] for r in read(BASE/'review/accepted_pairs.jsonl') if r['split']=='fit'}
 stats={};pairs=[]
 for folder in ['collect0','collect1']:
  for p in (BASE/folder).glob('*.pt'):
   record=torch.load(p,map_location='cpu',weights_only=False);row=record['row']
   if row['pair_id'] not in accepted:continue
   pairs.append(row['pair_id'])
   for state,label in [('positive',1.),('negative',-1.)]:
    for head,x in record['offline'][state].items():
     x=x.double();d=x.shape[-1]
     if head not in stats:stats[head]=[torch.zeros(d,d,dtype=torch.double),torch.zeros(d,dtype=torch.double),torch.zeros(d,dtype=torch.double),0]
     s=stats[head];s[0]+=x.T@x/x.shape[0];s[1]+=x.mean(0);s[2]+=label*x.mean(0);s[3]+=1
 result={};residuals=[]
 for head,(xx,total,cross,n) in stats.items():
  mu=total/n;cov=xx/n-mu[:,None]*mu[None,:];cross=cross/n
  eig,v=torch.linalg.eigh((cov+cov.T)/2);keep=eig>max(float(eig.max())*1e-8,1e-12)
  sqrt=(v[:,keep]*eig[keep].sqrt())@v[:,keep].T
  inv=(v[:,keep]*eig[keep].rsqrt())@v[:,keep].T
  u=inv@cross;norm=float(u.norm())
  if norm<1e-12:continue
  u=u/norm;a=sqrt@u;b=u@inv
  randoms={}
  for seed in [13,37,73]:
   gen=torch.Generator().manual_seed(seed+head[0]*8+head[1]);z=torch.randn(len(mu),generator=gen,dtype=torch.double)
   z=v[:,keep]@(v[:,keep].T@z);z=z/(z.norm()+1e-12)
   randoms[str(seed)]=(sqrt@z,z@inv)
  residual=float((cross-a*(b@cross)).norm()/(cross.norm()+1e-12));residuals.append(residual)
  result[head]={'mean':mu,'a':a,'b':b,'random':randoms,'residual_cross_covariance_ratio':residual}
 torch.save(result,OUT/'leace.pt')
 write(OUT/'leace_fit.json',{'fit_pairs':len(pairs),'fit_sources':sorted({r['source_id'] for r in read(BASE/'review/accepted_pairs.jsonl') if r['pair_id'] in accepted}),
  'heads':len(result),'max_residual_cross_covariance_ratio':max(residuals),'method':'rank-one whiten-project-unwhiten affine LEACE; weighted token covariance; same-rank random controls'})
 return result

def worker(shard):
 from experiments.aster_manuscript_clinical_20261030.classify import model_and_classifier
 from experiments.aster_manuscript_clinical_20261030.engine import ClinicalExecutor
 from experiments.aster_manuscript_clinical_20261030.native_memory import query_chunked_attention
 from experiments.aster_abc_diagnostics_20261030.c.audit import strict_parse
 from src.aster.manuscript_core import load_expert
 from .runtime import execute
 torch.set_num_threads(4);torch.manual_seed(20261031)
 tok,model,_=model_and_classifier();engine=ClinicalExecutor(model,tok,load_expert(BASE/'analysis/expert.json'),json.loads((BASE/'frozen.json').read_text()))
 selected=json.loads((OUT/'selected_head_control.json').read_text())['heads'];fits=json.loads((OUT/'key_gain_fits.json').read_text());gain=fits['mean']['rms_gain']
 rows=[r for r in read(BASE/'review/accepted_pairs.jsonl') if r['split']=='confirm']
 specs=[{'name':'zero','kind':'zero'},{'name':'mean_forward','kind':'mean','gain':gain,'direction_heads':selected,'full_rule_keys':True},
  {'name':'mean_reverse','kind':'mean','gain':-gain,'direction_heads':selected,'full_rule_keys':True},
  {'name':'leace','kind':'erase','erase_heads':selected,'full_rule_keys':True},
  *[{'name':'random_erase'+str(seed),'kind':'erase','erase_heads':selected,'erase_random_seed':seed,'full_rule_keys':True} for seed in [13,37,73]]]
 trained=json.loads((OUT/'behavior_summary.json').read_text())
 specs.append({'name':'behavior_direction','kind':'mean','gain':trained['gain'],'direction_heads':trained['heads'],'full_rule_keys':True,'direction_mix':trained['selected']['coefficients']})
 out=OUT/f'proof_{shard}.jsonl';start=time.time();count=0
 done={(x['case_id'],x['order'],x['arm']) for x in read(out)} if out.exists() else set()
 with query_chunked_attention(model,256),out.open('a') as f:
  for i,row in enumerate(rows):
   if i%4!=shard:continue
   for state in ['positive','negative']:
    target=row['target_rule_id'];true=target if state=='positive' else target[:-1]+('N' if target[-1]=='Y' else 'Y')
    case={'case_id':row['pair_id']+':'+state,'patient':row['states'][state],'guideline':row['guideline'],'question':row['question'],'gold_rules':[true]}
    for order in ['guide_first','patient_first']:
     for spec in specs:
      if (case['case_id'],order,spec['name']) in done:continue
      result=execute(engine,case,{**spec,'order':order},route=[target])
      _,path,invalid=strict_parse(result['text'])
      rec={'case_id':case['case_id'],'source_id':row['source_id'],'state':state,'order':order,'arm':spec['name'],
       'root_correct':bool(path) and path[0]==true,'target_selected':bool(path) and path[0]==target,
       'true_margin':result['decision_margin'],'target_margin':result['decision_margin']*(1 if state=='positive' else -1),'result':result}
      f.write(json.dumps(rec,ensure_ascii=False)+'\n');f.flush();count+=1
 write(OUT/f'proof_{shard}_status.json',{'status':'complete','records':count,'elapsed':time.time()-start,'peak_gb':torch.cuda.max_memory_allocated()/1e9})
def summarize():
 rows=[]
 for i in range(4):
  p=OUT/f'proof_{i}.jsonl'
  if p.exists():rows+=read(p)
 groups=defaultdict(list)
 for r in rows:groups[(r['order'],r['state'],r['arm'])].append(r)
 result=[]
 for (order,state,arm),rs in groups.items():
  result.append({'order':order,'state':state,'arm':arm,'n':len(rs),'root_correct':sum(r['root_correct'] for r in rs),
   'target_selected':sum(r['target_selected'] for r in rs),'mean_target_margin':sum(r['target_margin'] for r in rs)/len(rs)})
 write(OUT/'proof_summary.json',{'rows':result,'scope':'held-out age pairs from two exposed diagnostic sources; no new general clinical causal claim'})
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--fit',action='store_true');p.add_argument('--shard',type=int);p.add_argument('--summarize',action='store_true');a=p.parse_args()
 if a.fit:fit_erasers()
 elif a.summarize:summarize()
 else:worker(a.shard)
