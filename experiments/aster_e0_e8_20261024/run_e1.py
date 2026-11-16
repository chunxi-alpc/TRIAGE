import argparse,hashlib,json,os,time
from pathlib import Path
import torch
from engine import Engine
P=Path('/home/gcx/aster_e0_e8_20261024/private')
def main():
 a=argparse.ArgumentParser();a.add_argument('--model',required=True);a.add_argument('--name',required=True);a.add_argument('--limit',type=int,default=20);a=a.parse_args()
 assert os.environ['CUDA_VISIBLE_DEVICES'] in ['0','1','6']
 out=P/a.name;out.mkdir(exist_ok=False);start=time.time();e=Engine(a.model)
 allrows=list(map(json.loads,(P/'cases.jsonl').read_text().splitlines()));dev=[r for r in allrows if r['split']=='dev' and r['seed']==0];ids=sorted({r['id'] for r in dev},key=lambda x:hashlib.sha256(x.encode()).hexdigest())[:a.limit];dev=[r for r in dev if r['id'] in ids and r['variant'] in ['NC','AC','SW']]
 c=dev[0];changed=dict(c,patient=c['patient']+' A newly reported laboratory measurement is 7.')
 k1={};k2={}
 for case,capture in [(c,k1),(changed,k2)]:
  ids1,roles=e.render(case);e.forward(ids1,roles,len(ids1),'none',capture=capture)
 x,roles=e.render(c);y,_=e.render(changed);common=next((j for j,(a,b) in enumerate(zip(x,y)) if a!=b),min(len(x),len(y)))
 # Compare rule-only prefix; patient edit is causally downstream.
 deltas={str(l):float((k1[l][...,:common,:]-k2[l][...,:common,:]).abs().max()) for l in k1};del k1,k2
 (out/'T1.json').write_text(json.dumps({'max_abs':max(deltas.values()),'by_layer':deltas,'common_prefix_tokens':common,'pass':max(deltas.values())<=1e-5,'edit':'append patient fact; same-prefix check'}))
 with (out/'scores.jsonl').open('x') as f:
  for c in dev:
   t=time.time();blind=e.branch(c,blind=True);hidden=e.branch(c,hidden=True)
   p=[.01/(len(c['rules'])-1)]*len(c['rules']);p[c['gold_rule']]=.99
   arms={}
   for name,form,mode in [('group3','crop','none'),('group4','crop_action','none'),('group9_hard','full','hard'),('strict_diagnostic','full','strict')]:
    pred,logits=e.answer(c,form,mode,p,selected=c['gold_rule'],write_rule=True,blind=mode in ('hard','strict'),gate_options=True);arms[name]={'prediction':pred,'correct':pred==c['gold_option'],'logits':logits}
   r={k:c[k] for k in ['id','tree','variant','seed','gold_rule','gold_option']};r.update(blind=blind,hidden=hidden,arms=arms,seconds=time.time()-t);f.write(json.dumps(r)+'\n');f.flush()
 rows=list(map(json.loads,(out/'scores.jsonl').read_text().splitlines()));t2=sum(max(range(len(r['blind'])),key=lambda j:r['blind'][j])==max(range(len(r['hidden'])),key=lambda j:r['hidden'][j]) for r in rows)/len(rows)
 acc=lambda field:sum(max(range(len(r[field])),key=lambda j:r[field][j])==r['gold_rule'] for r in rows)/len(rows)
 t3={v:sum(r['arms']['group9_hard']['correct']-r['arms']['group3']['correct'] for r in rows if r['variant']==v)/sum(r['variant']==v for r in rows) for v in ['NC','AC','SW']}
 summary={'scope':'E1 engineering pilot on development only; NOT formal E2','cases':len(set(r['id'] for r in rows)),'rows':len(rows),'T1':json.loads((out/'T1.json').read_text()),'T2':{'agreement':t2,'blind_accuracy':acc('blind'),'hidden_accuracy':acc('hidden'),'pass':t2>=.9 and abs(acc('blind')-acc('hidden'))<=.02},'T3':{'hard_minus_crop_by_variant':t3,'pass':all(abs(v)<=.02 for v in t3.values())},'elapsed_seconds':time.time()-start,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'test_set_used':False}
 summary['gate_pass']=summary['T1']['pass'] and summary['T2']['pass'] and summary['T3']['pass'];(out/'summary.json').write_text(json.dumps(summary,indent=2))
 print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
