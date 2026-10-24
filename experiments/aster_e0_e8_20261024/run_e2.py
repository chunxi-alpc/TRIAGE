"""E2 core oracle sweep; requires a completed E1 gate; no test tuning."""
import argparse,hashlib,json,os,time
from pathlib import Path
from engine import Engine
P=Path('/home/gcx/aster_e0_e8_20261024/private')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--e1',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 assert os.environ['CUDA_VISIBLE_DEVICES'] in ['0','1','6']
 gate=json.loads(a.e1.read_text());assert gate['gate_pass'],'E1 failed: diagnose before E2'
 assert not a.out.exists(),'Refuse overwrite';a.out.mkdir(parents=True)
 e=Engine(a.model);rows=[r for r in map(json.loads,(P/'cases.jsonl').read_text().splitlines()) if r['split']=='dev'];start=time.time()
 with (a.out/'scores.jsonl').open('x') as f:
  for c in rows:
   p=[.01/(len(c['rules'])-1)]*len(c['rules']);p[c['gold_rule']]=.99
   methods=[('group3','crop','none',1.,'all'),('group4','crop_action','none',1.,'all')]+[(f'group9_{lam}_{layers}','full','hard' if lam=='inf' else 'gate',1. if lam=='inf' else lam,layers) for layers in ['all','last_half'] for lam in [.5,1.,2.,4.,8.,'inf']]
   for name,form,mode,lam,layers in methods:
    pred,scores=e.answer(c,form,mode,p,lam,layers);r={k:c[k] for k in ['id','tree','seed','variant','gold_option']};r.update(method=name,prediction=pred,correct=pred==c['gold_option'],logits=scores,oracle=True);f.write(json.dumps(r)+'\n');f.flush()
 (a.out/'status.json').write_text(json.dumps({'core_completed':True,'seconds':time.time()-start,'missing':['groups1,2,5,6,7,8,10','attention_mass'],'formal_E2_complete':False}))
if __name__=='__main__':main()
