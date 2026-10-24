import argparse,json,time,os,torch
from pathlib import Path
from engine import Engine
p=argparse.ArgumentParser();p.add_argument('--model');p.add_argument('--out');a=p.parse_args()
assert os.environ['CUDA_VISIBLE_DEVICES'] in ['0','1','6']
t=time.time();e=Engine(a.model);load=time.time()-t
rows=[json.loads(x) for x in Path('/home/gcx/aster_e0_e8_20261024/private/cases.jsonl').read_text().splitlines()];c=next(c for c in rows if c['split']=='dev' and c['variant']=='AC');safe={k:c[k] for k in ['patient','rules','options']};p=[.01/(len(c['rules'])-1)]*len(c['rules']);p[c['gold_rule']]=.99
res=[]
for name,kw in [('none',{}),('oracle_crop',{'form':'crop','selected':c['gold_rule']}),('action_all',{'mode':'hard','p':p,'gate_phase':'all'}),('soft',{'mode':'gate','p':p,'gate_phase':'all','lam':2.})]:
 t=time.time();pred,s=e.answer(safe,**kw);res.append({'method':name,'prediction':pred,'correct':pred==c['gold_option'],'seconds':time.time()-t})
t=time.time();bp=e.branch(safe,blind=True);bt=time.time()-t
out={'load_seconds':load,'answers':res,'branch_seconds':bt,'branch_probabilities':bp,'tokens':len(e.render(safe)[0]),'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30,'gpu':os.environ['CUDA_VISIBLE_DEVICES']};Path(a.out).write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
