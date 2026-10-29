import os,sys,time,json
from pathlib import Path
from types import SimpleNamespace
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
import run_stages as r
root=Path('/home/gcx/TRIAGE-worktrees')
a=SimpleNamespace(repo=root/'applicability-decision-20261029',run=root/'applicability-decision-20261029-run02',gpu=0)
sys.path.insert(0,str(a.repo))
p=r.read(Path(__file__).with_name('protocol.json'))
e=r.Engine(a,p);t=e.torch
s=r.samples(a,'fit')[0];st=e.state(s)
heads=[[17,5],[17,1],[17,4],[17,7]]
u,d=r.operators(a,e,heads)
result={'stage':'implementation_smoke','sample':s['id'],'uses_confirmation':False,'checks':[]}
with t.no_grad():
 full=e.model(input_ids=st.input_ids,use_cache=False,logits_to_keep=1).logits[0,-1]
 err=float((full[e.labels]-st.baseline[e.labels]).abs().max())
 assert err<1e-3,err
 result['cached_full_error']=err
 for component in ['key','value']:
  z=e.forward(st,replacement=(17,5,st.role_positions['action']['A'],st,component))
  assert t.equal(z,st.baseline),component
for bias in [False,True]:
 c=t.ones(len(heads),device=e.device) if bias else t.einsum('hdr,hd->hr',u,d)
 c=t.nn.Parameter(c/c.norm())
 def objective():
  v=c if bias else t.einsum('hdr,hr->hd',u,c)
  k,b,diag=e.edits(st,heads,v,1.,'joint',bias)
  z=e.forward(st,k,b)
  return z[e.labels[0]]-z[e.labels[1]],diag
 value,diag=objective();value.backward()
 assert c.grad is not None and bool(t.isfinite(c.grad).all())
 idx=tuple([0]*c.ndim);analytic=float(c.grad[idx]);epsilon=.01
 with t.no_grad():
  c[idx]+=epsilon;plus=float(objective()[0]);c[idx]-=2*epsilon;minus=float(objective()[0]);c[idx]+=epsilon
 finite=(plus-minus)/(2*epsilon)
 assert abs(analytic-finite)<.01,(analytic,finite)
 result['checks'].append({'bias':bias,'gradient':analytic,'finite_difference':finite,'rms':diag})
 with t.no_grad():
  v=c if bias else t.einsum('hdr,hr->hd',u,c)
  k,b,_=e.edits(st,heads,v,0.,'joint',bias)
  assert t.equal(e.forward(st,k,b),st.baseline)
t.cuda.synchronize();start=time.monotonic()
with t.no_grad():
 for _ in range(20): e.forward(st)
t.cuda.synchronize()
result['decision_seconds']=(time.monotonic()-start)/20
result['peak_memory_gib']=t.cuda.max_memory_allocated(0)/2**30
result['status']='passed'
r.dump(a.run/'smoke.json',result)
print(json.dumps(result),flush=True)
