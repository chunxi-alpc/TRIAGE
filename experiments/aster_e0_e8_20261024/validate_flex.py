import os,json,time,math
from pathlib import Path
import torch
import torch.nn.functional as F
from engine import Engine
from flex_engine import controls,compiled_flex,FlexEngine
P=Path('/home/gcx/aster_e0_e8_20261024/private/flex_validation_v1');P.mkdir(parents=True,exist_ok=True)
def main():
 assert os.environ['CUDA_VISIBLE_DEVICES']=='6';torch.set_num_threads(4);torch.manual_seed(17);start=time.time();report={'torch':torch.__version__,'gpu':6,'synthetic':[],'models':[]}
 def save():
  report.update(elapsed_seconds=time.time()-start,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30)
  (P/'progress.json').write_text(json.dumps(report,indent=2))
 n=129;phase=121;roles=[None]*n
 for i in range(10,30):roles[i]={'kind':'condition','rules':[1]}
 for i in range(40,60):roles[i]={'kind':'action','rules':[0]}
 for i in range(70,90):roles[i]={'kind':'action','rules':[1]}
 for i in range(100,120):roles[i]={'kind':'option','rules':[1]}
 p=[.9,.1];settings=[('none',{}),('blind',{}),('hard',{}),('hard',{'gate_phase':'all','gate_scope':'rule','gate_options':True}),('gate',{'gate_phase':'all'}),('gate',{'blind':True})]
 for dtype in (torch.float32,torch.bfloat16):
  q=torch.randn(1,8,n,32,device='cuda',dtype=dtype);k=torch.randn(1,2,n,32,device='cuda',dtype=dtype);v=torch.randn_like(k)
  for mode,kw in settings:
   score,block=controls(roles,phase,mode,p,2.,**kw);qi=torch.arange(n,device='cuda')[:,None];ki=torch.arange(n,device='cuda')[None,:];expected=qi>=ki
   act=torch.tensor([Engine.role(r)[0] in ('action','option') for r in roles],device='cuda')
   if mode=='blind' or kw.get('blind'):expected &= ~((qi<phase)&~act[:,None]&act[None,:])
   bad=[];bias=[]
   for r in roles:
    kind,rs=Engine.role(r);t=bool(rs) and (kind=='action' or (kind=='condition' and kw.get('gate_scope')=='rule') or (kind=='option' and kw.get('gate_options',False)));bad.append(t and 0 not in rs);bias.append(2*(math.log(max(p[j] for j in rs))-math.log(max(p))) if t else 0.)
   active=torch.ones((n,1),device='cuda',dtype=torch.bool) if kw.get('gate_phase')=='all' else qi>=phase
   if mode=='hard':expected &= ~(active&torch.tensor(bad,device='cuda')[None,:])
   assert torch.equal(block.mask_mod(0,0,qi,ki),expected),'access mismatch'
   dense=torch.zeros(n,n,device='cuda')
   if mode=='gate':dense+=active*torch.tensor(bias,device='cuda')[None,:]
   dense.masked_fill_(~expected,float('-inf'))
   with torch.no_grad():
    ref=F.scaled_dot_product_attention(q,k.repeat_interleave(4,1),v.repeat_interleave(4,1),attn_mask=dense.to(dtype));out=compiled_flex(q,k,v,score_mod=score,block_mask=block,enable_gqa=True)
   delta=float((ref.float()-out.float()).abs().max());tol=2e-5 if dtype==torch.float32 else .04;assert delta<tol,(mode,dtype,delta)
   report['synthetic'].append({'mode':mode,'controls':kw,'dtype':str(dtype),'max_abs':delta,'tolerance':tol,'pass':True});save()
 score,block=controls(roles,n,'blind');x=torch.randn(1,8,n,32,device='cuda');y=x.clone();acts=[i for i,r in enumerate(roles) if Engine.role(r)[0] in ('action','option')];y[:,:,acts]+=3
 with torch.no_grad():
  for _ in range(3):
   x=x+compiled_flex(x,x[:,::4],x[:,::4],score_mod=score,block_mask=block,enable_gqa=True)
   y=y+compiled_flex(y,y[:,::4],y[:,::4],score_mod=score,block_mask=block,enable_gqa=True)
 keep=[i for i in range(n) if i not in acts];delta=float((x[:,:,keep]-y[:,:,keep]).abs().max());assert delta==0,delta;report['indirect_access']={'layers':3,'protected_max_abs':delta,'pass':True};save()
 c=next(c for c in map(json.loads,Path('/home/gcx/aster_e0_e8_20261024/private/cases.jsonl').read_text().splitlines()) if c['split']=='dev' and c['seed']==0 and c['variant']=='AC');c={k:c[k] for k in ('patient','rules','options')}
 for name,path in [('qwen','Qwen3-8B'),('llama','Meta-Llama-3.1-8B-Instruct')]:
  e=FlexEngine('/home/gcx/TRIAGE/models/modelscope/'+path);flex=e.intervene;ids,roles=e.render(c);phase=len(ids);suffix=e.tok.encode('\nAnswer:',add_special_tokens=False);ids+=suffix;roles+=[None]*len(suffix);p=[.01/(len(c['rules'])-1)]*len(c['rules']);p[0]=.99
  for mode,kw in [('none',{}),('blind',{}),('hard',{'gate_phase':'all'}),('gate',{'gate_phase':'all','gate_options':True})]:
   e.intervene=Engine.intervene.__get__(e);ref=e.forward(ids,roles,phase,mode,p,2.,**kw)[-1];e.intervene=flex;actual=e.forward(ids,roles,phase,mode,p,2.,**kw)[-1]
   labels=[e.tok.encode(' '+chr(65+j),add_special_tokens=False)[0] for j in range(len(c['options']))];rp=ref[labels].softmax(0);ap=actual[labels].softmax(0)
   report['models'].append({'model':name,'mode':mode,'controls':kw,'tokens':len(ids),'max_logit_abs':float((actual-ref).abs().max()),'option_probability_max_abs':float((rp-ap).abs().max()),'same_option':int(rp.argmax())==int(ap.argmax()),'finite':bool(torch.isfinite(actual).all())});assert torch.isfinite(actual).all();save()
  del flex,e,ref,actual;torch.cuda.empty_cache()
 report['complete']=True;report['gpu_hours']=(time.time()-start)/3600;save();(P/'results.json').write_text(json.dumps(report,indent=2));print('complete',flush=True)
if __name__=='__main__':main()
