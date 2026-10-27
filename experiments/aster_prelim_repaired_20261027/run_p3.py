"""P3 layer knockout, neutral-to-familiar K/V patching, and dev-only probes."""
import argparse,collections,hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
from run_p1 import dump,sha
BASE=Path(__file__).resolve().parent

def main():
 from frozen_guard import check
 check()
 p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--model',required=True);a=p.parse_args()
 assert json.loads((BASE/'results/p2/gate_status.json').read_text())['P2']=='PASS_NEUTRAL_GATE'
 out=BASE/'private'/('p3_'+a.name);out.mkdir(exist_ok=False);start=time.time();torch.set_num_threads(4);torch.manual_seed(20261027);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True,padding_side='left')
 if tok.pad_token_id is None:tok.pad_token=tok.eos_token
 model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,torch_dtype=torch.float32,attn_implementation='sdpa',device_map={'':'cuda:0'}).eval()
 layers=model.model.layers;nl=len(layers);p2=BASE/'private'/('p2_'+a.name)
 prompts={r['sha256']:r['text'] for r in map(json.loads,(p2/'prompts.jsonl').read_text().splitlines())}
 links=[r for r in map(json.loads,(p2/'predictions.jsonl').read_text().splitlines()) if r['control']=='standard']
 idx={(r['id'],r['kind'],r['condition']):r for r in links}
 substitutions={r['id']:r for r in map(json.loads,(p2/'substitutions.jsonl').read_text().splitlines())}
 rows=[r for r in map(json.loads,(BASE/'private/cases.jsonl').read_text().splitlines()) if r['variant']=='SW'];assert len(rows)==json.loads((BASE/'results/data_audit.json').read_text())['base_count']
 choices=[tok.encode(x,add_special_tokens=False) for x in ['1','2']];assert all(len(x)==1 for x in choices);choiceids=[x[0] for x in choices]
 state={'mode':'none','window':set(),'cache':{},'spans':[],'decoy':[]};handles=[]
 def pre(layer):
  def hook(module,args,kw):
   if state['mode']!='knock' or layer not in state['window']:return
   h=kw['hidden_states'] if 'hidden_states' in kw else args[0];b,l,_=h.shape;mask=kw.get('attention_mask')
   if mask is None:
    mask=torch.zeros((b,1,l,l),device=h.device,dtype=h.dtype)
    mask.masked_fill_(torch.ones((l,l),device=h.device,dtype=torch.bool).triu(1),torch.finfo(h.dtype).min)
    mask.masked_fill_(~state['attention'].bool()[:,None,None,:],torch.finfo(h.dtype).min)
   elif mask.dtype==torch.bool:mask=torch.zeros_like(mask,dtype=h.dtype).masked_fill_(~mask,torch.finfo(h.dtype).min)
   else:mask=mask.clone()
   for i,positions in enumerate(state['spans']):mask[i,:,-1,positions]=torch.finfo(h.dtype).min
   assert mask.dtype.is_floating_point
   state['knockout_hook_calls']=state.get('knockout_hook_calls',0)+1
   if not state.get('knockout_mask_verified'):
    assert all(bool((mask[i,0,-1,pos]<-1e4).all()) for i,pos in enumerate(state['spans']))
    state['knockout_mask_verified']=True
   kw['attention_mask']=mask;return args,kw
  return hook
 def projection(layer,kind):
  def hook(module,args,result):
   if state['mode']=='capture':
    state['cache'][layer,kind]=[result[i,positions].detach().clone() for i,positions in enumerate(state['decoy'])]
   elif state['mode'] in ('patch_'+kind,'patch_KV'):
    result=result.clone()
    for i,positions in enumerate(state['decoy']):result[i,positions]=state['cache'][layer,kind][i]
    return result
  return hook
 for i,layer in enumerate(layers):
  handles.append(layer.self_attn.register_forward_pre_hook(pre(i),with_kwargs=True))
  handles.append(layer.self_attn.k_proj.register_forward_hook(projection(i,'K')))
  handles.append(layer.self_attn.v_proj.register_forward_hook(projection(i,'V')))
 def span_indices(prompt,r,action_override=None):
  offsets=tok(prompt,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping'];spans=[];decoy=[]
  for q in r['rules']:
   action=action_override if action_override is not None and q['bid']!=r['gold'] else q['action']
   prefix=f"Branch {q['bid']}: IF {q['condition']} THEN "
   pos=prompt.index(prefix)+len(prefix);assert prompt[pos:pos+len(action)]==action
   tokens=[i for i,(b,e) in enumerate(offsets) if b<pos+len(action) and e>pos]
   assert tokens;spans+=tokens
   if q['bid']!=r['gold']:decoy=tokens
  return spans,decoy
 manifest={'stage':'P3','name':a.name,'status':'running','model':a.model,'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'layers':nl,'cases':len(rows),'batch':4,'script_sha256':sha(__file__),'protocol_sha256':sha(BASE/'p3_protocol.json'),'readout':'normalized next-token probabilities of 1/2, not full free generation','knockout':'last input query -> both terminal action spans; other queries and options remain visible','patch':'neutral to familiar, k_proj/v_proj output before Qwen k_norm and RoPE; decoy tokens only; all layers','probe':'last input position after every block; leave one dev tree out; separate blind/visible linear classifiers','test_data_used':False}
 dump(out/'manifest.json',manifest);predictions=[];hidden={'blind':[],'visible':[]}
 def forward(texts,rs,method,capture_hidden=False):
  x=tok(texts,return_tensors='pt',padding=True,add_special_tokens=False).to('cuda');state['attention']=x.attention_mask
  with torch.inference_mode():y=model(**x,use_cache=False,output_hidden_states=capture_hidden)
  probs=y.logits[:,-1,choiceids].float().softmax(-1).cpu().numpy()
  for r,pv in zip(rs,probs):
   pred=str(int(pv.argmax())+1);z={'model':a.name,'id':r['id'],'tree':r['tree'],'method':method,'gold':r['gold'],'prediction':pred,'correct':pred==r['gold'],'branch_probabilities':pv.tolist()};predictions.append(z);raw.write(json.dumps(z)+'\n')
  if capture_hidden:hidden[method].append(np.stack([h[:,-1].float().cpu().numpy() for h in y.hidden_states[1:]],axis=1))
  del y,x
 def padded_spans(texts,rs,neutral=False):
  lengths=[len(tok.encode(t,add_special_tokens=False)) for t in texts];maximum=max(lengths);allsp=[];ds=[]
  for t,r,l in zip(texts,rs,lengths):
   spans,d=span_indices(t,r,substitutions[r['id']]['neutral'] if neutral else None);allsp.append([k+maximum-l for k in spans]);ds.append([k+maximum-l for k in d])
  return allsp,ds
 with (out/'predictions.jsonl').open('w',buffering=1) as raw:
  for offset in range(0,len(rows),4):
   rs=rows[offset:offset+4]
   fam=[prompts[idx[r['id'],'familiar','visible']['prompt_sha256']] for r in rs]
   neu=[prompts[idx[r['id'],'neutral','visible']['prompt_sha256']] for r in rs]
   blind=[prompts[idx[r['id'],'familiar','blind']['prompt_sha256']] for r in rs]
   sf,df=padded_spans(fam,rs);sn,dn=padded_spans(neu,rs,True)
   aligned=[]
   for ft,nt,fpos,npos in zip(fam,neu,df,dn):
    fi=tok.encode(ft,add_special_tokens=False);ni=tok.encode(nt,add_special_tokens=False);assert len(fi)==len(ni)
    shift=max(len(tok.encode(t,add_special_tokens=False)) for t in fam)-len(fi)
    changed=[i+shift for i,(u,v) in enumerate(zip(fi,ni)) if u!=v]
    region=sorted(set(fpos+npos+changed));assert region
    region=list(range(min(region),max(region)+1));aligned.append(region)
    assert all(u==v for i,(u,v) in enumerate(zip(fi,ni)) if i+shift not in region)
   df=dn=aligned
   state.update(mode='none',spans=sf,decoy=df);forward(blind,rs,'blind',True);forward(fam,rs,'visible',True)
   state.update(mode='capture',decoy=dn,cache={});forward(neu,rs,'neutral')
   for mode in ['K','V','KV']:
    state.update(mode='patch_'+mode,decoy=df);forward(fam,rs,'patch_'+mode)
   state.update(mode='knock',spans=sf)
   for lo in range(nl-3):
    state['window']=set(range(lo,lo+4));forward(fam,rs,f'knock_{lo+1}_{lo+4}')
   state['window']=set(range(nl));forward(fam,rs,'knock_all')
   state.update(mode='none',cache={});dump(out/'status.json',{'cases_completed':offset+len(rs),'cases_total':len(rows),'seconds':time.time()-start,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30})
   print(json.dumps({'cases_completed':offset+len(rs),'total':len(rows)}),flush=True)
 for h in handles:h.remove()
 # Probe features are kept private. No test trees enter training or scoring.
 features={k:np.concatenate(v,axis=0) for k,v in hidden.items()};np.savez_compressed(out/'probe_features.npz',**features)
 labels=np.array([int(r['gold'])-1 for r in rows]);trees=np.array([r['tree'] for r in rows]);probe=[]
 del model;torch.cuda.empty_cache()
 from sklearn.linear_model import LogisticRegression
 from sklearn.preprocessing import StandardScaler
 from sklearn.pipeline import make_pipeline
 for condition,xx in features.items():
  for layer in range(nl):
   pred=np.zeros(len(rows),dtype=int)
   for tree in sorted(set(trees)):
    train=trees!=tree;valid=~train
    clf=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,solver='liblinear',max_iter=1000,random_state=20261027));clf.fit(xx[train,layer],labels[train]);pred[valid]=clf.predict(xx[valid,layer])
   probe.append({'model':a.name,'condition':condition,'layer':layer+1,'correct':int((pred==labels).sum()),'n':len(rows),'accuracy_pct':float((pred==labels).mean()*100),'predictions':pred.tolist()})
 dump(out/'probes.json',probe);manifest.update(status='completed',wall_seconds=time.time()-start,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,rows=len(predictions),knockout_mask_verified=state.get('knockout_mask_verified',False),knockout_hook_calls=state.get('knockout_hook_calls',0));dump(out/'manifest.json',manifest);dump(out/'DONE.json',{'status':'completed'})
if __name__=='__main__':main()
