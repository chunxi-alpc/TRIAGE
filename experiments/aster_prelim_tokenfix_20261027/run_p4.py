"""P4 genuine model tests; ordinary text pruning remains the primary U4 reference."""
import argparse,collections,importlib,json,os,time,copy
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from attention_engine import Engine
from run_p1 import dump,sha
BASE=Path(__file__).resolve().parent

def main():
 from frozen_guard import check
 check()
 p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--model',required=True);a=p.parse_args()
 assert json.loads((BASE/'results/p3/gate_status.json').read_text())['P3']=='COMPLETED'
 out=BASE/'private'/('p4_'+a.name);out.mkdir(exist_ok=False);start=time.time();torch.set_num_threads(4);torch.manual_seed(20261027);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
 model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,torch_dtype=torch.float32,attn_implementation='sdpa',device_map={'':'cuda:0'}).eval();engine=Engine(model,tok)
 rows=[json.loads(s) for s in (BASE/'private/cases.jsonl').read_text().splitlines()];groups=collections.defaultdict(dict)
 for r in rows:groups[r['id']][r['variant']]=r
 ids=sorted(groups);records=[];u1=[]
 manifest={'stage':'P4','name':a.name,'status':'running','gpu':os.environ['CUDA_VISIBLE_DEVICES'],'cases':len(ids),'U1_cases':20,'dtype':'float32','TF32':False,'U2_rate_threshold_removed_by_user':True,'engine_sha256':sha(BASE/'attention_engine.py'),'script_sha256':sha(__file__),'protocol_sha256':sha(BASE/'p4_protocol.json'),'serialization':'tokenize typed condition/action chunks separately; fixed boundary tokens across variants','blind_positions':'non-action cumsum position IDs; actions invisible at every layer/query','U3_reference':'delete all action chunks from identical typed serialization','U4_hard_gate_positions':'compact positions excluding competitor action tokens; original positions also evaluated','U4_primary_reference':'delete competitor rule action tokens; options retained and reindex positions normally','U4_secondary_reference':'same deletion preserving original positions; diagnostic only'}
 dump(out/'manifest.json',manifest)
 # U1 captures actual post-RoPE keys in every layer, before cache or attention.
 module=importlib.import_module(model.model.layers[0].self_attn.__class__.__module__);orig=module.apply_rotary_pos_emb;capture={'mode':None,'layer':0,'keys':[],'end':0,'max':0.0}
 def rotary(*args,**kwargs):
  q,k=orig(*args,**kwargs)
  if capture['mode']=='save':capture['keys'].append(k[:,:,:capture['end']].detach().clone())
  elif capture['mode']=='compare':
   old=capture['keys'][capture['layer']];new=k[:,:,:capture['end']];capture['max']=max(capture['max'],float((old-new).abs().max()))
  capture['layer']+=1;return q,k
 module.apply_rotary_pos_emb=rotary
 try:
  for tree in sorted({groups[i]['NC']['tree'] for i in ids}):
   for cid in [i for i in ids if groups[i]['NC']['tree']==tree][:5]:
    r=groups[cid]['NC'];one=engine.pack(r,extra_patient=' Additional recorded age: 50 years.');two=engine.pack(r,extra_patient=' Additional recorded age: 60 years.');assert len(one['ids'])==len(two['ids'])
    capture.update(mode='save',layer=0,keys=[],end=one['guide_end'],max=0.0);engine.score(one,'blind')
    capture.update(mode='compare',layer=0);engine.score(two,'blind');u1.append({'id':cid,'tree':tree,'post_rope_key_max_abs_difference':capture['max']});capture.update(mode=None,keys=[])
 finally:module.apply_rotary_pos_emb=orig
 with (out/'records.jsonl').open('w',buffering=1) as f:
  for index,cid in enumerate(ids):
   vs=groups[cid];preds={};probs={};visible_ids=[];branch_logits={}
   for volume in ['NC','IC','AC','SW']:
    pack=engine.pack(vs[volume]);visible_ids.append([x for x,a in zip(pack['ids'],pack['roles']) if not a]);z=engine.score(pack,'blind');preds[volume],probs[volume]=engine.choose(z,['1','2']);branch_logits[volume]=[float(z[tok.encode(x,add_special_tokens=False)[0]]) for x in ['1','2']]
   n=vs['NC'];pack=engine.pack(n)
   altered=copy.deepcopy(pack);replacement=tok.encode('Z',add_special_tokens=False)[0]
   for j,role in enumerate(altered['roles']):
    if role:altered['ids'][j]=replacement
   zz=engine.score(altered,'blind');fixed_delta=max(abs(float(zz[tok.encode(x,add_special_tokens=False)[0]])-branch_logits['NC'][i]) for i,x in enumerate(['1','2']))
   visible_equal=all(x==visible_ids[0] for x in visible_ids)
   deleted,_=engine.choose(engine.score(pack,'blind',prune=True),['1','2'])
   sw=vs['SW'];g=int(sw['gold'])-1;action=next(q['action'] for q in sw['rules'] if q['bid']==sw['gold']);gold=chr(65+sw['options'].index(action));labels=[chr(65+i) for i in range(len(sw['options']))];pack=engine.pack(sw,task='action')
   gate,gp=engine.choose(engine.score(pack,'execute',selected=g,hard=True,compact_hard_positions=True),labels)
   original,op=engine.choose(engine.score(pack,'execute',selected=g,hard=True),labels)
   prune,pp=engine.choose(engine.score(pack,'execute',selected=g,prune=True),labels)
   aligned,ap=engine.choose(engine.score(pack,'execute',selected=g,prune=True,preserve_positions=True),labels)
   r={'model':a.name,'id':cid,'tree':n['tree'],'blind_gold':n['gold'],'U2_blind_predictions':preds,'U2_all_four_agree':len(set(preds.values()))==1,'U2_blind_probabilities':probs,'U2_visible_tokens_identical':visible_equal,'U2_fixed_shape_content_logit_max_difference':fixed_delta,'U2_branch_logits':branch_logits,'U3_text_hidden_prediction':deleted,'U3_mask_prediction':preds['NC'],'U4_gold_option':gold,'U4_original_position_hard_prediction':original,'U4_original_position_hard_probabilities':op,'U4_hard_gate_prediction':gate,'U4_plain_prune_prediction':prune,'U4_position_preserved_prune_prediction':aligned,'U4_gate_probabilities':gp,'U4_plain_prune_probabilities':pp,'U4_position_preserved_prune_probabilities':ap};records.append(r);f.write(json.dumps(r)+'\n')
   if (index+1)%20==0:dump(out/'status.json',{'cases_completed':index+1,'seconds':time.time()-start,'peak_GiB':torch.cuda.max_memory_allocated()/2**30});print(json.dumps({'cases_completed':index+1}),flush=True)
 u2=sum(r['U2_all_four_agree'] for r in records)/len(records)
 u3mask=sum(r['U3_mask_prediction']==r['blind_gold'] for r in records)/len(records)*100;u3text=sum(r['U3_text_hidden_prediction']==r['blind_gold'] for r in records)/len(records)*100
 u4gate=sum(r['U4_hard_gate_prediction']==r['U4_gold_option'] for r in records)/len(records)*100;u4prune=sum(r['U4_plain_prune_prediction']==r['U4_gold_option'] for r in records)/len(records)*100
 result={'model':a.name,'U1_max_abs_difference':max(r['post_rope_key_max_abs_difference'] for r in u1),'U1_n':len(u1),'U2_all_four_argmax_agreement':u2,'U3_mask_accuracy_pct':u3mask,'U3_text_hidden_accuracy_pct':u3text,'U3_gap_pp':abs(u3mask-u3text),'U4_hard_gate_accuracy_pct':u4gate,'U4_plain_prune_accuracy_pct':u4prune,'U4_gap_pp':abs(u4gate-u4prune),'U4_prediction_agreement':sum(r['U4_hard_gate_prediction']==r['U4_plain_prune_prediction'] for r in records)/len(records),'U4_position_preserved_prediction_agreement':sum(r['U4_original_position_hard_prediction']==r['U4_position_preserved_prune_prediction'] for r in records)/len(records),'U4_position_preserved_max_probability_difference':max(max(abs(a-b) for a,b in zip(r['U4_original_position_hard_probabilities'],r['U4_position_preserved_prune_probabilities'])) for r in records)}
 result['U4_original_position_hard_accuracy_pct']=100*sum(r['U4_original_position_hard_prediction']==r['U4_gold_option'] for r in records)/len(records)
 result['U4_aligned_gate_prune_max_probability_difference']=max(max(abs(a-b) for a,b in zip(r['U4_gate_probabilities'],r['U4_plain_prune_probabilities'])) for r in records)
 result['U2_max_fixed_shape_content_logit_difference']=max(r['U2_fixed_shape_content_logit_max_difference'] for r in records)
 result['U2_visible_tokens_identical']=all(r['U2_visible_tokens_identical'] for r in records)
 result['U2_rate_threshold_removed_by_user']=True
 result['passes']={'U1':result['U1_max_abs_difference']==0,'U2':result['U2_visible_tokens_identical'] and result['U2_max_fixed_shape_content_logit_difference']==0,'U3':result['U3_gap_pp']<=2,'U4':result['U4_gap_pp']<=1};result['all_pass']=all(result['passes'].values())
 dump(out/'summary.json',result);dump(out/'U1_keys.json',u1);manifest.update(status='completed',wall_seconds=time.time()-start,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30);dump(out/'manifest.json',manifest);dump(out/'DONE.json',{'status':'completed'});print(json.dumps(result))
if __name__=='__main__':main()
