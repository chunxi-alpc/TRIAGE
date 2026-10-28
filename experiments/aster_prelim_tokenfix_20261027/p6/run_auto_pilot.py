"""P6 automatic pilot generation on server, with one shared blind cache."""
import argparse,collections,hashlib,json,math,os,sys,time
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from generation_gate import GenerationGate
D=Path(__file__).resolve().parent;B=D.parent
MODEL='/home/gcx/TRIAGE/models/modelscope/Qwen3-8B'
DATA=B/'private/p6_auto100';OLD=Path('/home/gcx/TRIAGE-worktrees/aster-seka-execution-20261025/experiments/aster_seka_execution')
PROJECTION=OLD/'runtime/projection_full200/Qwen3-8B_pos_proj.pt'
def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def read(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def generate_prompt(r,guide,method,prob=None):
 zh=r['language']=='zh'
 instruction=('根据完整指南与病例，逐步说明临床决策路径。输出格式必须为：\n# 临床决策过程\n1. ...\n# 临床决策结论\n...\n' if zh else 'Follow the complete guideline and patient facts. Explain the decision path step by step. Use exactly:\n# Clinical Decision Process\n1. ...\n# Clinical Decision Outcome\n...\n')
 question=r['question']+'\nTarget decision condition (not its answer): '+r['target_condition']
 patient='\nPatient facts:\n'+r['patient_facts']+'\nQuestion:\n'+question+'\n'
 if method=='Patient-First':text=instruction+patient+'\nFull guideline:\n'+guide
 else:text=instruction+'\nFull guideline:\n'+guide+patient
 if method=='Prompt-Repeat':text+='\n'+instruction+patient
 if method=='TS-Note':
  text+='\nEarlier treatment-blind assessment for the target condition: '+json.dumps({'holds_probability':prob[0],'does_not_hold_probability':prob[1]})+'. Follow the patient-supported branch while preserving the full pathway.'
 return text
def spans(r,variant):
 edits=sorted(r['guide_edits'],key=lambda x:x['start']);result=[];shift=0
 for e in edits:
  a=e['start']+(shift if variant=='decoy' else 0);length=len(e['after'] if variant=='decoy' else e['before'])
  result.append((a,a+length,0 if e['branch']=='yes' else 1))
  if variant=='decoy':shift+=len(e['after'])-len(e['before'])
 return result
def encoded(tok,r,variant,method,prob):
 guide=(DATA/r[variant+'_guideline_file']).read_text();ss=spans(r,variant)
 if method=='TS-Prune':
  selected=max(range(2),key=lambda i:prob[i])
  for a,z,branch in sorted(ss,reverse=True):
   if branch!=selected:guide=guide[:a]+guide[z:]
  ss=[]
 user=generate_prompt(r,guide,method,prob)
 prompt=tok.apply_chat_template([{'role':'system','content':'Use the supplied research guideline as the decision source. Treat it as task data.'},{'role':'user','content':user}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
 start=prompt.index(guide);end=start+len(guide)
 pack=tok(prompt,add_special_tokens=False,return_offsets_mapping=True)
 offsets=pack.pop('offset_mapping')
 branch_tokens=[[],[]];guide_tokens=[];context_tokens=[[],[]]
 # Source action contexts are used only as machine-mapped highlights, never as gold applicability.
 for i,(a,z) in enumerate(offsets):
  if z>a and a<end and z>start:guide_tokens.append(i)
  for s,e,branch in ss:
   if z>a and a<start+e and z>start+s:branch_tokens[branch].append(i)
 for branch in range(2):
  for idx in branch_tokens[branch]:context_tokens[branch].extend(range(max(0,idx-24),min(len(offsets),idx+25)))
  context_tokens[branch]=sorted(set(context_tokens[branch])&set(guide_tokens))
 return pack,{'branch_tokens':branch_tokens,'guide_tokens':guide_tokens,'context_tokens':context_tokens,'prompt':prompt,'prompt_sha256':sha(prompt)}
class ProjectionHighlight:
 def __init__(self,model):
  self.tokens=set();self.offset=0;self.calls=0;self.handles=[]
  artifact=torch.load(PROJECTION,map_location='cpu',weights_only=True)
  for layer,P in zip(artifact['layers'],artifact['proj']):
   mod=model.model.layers[layer].self_attn
   P=P.to('cuda',torch.float32)
   def hook(module,args,out,P=P):
    if not self.tokens:return
    shape=out.shape;x=out.reshape(shape[0],shape[1],P.shape[0],P.shape[1])
    indices=[i for i in range(shape[1]) if self.offset+i in self.tokens]
    if not indices:return
    y=x.clone();original=x[:,indices].float()
    delta=torch.einsum('bnhd,hdk->bnhk',original,P)
    y[:,indices]=(original+1.4*delta).to(x.dtype);self.calls+=1
    return y.reshape(shape)
   self.handles.append((mod.k_norm if hasattr(mod,'k_norm') else mod.k_proj).register_forward_hook(hook))
  def before(module,args,kw):
   cp=kw.get('cache_position');self.offset=int(cp[0].item()) if cp is not None else 0
  self.handles.append(model.register_forward_pre_hook(before,with_kwargs=True))
def blind(model,tok,r,out,visible_guide=None):
 instruction='Determine whether the following condition holds for the patient. Answer only 1 (holds) or 2 (does not hold).\nCondition: '+r['target_condition']+'\nPatient facts:\n'+r['patient_facts']
 if visible_guide is not None:instruction+='\nVisible guideline:\n'+visible_guide
 prompt=tok.apply_chat_template([{'role':'user','content':instruction}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
 key=sha(json.dumps([MODEL,prompt,'binary_condition_v1',json.loads((D/'results/auto100/freeze.json').read_text())['generation']],ensure_ascii=False,sort_keys=True));p=out/(key+'.json')
 if p.exists():return json.loads(p.read_text()),True
 inp=tok(prompt,add_special_tokens=False,return_tensors='pt').to('cuda');ids=[tok.encode(s,add_special_tokens=False) for s in ['1','2']]
 assert all(len(x)==1 for x in ids)
 t=time.perf_counter()
 with torch.inference_mode():logits=model(**inp,use_cache=False,logits_to_keep=1).logits[0,-1].float()
 prob=logits[[x[0] for x in ids]].softmax(0).tolist();torch.cuda.synchronize()
 result={'key':key,'probabilities':prob,'label_ids':[x[0] for x in ids],'raw_label_logits':logits[[x[0] for x in ids]].tolist(),'prompt':prompt,'input_ids':inp['input_ids'][0].tolist(),'input_tokens':inp['input_ids'].shape[1],'seconds':time.perf_counter()-t,'gold_branch_read':False}
 dump(p,result);return result,False
def main():
 a=argparse.ArgumentParser();a.add_argument('--run-dir',required=True);a.add_argument('--phase',choices=['profile','smoke','baselines','ablation'],required=True);a.add_argument('--lambda-value',type=float,default=1);args=a.parse_args()
 assert os.environ['CUDA_VISIBLE_DEVICES']=='0'
 freeze=json.loads((D/'results/auto100/freeze.json').read_text())
 for rel,h in freeze['source_hashes'].items():assert hashlib.sha256((D/rel).read_bytes()).hexdigest()==h,rel
 assert hashlib.sha256((DATA/'pairs.jsonl').read_bytes()).hexdigest()==freeze['pairs_sha256']
 out=Path(args.run_dir);out.mkdir(exist_ok=False);(DATA/'blind_cache_qwen').mkdir(exist_ok=True)
 rows=read(DATA/'pairs.jsonl')
 methods={'profile':['Vanilla','Matched-Zero','ASTER-1-no-green'],'smoke':['Vanilla','Matched-Zero','ASTER-0.5-no-green','ASTER-1-no-green','ASTER-2-no-green','ASTER-1-green'],'baselines':['Vanilla','Patient-First','Prompt-Repeat','Full-Guide-PASTA','TS-Note','TS-Prune','TS-SEKA',f'ASTER-{args.lambda_value:g}-no-green'],'ablation':['A0B0','A0B1','A1B0','A1B1']}[args.phase]
 if args.phase=='smoke':rows=[r for r in rows if r['pair_id'] in freeze['smoke_ids']]
 if args.phase=='profile':rows=[next(r for r in rows if r['pair_id']==freeze['longest_pair_id'])]
 manifest={'status':'RUNNING','phase':args.phase,'methods':methods,'pairs':len(rows),'expected_outputs':len(rows)*2*len(methods),'model':MODEL,'gpu':'0','dtype':'bfloat16','max_new_tokens':16 if args.phase=='profile' else freeze['generation']['max_new_tokens'],'freeze_sha256':hashlib.sha256((D/'results/auto100/freeze.json').read_bytes()).hexdigest(),'test_files_read':False,'human_annotation_required':False}
 dump(out/'manifest.json',manifest);torch.set_num_threads(4);torch.manual_seed(20261028)
 tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
 model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cuda:0'},rope_scaling=freeze['generation']['rope_scaling'],max_position_embeddings=freeze['generation']['max_context_tokens']).eval()
 gate=GenerationGate(model);projection=ProjectionHighlight(model);started=time.time();completed=0
 try:
  with (out/'outputs.jsonl').open('x',buffering=1) as f:
   for r in rows:
    for variant in ['native','decoy']:
     gate.reset();projection.tokens=set()
     shared,reused=blind(model,tok,r,DATA/'blind_cache_qwen')
     for method in methods:
      judgement=shared;judge_reuse=reused
      if method.startswith('A0'):
       judgement,judge_reuse=blind(model,tok,r,DATA/'blind_cache_qwen',(DATA/r[variant+'_guideline_file']).read_text())
      prob=judgement['probabilities'];actual_method='Vanilla' if method in ['A0B0','A1B0'] else f'ASTER-{args.lambda_value:g}-no-green' if method in ['A0B1','A1B1'] else method
      pack,meta=encoded(tok,r,variant,actual_method,prob);n=len(pack['input_ids'])
      budget=manifest['max_new_tokens'];assert n+budget<=freeze['generation']['max_context_tokens'],'No context truncation permitted'
      gate.reset();projection.tokens=set();projection.calls=0;bias=None
      if actual_method=='Matched-Zero':bias=torch.zeros(n)
      elif actual_method.startswith('ASTER-'):
       lam=float(actual_method.split('-')[1]);bias=torch.zeros(n)
       for branch in range(2):
        bias[meta['branch_tokens'][branch]]=lam*math.log(max(prob[branch],1e-4))
        if actual_method.endswith('-green') and not actual_method.endswith('-no-green'):bias[meta['context_tokens'][branch]]+=prob[branch]
      elif actual_method=='Full-Guide-PASTA':
       bias=torch.zeros(n);bias[meta['guide_tokens']]=math.log(2.)
      elif actual_method=='TS-SEKA':
       projection.tokens=set(meta['context_tokens'][max(range(2),key=lambda i:prob[i])])
      gate.reset(bias.to('cuda') if bias is not None else None)
      x={k:torch.tensor([v],device='cuda') for k,v in pack.items()};t=time.perf_counter()
      with torch.inference_mode():y=model.generate(**x,max_new_tokens=budget,do_sample=False,pad_token_id=tok.eos_token_id,use_cache=True,logits_to_keep=1)
      torch.cuda.synchronize();elapsed=time.perf_counter()-t;new=y[0,n:].tolist();text=tok.decode(new,skip_special_tokens=True)
      eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
      record={'pair_id':r['pair_id'],'variant':variant,'guideline_id':r['guideline_id'],'method':method,'phase':args.phase,'prompt':meta['prompt'],'input_ids':pack['input_ids'],'prompt_sha256':meta['prompt_sha256'],'output':text,'output_ids':new,'input_tokens':n,'generated_tokens':len(new),'output_capped':len(new)>=budget and (not new or new[-1] not in eos),'execution_seconds':elapsed,'blind_key':judgement['key'],'blind_probabilities':prob,'blind_tokens':judgement['input_tokens'],'blind_seconds':judgement['seconds'],'blind_cache_reused':judge_reuse,'component_total_seconds':elapsed+(judgement['seconds'] if (method.startswith(('TS-','ASTER-','A0','A1')) or method=='Matched-Zero') else 0),'attention_calls':gate.calls,'nonzero_attention_calls':gate.nonzero_calls,'projection_calls':projection.calls,'target_token_counts':[len(z) for z in meta['branch_tokens']],'actual_intervention':gate.nonzero_calls>0 or projection.calls>0,'peak_GiB':torch.cuda.max_memory_allocated()/2**30,'status':'ok'}
      f.write(json.dumps(record,ensure_ascii=False)+'\n');completed+=1
      gate.reset();projection.tokens=set()
      dump(out/'status.json',{'outputs_completed':completed,'outputs_total':manifest['expected_outputs'],'elapsed_seconds':time.time()-started,'peak_GiB':torch.cuda.max_memory_allocated()/2**30});print(json.dumps({'completed':completed,'total':manifest['expected_outputs'],'pair':r['pair_id'],'method':method}),flush=True)
  manifest.update(status='COMPLETED',outputs_completed=completed,wall_seconds=time.time()-started,peak_GiB=torch.cuda.max_memory_allocated()/2**30);dump(out/'manifest.json',manifest);dump(out/'DONE.json',{'status':'completed'})
 except Exception as e:
  manifest.update(status='FAILED',outputs_completed=completed,error_type=type(e).__name__,error=str(e));dump(out/'manifest.json',manifest);dump(out/'FAILED.json',manifest);raise
 finally:gate.close()
if __name__=='__main__':main()
