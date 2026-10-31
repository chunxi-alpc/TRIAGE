"""Conditional real K/V replacements and pre-RoPE difference transfer."""
import argparse,json,time
from collections import defaultdict
import torch
from .run import BASE,read,write
from .module_one_data import M1
from .runtime import execute
from src.aster.manuscript_core import rotate
from experiments.aster_manuscript_20261030.controls import directions_for

def extract_pair(engine,row):
 target=row['target_rule_id'];out={}
 for state in ['positive','negative']:
  prompt=engine.builder.prepare(row['guideline'],row['states'][state],row['question'],online=False)
  prompt.prefix_length=prompt.input_ids.shape[1]-1
  prefix=engine.backend.prefill(prompt);ids=list(prompt.positions[target]);heads={}
  for layer,(k,v) in enumerate(prefix.cache.to_legacy_cache()):
   if layer==0:continue
   for head in range(engine.backend.n_kv):
    pre=rotate(k[0,head,ids].detach().double().cpu(),prefix.cos[ids],-prefix.sin[ids])
    heads[layer,head]={'key':pre,'rotated_key':k[0,head,ids].detach().cpu(),'value':v[0,head,ids].detach().cpu()}
  out[state]={'heads':heads,'ids':prompt.input_ids[0,ids].tolist(),'positions':ids}
  del prefix
 if out['positive']['ids']!=out['negative']['ids'] or out['positive']['positions']!=out['negative']['positions']:
  raise ValueError('Pair target positions do not align')
 return out

def apply_pair_patch(cache,prompt,prefix,target,recipient,donor,means,mode,sign):
 ids=list(prompt.positions[target])
 if prompt.input_ids[0,ids].tolist()!=recipient['ids'] or recipient['ids']!=donor['ids']:
  raise ValueError('Online/offline target token mapping differs')
 for (layer,head),source in donor['heads'].items():
  k=cache.layers[layer].keys;v=cache.layers[layer].values
  cos=prefix.cos[ids].to(k.device);sin=prefix.sin[ids].to(k.device)
  src=source['key'].to(k.device);dest=recipient['heads'][layer,head]['key'].to(k.device)
  if mode in ['key','kv']:
   if donor.get('positions') is not None and donor['positions']!=ids:raise ValueError('Direct replacement requires identical absolute positions')
   k[0,head,ids]=source['rotated_key'].to(k.device,dtype=k.dtype) if 'rotated_key' in source else rotate(src,cos,sin).to(k.dtype)
  elif mode=='delta':
   k[0,head,ids]=(k[0,head,ids].double()+rotate(src-dest,cos,sin)).to(k.dtype)
  elif mode=='mean':
   unit=means[layer,head].to(k.device);unit=unit/unit.norm()
   delta=(src-dest).norm(dim=-1,keepdim=True)*unit[None,:]*sign
   k[0,head,ids]=(k[0,head,ids].double()+rotate(delta,cos,sin)).to(k.dtype)
  if mode in ['value','kv']:v[0,head,ids]=source['value'].to(v.device,dtype=v.dtype)
 return cache

def run_pair(engine,row,skip=None):
 from experiments.aster_abc_diagnostics_20261030.c.audit import strict_parse
 snapshots=extract_pair(engine,row)
 expert=dict(engine.expert);expert['selected_heads']=[str(l)+':'+str(h) for l in range(1,len(engine.backend.blocks)) for h in range(engine.backend.n_kv)]
 means=directions_for(expert,'mean')
 for state in ['positive','negative']:
  other='negative' if state=='positive' else 'positive';target=row['target_rule_id']
  true=target if state=='positive' else row['verification']['negative_rule_id']
  donor_rule=target if other=='positive' else row['verification']['negative_rule_id']
  case={'case_id':row['pair_id']+':'+state,'patient':row['states'][state],
   'guideline':row['guideline'],'question':row['question'],'gold_rules':[true]}
  def patch(cache,prompt,prefix,spec):
   return apply_pair_patch(cache,prompt,prefix,target,snapshots[state],snapshots[other],means,spec['patch_mode'],1 if other=='positive' else -1)
  engine.paired_patch=patch
  for order,modes in [('patient_first',['zero','key','value','kv','mean']),('guide_first',['zero','delta','mean'])]:
   for mode in modes:
    if skip and (case['case_id'],order+'_'+mode) in skip:continue
    spec={'name':order+'_'+mode,'order':order,'kind':'zero' if mode=='zero' else 'paired_patch','patch_mode':mode}
    result=execute(engine,case,spec)
    _,path,invalid=strict_parse(result['text'])
    yield {'case_id':case['case_id'],'pair_id':row['pair_id'],'source_id':row['source_id'],'state':state,'order':order,
     'arm':spec['name'],'mode':mode,'status':'complete','donor_rule':donor_rule,
     'recipient_root_correct':bool(path) and path[0]==true,'donor_root_selected':bool(path) and path[0]==donor_rule,
     'invalid':invalid,'donor_margin':-result['decision_margin'],'result':result,
     'scope':'280 heads; target rule identifier/action only; mean edit matches each true key-difference norm'}
 del engine.paired_patch

def summarize():
 rs=[]
 for shard in range(4):
  p=M1/f'diagnostic_{shard}.jsonl'
  if p.exists():rs+=read(p)
 rs=list({(r['case_id'],r['arm']):r for r in rs}.values())
 groups=defaultdict(list);zero={(r['case_id'],r['order']):r for r in rs if r['mode']=='zero'}
 for r in rs:groups[r['order'],r['mode']].append(r)
 result=[]
 for (order,mode),items in groups.items():
  result.append({'order':order,'mode':mode,'n':len(items),'recipient_root_correct':sum(r['recipient_root_correct'] for r in items),
   'donor_root_selected':sum(r['donor_root_selected'] for r in items),
   'mean_donor_margin':sum(r['donor_margin'] for r in items)/len(items),
   'mean_margin_change':sum(r['donor_margin']-zero[r['case_id'],order]['donor_margin'] for r in items)/len(items)})
 write(M1/'diagnostic_summary.json',{'rows':result,'pairs':4,'sources':2,'independence':'exposed age pairs, opposite targets are not independent patients'})

def worker(shard):
 from .module_one import make_engine
 from experiments.aster_manuscript_clinical_20261030.native_memory import query_chunked_attention
 engine=make_engine();pairs=[r for r in read(BASE/'review/accepted_pairs.jsonl') if r['split']=='confirm']
 output=M1/f'diagnostic_{shard}.jsonl';done={(r['case_id'],r['arm']) for r in read(output)} if output.exists() else set()
 start=time.time();count=0
 with query_chunked_attention(engine.model,256),torch.inference_mode(),output.open('a') as f:
  for i,pair in enumerate(pairs):
   if i%4!=shard:continue
   expected={(pair['pair_id']+':'+state,order+'_'+mode) for state in ['positive','negative'] for order,modes in [('patient_first',['zero','key','value','kv','mean']),('guide_first',['zero','delta','mean'])] for mode in modes}
   if expected<=done:continue
   for row in run_pair(engine,pair,done):
    if (row['case_id'],row['arm']) in done:continue
    f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush();count+=1
 write(M1/f'diagnostic_{shard}_status.json',{'status':'complete','records':count,'elapsed':time.time()-start,'peak_gb':torch.cuda.max_memory_allocated()/1e9})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--shard',type=int);a=p.parse_args();worker(a.shard)
