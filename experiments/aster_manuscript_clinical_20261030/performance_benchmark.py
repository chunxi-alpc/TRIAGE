"""Development-only exact-output, attention and RNG comparison for execution speed."""
import argparse,contextlib,copy,json,time
from pathlib import Path
import torch
from .data import BASE,read,write
from .classify import model_and_classifier
from .freeze import classification_records
from .engine import ClinicalExecutor,METHODS
from .official_score import LocalOfficialJudge
from .prefill_replay import replay_prefill
from src.aster.manuscript_core import load_expert

def main():
 p=argparse.ArgumentParser();p.add_argument('--graphs',action='store_true');a=p.parse_args();r=BASE;out=r/'performance01';torch.set_num_threads(4);torch.manual_seed(20261030)
 tok,model,_=model_and_classifier();freeze=json.loads((r/'frozen.json').read_text());freeze.update(native_attention_query_chunk_size=2048,native_attention_balanced=True)
 expert=load_expert(r/'analysis/expert.json');records={x['case_id']:x for x in classification_records(r) if x['status']=='complete'}
 cases=read(r/'cases/development.jsonl');seen=set();selected=[]
 for c in cases:
  if c['source_id'] not in seen:selected.append(c);seen.add(c['source_id'])
 # Before looking at any benchmark result: first original development case per source.
 selected=selected[:2];result={'selection':'first development case per source, no official-test correctness selection','cases':len(selected),'graphs':a.graphs,'comparisons':[],'scores':[]}
 def save():write(out/('graph_benchmark.json' if a.graphs else 'exact_benchmark.json'),result)
 graph=contextlib.nullcontext()
 if a.graphs:
  from .decode_graphs import decode_mlp_graphs
  graph=decode_mlp_graphs(model)
 baseline={};judge=LocalOfficialJudge(model,tok)
 for ci,c in enumerate(selected):
  engine=ClinicalExecutor(model,tok,expert,freeze)
  raw=records[c['case_id']]['raw_decisions']
  for method in METHODS:
   x=engine.execute(c,raw,method);baseline[ci,method]=x
  text=baseline[ci,'spectral']['text'];t=time.perf_counter();score=judge.score(c,text,20261030+ci);torch.cuda.synchronize()
  baseline[ci,'score']=(score,time.perf_counter()-t,torch.cuda.get_rng_state().clone(),torch.get_rng_state().clone())
  save()
 if a.graphs:
  prompt=tok.apply_chat_template([{'role':'user','content':selected[0]['patient']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
  inputs=tok(prompt,return_tensors='pt',add_special_tokens=False).to(model.device)
  reference_fixed=[]
  with torch.inference_mode():
   state=model(**inputs,use_cache=True,logits_to_keep=1)
   for token in [42,1234,51]:
    state=model(input_ids=torch.tensor([[token]],device=model.device),past_key_values=state.past_key_values,use_cache=True,logits_to_keep=1)
    reference_fixed.append((state.logits.clone(),[(k.clone(),v.clone()) for k,v in state.past_key_values.to_legacy_cache()]))
 with graph as graphs:
  if a.graphs:
   with torch.inference_mode():
    state=model(**inputs,use_cache=True,logits_to_keep=1);errors=[]
    for j,token in enumerate([42,1234,51]):
     state=model(input_ids=torch.tensor([[token]],device=model.device),past_key_values=state.past_key_values,use_cache=True,logits_to_keep=1)
     ref_logits,ref_cache=reference_fixed[j]
     errors.append({'logits':float((state.logits-ref_logits).abs().max()),'cache_max':max(float((a-b).abs().max()) for x,y in zip(state.past_key_values.to_legacy_cache(),ref_cache) for a,b in zip(x,y))})
   result['fixed_token_comparisons']=errors;save()
  for ci,c in enumerate(selected):
   engine=ClinicalExecutor(model,tok,expert,freeze,exact_execution=True);raw=records[c['case_id']]['raw_decisions']
   for method in METHODS:
    x=engine.execute(c,raw,method);old=baseline[ci,method]
    equal=x['token_ids']==old['token_ids'] and x['text']==old['text'] and x['attention']==old['attention'] and x['edit']==old['edit']
    error=float((x['first_logits']-old['first_logits']).abs().max()) if 'first_logits' in x else None
    result['comparisons'].append({'development_source_index':ci,'method':method,'tokens_attention_edit_equal':equal,'first_logits_max_error':error,'reference_seconds':old['seconds'],'optimized_seconds':x['seconds'],'tokens':len(x['token_ids'])});save()
   old,seconds,cuda_rng,cpu_rng=baseline[ci,'score'];t=time.perf_counter()
   with replay_prefill(model) as replay:
    new=judge.score(c,baseline[ci,'spectral']['text'],20261030+ci);perf=replay.stats()
   torch.cuda.synchronize();duration=time.perf_counter()-t
   fields=['status','path_draws','outcome_draws','Qwen_judged_official_accuracy','Qwen_judged_official_PCC']
   row={'development_source_index':ci,'all_draws_equal':all(new[k]==old[k] for k in fields),'cuda_rng_equal':torch.equal(cuda_rng,torch.cuda.get_rng_state()),'cpu_rng_equal':torch.equal(cpu_rng,torch.get_rng_state()),'reference_seconds':seconds,'optimized_seconds':duration,'prefill':perf}
   result['scores'].append(row);save()
  if graphs is not None and hasattr(graphs,'summary'):result['graphs_statistics']=graphs.summary()
 result['status']='passed' if all(x['tokens_attention_edit_equal'] and x['first_logits_max_error'] in [None,0.] for x in result['comparisons']) and all(x['all_draws_equal'] and x['cuda_rng_equal'] and x['cpu_rng_equal'] for x in result['scores']) else 'failed'
 if a.graphs and any(x['logits'] or x['cache_max'] for x in result['fixed_token_comparisons']):result['status']='failed'
 result['peak_memory_gb']=torch.cuda.max_memory_allocated()/1e9;save()
if __name__=='__main__':main()
