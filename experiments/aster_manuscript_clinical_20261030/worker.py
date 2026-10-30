"""Clinical development and frozen original-case evaluation, atomic per-unit resume."""
import argparse,json,subprocess,time,traceback
from pathlib import Path
import torch
from src.aster.manuscript_core import load_expert,edit_kv_layers
from .data import read,write,REV
from .classify import model_and_classifier,classify_case
from .freeze import classification_records
from .engine import ClinicalExecutor,METHODS,grade
from .queue import assigned
def atomic(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.partial');write(temporary,obj);temporary.replace(path)
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--phase',choices=['gain','confirm','official_test'],required=True);p.add_argument('--shard',type=int,required=True);p.add_argument('--resume',action='store_true');p.add_argument('--official-judge',action='store_true');p.add_argument('--native-query-chunks',type=int,default=0);p.add_argument('--dynamic',action='store_true');p.add_argument('--native-balanced',action='store_true');p.add_argument('--exact-execution',action='store_true');p.add_argument('--decode-graphs',action='store_true');p.add_argument('--worker-tag');a=p.parse_args()
 execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();run=Path(a.run);out=run/(a.phase+(a.worker_tag or str(a.shard)));out.mkdir(exist_ok=a.resume);torch.set_num_threads(4);torch.manual_seed(20261030);start=time.time()
 freeze=json.loads((run/('clinical_freeze_pre_gain.json' if a.phase=='gain' else 'frozen.json')).read_text());expert=load_expert(run/'analysis/expert.json')
 if a.native_query_chunks:freeze['native_attention_query_chunk_size']=a.native_query_chunks
 freeze['native_attention_balanced']=a.native_balanced
 tok,model,classifier=model_and_classifier();engine=ClinicalExecutor(model,tok,expert,freeze,exact_execution=a.exact_execution)
 cases=read(run/'cases'/('development.jsonl' if a.phase=='gain' else a.phase+'.jsonl'))
 judge=None
 if a.official_judge:
  from .official_score import LocalOfficialJudge,score_case
  judge=LocalOfficialJudge(model,tok,exact_execution=a.exact_execution)
 atomic(out/'execution_manifest.json',{'execution_code_commit':execution_commit,'native_attention_query_chunk_size':a.native_query_chunks,'native_attention_balanced':a.native_balanced,'dynamic_case_queue':a.dynamic,'tensor_batch_size':1,'exact_execution_optimizations':a.exact_execution,'decode_graphs':a.decode_graphs,'official_Qwen_judge':a.official_judge,'model_revision':REV,'frozen_method_code':freeze.get('frozen_code_commit',freeze['code_commit'])})
 cached={r['case_id']:r for r in classification_records(run) if r['status']=='complete'} if a.phase=='gain' else {}
 # Actual clinical prompt comparison to the original reference, before method results.
 ref=next(c for c in read(run/'cases/development.jsonl') if 'source_failure' not in c)
 raw=cached[ref['case_id']]['raw_decisions'] if ref['case_id'] in cached else classify_case(classifier,ref)['raw_decisions']
 zero=engine.execute(ref,raw,'zero',gain=0.)
 biaszero=engine.execute(ref,raw,'score_bias',gain=0.)
 prompt=engine.builder.prepare(ref['guideline'],ref['patient'],ref['question'],online=True);prefix=engine.backend.prefill(prompt)
 with engine.backend.session():
  clone=type(prefix.cache).from_legacy_cache(tuple((k.clone(),v.clone()) for k,v in prefix.cache.to_legacy_cache()))
  native=engine.backend._forward(prompt.input_ids[:,prompt.prefix_length:],cache=clone,start=prompt.prefix_length,use_cache=True).logits[0,-1].float().cpu()
 assert torch.equal(zero['first_logits'],native) and torch.equal(biaszero['first_logits'],native)
 reference={'status':'passed','clinical_prompt_reference':True,'zero_max_logit_error':float((zero['first_logits']-native).abs().max()),'bias_zero_max_logit_error':float((biaszero['first_logits']-native).abs().max()),'model_revision':REV}
 atomic(out/'reference.json',reference)
 graphs=None
 if a.decode_graphs:
  from .decode_graphs import DecodeMLPGraphs
  graphs=DecodeMLPGraphs(model).prepare().install()
 units=0;failures=0;native_units=0
 for i,case,casepath in assigned(cases,run,out,a.shard,a.dynamic,a.official_judge):
  casepath.mkdir(parents=True,exist_ok=True)
  classpath=casepath/'classification.json'
  if classpath.exists():cls=json.loads(classpath.read_text())
  elif case['case_id'] in cached:cls=cached[case['case_id']]
  elif 'source_failure' in case:cls={'status':'failed','error':case['source_failure']}
  else:cls=classify_case(classifier,case);cls['status']='complete';cls['model_revision']=REV;atomic(classpath,cls)
  conditions=[('spectral',g) for g in [.25,1.,4.]] if a.phase=='gain' else [(m,freeze['g_plus']) for m in METHODS]
  for method,gain in conditions:
   name=method+('_'+str(gain) if a.phase=='gain' else '')
   destination=casepath/(name+'.json')
   if destination.exists():
    old=json.loads(destination.read_text());units+=1;failures+=old['status']=='failed';continue
   record={'case_id':case['case_id'],'source_id':case['source_id'],'phase':a.phase,'method':method,'gain':gain,'model_revision':REV,
    'execution_code_commit':execution_commit,'execution_profile':'performance01' if a.exact_execution else 'reference','decode_graphs':a.decode_graphs and method!='native_official','source_split':case['source_split'],'previous_PR40_exposure':case.get('previous_PR40_exposure',False)}
   try:
    if 'source_failure' in case or cls.get('status')=='failed':raise ValueError(case.get('source_failure',cls.get('error','classification unavailable')))
    suspend_graphs=graphs is not None and method=='native_official'
    if suspend_graphs:graphs.restore()
    try:result=engine.execute(case,cls['raw_decisions'],method,gain=gain,capture=a.phase!='gain')
    finally:
     if suspend_graphs:graphs.install()
    result.pop('first_logits',None)
    record.update(status='complete',result=result,classification_failures=cls['classification_failures'],classification_seconds=cls['classification_seconds'],classification_forwards=cls['classification_forwards'])
    if method=='native_official':
     record['score']={'primary_score_applicable':False,'official_semantic_score':'pending Qwen-adapted official scorer'};native_units+=1
    else:record['score']={'primary_score_applicable':True,**grade(case,result['text'])}
   except Exception as e:
    failures+=1;record.update(status='failed',error_type=type(e).__name__,error=str(e),score={'primary_score_applicable':method!='native_official','final_leaf_correct':False,'path_f1':0.,'exact_path':False,'invalid_generation':False,'scoring_failure':False})
    traceback.print_exc()
    if torch.cuda.is_available():torch.cuda.empty_cache()
   atomic(destination,record);units+=1
  if judge is not None:score_case(judge,case,casepath,METHODS,i)
  write(out/'status.json',{'status':'running','completed_units':units,'technical_failures':failures,'registered_cases':len(cases),'dynamic_queue':a.dynamic,'phase':a.phase})
 if graphs is not None:
  atomic(out/'decode_graphs.json',graphs.summary());graphs.restore()
 write(out/'status.json',{'status':'complete','completed_units':units,'technical_failures':failures,'registered_cases':len(cases),'dynamic_queue':a.dynamic,'phase':a.phase,
  'elapsed_seconds':time.time()-start,'peak_memory_gb':torch.cuda.max_memory_allocated()/1e9,'model_revision':REV,'frozen_code_commit':freeze.get('frozen_code_commit',freeze['code_commit']),
  'execution_code_commit':execution_commit})
if __name__=='__main__':main()

