"""Batch all five path and five conclusion judge draws, retaining every response."""
import torch,json
from pathlib import Path
from official_judge import Judge,parse_record,pcc,acc,dynamic_time_warping
class BatchJudge(Judge):
 def score_many(self,items,batch_size=8,cache_dirs=None):
  results=[None]*len(items);jobs=[];prompts=[];contexts={};cached={}
  cache_dirs=cache_dirs or [None]*len(items)
  for i,(c,raw) in enumerate(items):
   try:
    _,split,gid,pid=c['case_id'].split(':');gid=int(gid);pid=int(pid);refs=self.refs[split]
    ref=next(v for k,v in refs.items() if k[0]==gid and k[1]==pid)
    record={'id':c['case_id'],'case_path_id':gid,'path_index':pid,'filename':ref['filename'],'language':ref['language'],'llm_output':raw}
    parsed=parse_record(record);reference=pcc.load_reference_steps(ref);candidate=pcc.get_candidate_steps(parsed)
    path_prompt=pcc.build_prompt(ref['language'],reference,candidate)
    template=acc.TEMPLATE_ZH if ref['language']=='zh' else acc.TEMPLATE_EN
    accuracy_prompt=acc.render_template(template,{'Reference_Conclusion':acc.load_reference_conclusion(ref),'Candidate_Conclusion':acc.candidate_conclusion(parsed)})
    contexts[i]={'record':record,'parsed':parsed,'reference':reference,'candidate':candidate,'path':[],'accuracy':[]}
    for kind,prompt in [('path',path_prompt),('accuracy',accuracy_prompt)]:
     for repeat in range(5):
      text=self.e.tok.apply_chat_template([{'role':'system','content':prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
      cache=Path(cache_dirs[i])/(kind+'_'+str(repeat)+'.json') if cache_dirs[i] else None
      if cache and cache.exists():
       saved=json.loads(cache.read_text())
       if saved['prompt']!=text:raise RuntimeError('judge cache prompt mismatch')
       cached[(i,kind,repeat)]=RuntimeError(saved['response']) if saved.get('technical_failure') else saved['response']
      else:jobs.append((i,kind,repeat));prompts.append(text)
   except Exception as ex:results[i]=(False,True,{'official_pcc':1.,'official_accuracy':0.,'official_failure':str(ex)})
  def persist(index,raw):
   i,kind,repeat=jobs[index]
   if cache_dirs[i]:
    folder=Path(cache_dirs[i]);folder.mkdir(parents=True,exist_ok=True);path=folder/(kind+'_'+str(repeat)+'.json');temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps({'prompt':prompts[index],'response':str(raw),'technical_failure':isinstance(raw,Exception)},ensure_ascii=False));temp.replace(path)
  outputs=self.e.generate_many(prompts,8192,sample=True,batch_size=batch_size,on_result=persist)
  cached.update({job:raw for job,raw in zip(jobs,outputs)})
  for i in contexts:
   for kind in ('path','accuracy'):
    contexts[i][kind]=[cached[(i,kind,repeat)] for repeat in range(5)]
  for i,ctx in contexts.items():
   path_raw=ctx['path'];accuracy_raw=ctx['accuracy'];record=ctx['record']
   def base():return {k:record[k] for k in ['id','case_path_id','path_index','filename','language']}|{'eval_method':'qwen','reference_source':'gt_paths'}
   path_result=base();accuracy_result=base();cost=1.;accuracy=0.
   try:
    if len(path_raw)!=5:raise RuntimeError('path judge draw denominator is not five')
    for x in path_raw:
     if isinstance(x,Exception):raise x
    matrices=[pcc.parse_pairwise_score_matrix(x,ctx['reference'],ctx['candidate']) for x in path_raw]
    path_result['results']=matrices;matrix=torch.tensor(matrices).mean(0);_,cost,_=dynamic_time_warping((1-(matrix-1)/4).tolist())
   except Exception as ex:path_result.update(skipped=True,reason=str(ex))
   try:
    if len(accuracy_raw)!=5:raise RuntimeError('conclusion judge draw denominator is not five')
    for x in accuracy_raw:
     if isinstance(x,Exception):raise x
    scores=[acc.parse_accuracy_score(x) for x in accuracy_raw];accuracy_result['results']=scores;accuracy=sum((x-1)/4 for x in scores)/5
   except Exception as ex:accuracy_result.update(skipped=True,reason=str(ex))
   metrics={'official_pcc':cost,'official_accuracy':accuracy,'official_pathway_judge':path_result,'official_accuracy_judge':accuracy_result,'official_judge_failed':bool(path_result.get('skipped') or accuracy_result.get('skipped')),'parsed_native_output':ctx['parsed']['llm_output'],'judge_draws_requested':{'path':5,'accuracy':5},'raw_path_judge_responses':[str(x) for x in path_raw],'raw_accuracy_judge_responses':[str(x) for x in accuracy_raw]}
   results[i]=(accuracy==1.,False,metrics)
  return results
