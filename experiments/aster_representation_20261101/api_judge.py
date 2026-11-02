"""Resumable API judging of frozen main-method predictions; CPU/network only."""
import concurrent.futures as cf
import json,os,time,urllib.request,urllib.error,argparse,threading
from pathlib import Path
from experiments.aster_manuscript_clinical_20261030 import official_score as official
BASE=Path('/home/gcx/aster_representation_20261101/official_test_20261102/codex_judging')
OUT=BASE/'api_gpt_6_1_sol'
CONFIG=json.loads((Path.home()/'.config/aster/judge_api.json').read_text())
MODEL=CONFIG['model']
LOCK=threading.Lock()
def read(p):
 try:return json.loads(p.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return None
def write(p,d):
 p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2));t.replace(p)
def request(prompt,limit):
 body={'model':MODEL,'input':[{'role':'user','content':prompt}],'max_output_tokens':limit}
 req=urllib.request.Request(CONFIG['base_url']+'/responses',data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+CONFIG['api_key'],'Content-Type':'application/json'})
 for attempt in range(5):
  try:
   with urllib.request.urlopen(req,timeout=180) as response:d=json.load(response)
   text='\n'.join(c['text'] for o in d.get('output',[]) for c in o.get('content',[]) if c.get('type')=='output_text')
   if d.get('status')=='incomplete' or not text:raise ValueError('Incomplete API response')
   return text,d
  except (urllib.error.URLError,TimeoutError,ValueError) as exc:
   if attempt==4:raise RuntimeError(type(exc).__name__+': API request failed')
   time.sleep(min(60,3*2**attempt))
def score(case):
 cid=case['case_id'];folder=OUT/'cases'/cid.replace(':','_')
 unit=folder/'result.json';prior=read(unit)
 if prior and prior.get('status')=='complete':return prior
 start=time.time()
 try:
  refs=case['reference_steps'];preds=case['candidate_steps']
  path_prompt=official.pcc.build_prompt(case['language'],refs,preds)
  outcome_prompt=official.acc.render_template(official.acc.TEMPLATE_ZH if case['language']=='zh' else official.acc.TEMPLATE_EN,{'Reference_Conclusion':case['reference_conclusion'],'Candidate_Conclusion':case['candidate_conclusion']})
  values={}
  for stage,prompt in [('path',path_prompt),('outcome',outcome_prompt)]:
   values[stage]=[]
   for i in range(5):
    p=folder/f'{stage}_{i+1}.json';saved=read(p)
    if saved and saved.get('status')=='complete':
     if saved['model']!=MODEL or saved['prompt']!=prompt:raise ValueError('Existing scoring protocol differs')
     values[stage].append(saved['score']);continue
    for retry in range(3):
     text,response=request(prompt,max(8192,14*len(refs)*len(preds)+2048) if stage=='path' else 4096)
     try:
      value=official.pcc.parse_pairwise_score_matrix(text,refs,preds) if stage=='path' else official.acc.parse_accuracy_score(text)
      break
     except Exception:
      write(folder/f'{stage}_{i+1}_invalid_{retry}.json',{'text':text,'response_id':response.get('id')})
      if retry==2:raise ValueError('Judge output could not be parsed')
    write(p,{'status':'complete','model':MODEL,'returned_model':response.get('model'),'prompt':prompt,'raw_output':text,'score':value,'usage':response.get('usage'),'response_id':response.get('id')})
    values[stage].append(value)
  import numpy as np
  matrix=np.asarray(values['path'],dtype=float).mean(axis=0)
  _,dtw,_=official.dynamic_time_warping((1-(matrix-1)/4).tolist())
  result={'case_id':cid,'status':'complete','model':MODEL,'draws_per_metric':5,'DTW_PCC':dtw,'Acc':sum((v-1)/4 for v in values['outcome'])/5,'seconds':time.time()-start}
 except Exception as exc:
  result={'case_id':cid,'status':'failed','error_type':type(exc).__name__,'error':str(exc).replace(CONFIG['api_key'],'[redacted]')[:250],'seconds':time.time()-start}
 write(unit,result)
 return result
def aggregate(cases):
 rows=[read(OUT/'cases'/c['case_id'].replace(':','_')/'result.json') for c in cases]
 valid=[r for r in rows if r and r['status']=='complete']
 done=len(valid)==len(cases)==7458
 d={'status':'complete' if done else 'incomplete','registered':len(cases),'scored':len(valid),'failed':sum(bool(r and r['status']=='failed') for r in rows),'judge':MODEL,'draws_per_metric':5,'metrics':{k:sum(r[k] for r in valid)/len(valid) if done else None for k in ['DTW_PCC','Acc']}}
 write(OUT/'summary.json',d)
 if done:
  paper=Path('/home/gcx/TRIAGE-worktrees/aster-bottleneck-revision-20261031/docs/ASTER/manuscript.zh.md')
  text=paper.read_text();start='<!-- ASTER_FINAL_MAIN_START -->';end='<!-- ASTER_FINAL_MAIN_END -->'
  before,rest=text.split(start,1);table,after=rest.split(end,1)
  lines=table.split('\n')
  for i,line in enumerate(lines):
   if line.startswith('| **处置盲内部干预（本文）** |'):
    cells=line.split('|');cells[3]=' **%.2f** '%(100*d['metrics']['DTW_PCC']);cells[4]=' **%.2f** '%(100*d['metrics']['Acc']);lines[i]='|'.join(cells)
  text=before+start+'\n'.join(lines)+end+after
  text=text.replace('**本文：结构化指南／Qwen裁判**','**本文：结构化指南／gpt-6.1-sol裁判**')
  text=text.replace('语义评分裁判为Qwen3-8B','主方法语义评分裁判为gpt-6.1-sol（第三方接口返回名称）')
  text=text.replace('以Qwen3-8B为裁判，每个输入采样五次','主方法以gpt-6.1-sol为裁判，每个输入独立采样五次')
  paper.write_text(text)
 values=['—' if d['metrics'][k] is None else '%.2f'%(100*d['metrics'][k]) for k in ['DTW_PCC','Acc']]
 (OUT/'REPORT.zh.md').write_text('# 主方法API裁判结果\n\n裁判：'+MODEL+'；完成 '+str(len(valid))+'/7458，每项五次独立请求。\n\n| DTW-PCC↓ | Acc↑ |\n|---:|---:|\n| '+' | '.join(values)+' |\n')
 return d
def main():
 p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');p.add_argument('--workers',type=int,default=8);a=p.parse_args()
 OUT.mkdir(exist_ok=True)
 import fcntl
 lock=(OUT/'runner.lock').open('w')
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 cases=[json.loads(line) for line in (BASE/'cases.jsonl').read_text().splitlines()]
 chosen=cases[1:3] if a.pilot else cases
 start=time.time()
 write(OUT/'launch.json',{'status':'running','pid':os.getpid(),'pilot':a.pilot,'workers':a.workers,'cases':len(chosen),'model':MODEL,'metrics':['DTW_PCC','Acc'],'draws_per_metric':5,'started_at':start})
 results=[]
 with cf.ThreadPoolExecutor(max_workers=a.workers) as pool:
  for result in pool.map(score,chosen):
   results.append(result)
   if len(results)%50==0:aggregate(cases)
 if not a.pilot:
  for retry_round in range(2):
   failed_ids={r['case_id'] for r in results if r['status']!='complete'}
   if not failed_ids:break
   retry_cases=[c for c in chosen if c['case_id'] in failed_ids]
   with cf.ThreadPoolExecutor(max_workers=a.workers) as pool:retried=list(pool.map(score,retry_cases))
   updated={r['case_id']:r for r in retried}
   results=[updated.get(r['case_id'],r) for r in results]
 summary=aggregate(cases)
 if a.pilot:write(OUT/'pilot.json',{'elapsed':time.time()-start,'results':results})
 write(OUT/'launch.json',{'status':'complete' if all(r['status']=='complete' for r in results) else 'incomplete','pid':os.getpid(),'pilot':a.pilot,'workers':a.workers,'model':MODEL,'elapsed':time.time()-start,'summary':summary})
 print(json.dumps({'pilot':a.pilot,'elapsed':time.time()-start,'completed':sum(r['status']=='complete' for r in results),'failed':sum(r['status']!='complete' for r in results)}),flush=True)
if __name__=='__main__':main()
