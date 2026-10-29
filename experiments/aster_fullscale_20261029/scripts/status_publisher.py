"""Publish aggregate-only monitor snapshots; never launch or modify GPU tasks."""
import argparse,json,time,os,subprocess,fcntl
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone,timedelta
p=argparse.ArgumentParser(__doc__);p.add_argument('--run',required=True);p.add_argument('--once',action='store_true');p.add_argument('--interval',type=int,default=1800);a=p.parse_args()
root=Path(a.run);repo=Path(__file__).resolve().parents[3];public=repo/'experiments/aster_fullscale_20261029';lock=(root/'status_publisher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
PLAN={'confirm':5126,'official_test':7458,'med':852};METHODS=['baseline','learned','direct','mean','random','reverse','oracle','zero']
def read(path):
 try:return json.loads(path.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return {}
def write(path,value):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)
def alive(pid):
 try:return (Path('/proc')/str(pid)/'stat').read_text().split()[2]!='Z'
 except (OSError,IndexError):return False
def age(path):
 try:return round(time.time()-path.stat().st_mtime,1)
 except OSError:return None
def elapsed(pid):
 try:return float(Path('/proc/uptime').read_text().split()[0])-int((Path('/proc')/str(pid)/'stat').read_text().split()[21])/os.sysconf('SC_CLK_TCK')
 except Exception:return None

def snapshot():
 state=read(root/'serial_status.json');now=datetime.now(timezone.utc)
 data={'snapshot_version':1,'updated_at_utc':now.isoformat(),'updated_at_shanghai':now.astimezone(timezone(timedelta(hours=8))).isoformat(),'source':'server_aggregate_only','phase':state.get('phase'),'status':state.get('status'),'controller_alive':alive(state.get('controller_pid',0)),'controller_status_age_seconds':age(root/'serial_status.json'),'planned_cases':PLAN,'methods':METHODS,'judge_draws_per_parsed_distinct_output':{'path':5,'conclusion':5},'phases':{},'worker_progress':{},'gpu_allocation':{},'privacy':'No clinical text, patient identifiers, prompts or model responses are published.'}
 for phase,target in PLAN.items():
  cells={};duplicates=0;latest=None
  for path in (root/'serial'/phase).glob('worker_*/*.jsonl'):
   latest=max(latest or 0,path.stat().st_mtime)
   with path.open() as stream:
    for line in stream:
     try:r=json.loads(line)
     except json.JSONDecodeError:continue
     if r.get('method') not in METHODS:continue
     key=(r['case_id'],r['method'])
     if key in cells:duplicates+=1
     else:cells[key]=r
  grouped=defaultdict(set);failure=Counter();failure_cases=defaultdict(set);by_method=Counter();invalid=0;judge_failed=0
  for (case,method),r in cells.items():
   grouped[case].add(method);by_method[method]+=1;invalid+=bool(r.get('invalid_output'));judge_failed+=bool(r.get('official_judge_failed'))
   if r.get('technical_failure'):
    reason=str(r.get('failure_reason','')).lower();category='context_overflow' if 'context_overflow' in reason else 'cuda_oom' if 'out of memory' in reason else 'other_technical';failure[category]+=1;failure_cases[category].add(case)
  complete=sum(len(v)==len(METHODS) for v in grouped.values())
  data['phases'][phase]={'planned_cases':target,'complete_cases':complete,'partial_cases':len(grouped)-complete,'unique_prediction_records':len(cells),'planned_prediction_records':target*len(METHODS),'duplicate_cells':duplicates,'records_by_method':dict(by_method),'technical_failure_records':dict(failure),'technical_failure_cases':{k:len(v) for k,v in failure_cases.items()},'invalid_output_records':invalid,'judge_failed_records':judge_failed,'latest_output_age_seconds':round(time.time()-latest,1) if latest else None}
  allocation=read(root/'scheduler'/(phase+'_allocation.json'));data['gpu_allocation'][phase]={k:allocation[k] for k in ('cards','world') if k in allocation}
  for statusfile in (root/'serial'/phase).glob('worker_*/status.json'):
   r=read(statusfile);worker=int(statusfile.parent.name.split('_')[-1]);info=read(root/'scheduler'/(phase+'_'+str(worker)+'.json'));progress=read(next(statusfile.parent.glob('*_progress.json'),statusfile.parent/'not_present'))
   item={k:r[k] for k in ['status','batch_mode','generation_batch_size','judge_batch_size','world'] if k in r};item.update(alive=alive(info.get('pid',0)),gpu=info.get('gpu'),elapsed_seconds=elapsed(info.get('pid',0)))
   for k in ('completed_cases_this_invocation','written_records_this_invocation','wall_seconds'):
    if k in progress:item[k]=progress[k]
   transport=progress.get('batch_transport',{});item['oom_batch_splits']=transport.get('oom_splits',0);item['token_budget_splits']=transport.get('token_budget_splits',0);item['progress_age_seconds']=age(statusfile.parent/('med_heldout_progress.json' if phase=='med' else 'mega_'+phase+'_progress.json'))
   data['worker_progress'][phase+'_'+str(worker)]=item
 data['complete_cases_total']=sum(v['complete_cases'] for v in data['phases'].values());data['planned_cases_total']=sum(PLAN.values());data['unique_prediction_records_total']=sum(v['unique_prediction_records'] for v in data['phases'].values())
 active=[v for k,v in data['worker_progress'].items() if k.startswith(str(data['phase'])+'_') and v.get('alive')]
 rate=sum(v.get('completed_cases_this_invocation',0)/max(v.get('elapsed_seconds') or 0,v['wall_seconds']) for v in active if v.get('wall_seconds',0)>0)
 remaining_mega=sum(data['phases'][phase]['planned_cases']-data['phases'][phase]['complete_cases'] for phase in ('confirm','official_test'))
 if rate>0 and data['phase'] in ('confirm','official_test'):
  days=remaining_mega/rate/86400
  data['eta']={'basis':'sum of each live worker completed-case increments divided by current actual elapsed time including in-flight batches; future guideline mix may differ','mega_cases_per_hour':round(rate*3600,2),'remaining_mega_cases':remaining_mega,'remaining_mega_days_at_current_rate':round(days,2),'approximate_overall_days_range':[round(days*.7,1),round(days*1.5+.1,1)]}
 try:
  raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10)
  data['gpu_status']=[dict(zip(['index','memory_used_mib','utilization_percent'],[int(v.strip()) for v in line.split(',')])) for line in raw.splitlines()]
 except Exception:data['gpu_status']=[]
 repair=read(root/'oom_repair_summary.json');data['historical_transport_repair']={k:repair[k] for k in ['retry_cells','rerouted_cases','counts'] if k in repair}
 aggregate=read(root/'serial/summary.json');allowed={'n_cases','n_guideline_clusters','method_accuracy_micro','corrected','regressed','net_corrected_minus_regressed','official_pcc','official_accuracy','failure_rows','invalid_output_rows','coverage_rows','continuous_guideline_statistics','route_failures','judge_failures'}
 if aggregate:data['completed_stage_metrics']={key:{k:v for k,v in value.items() if k in allowed} for key,value in aggregate.items()}
 return data

def publish(data):
 write(public/'AUTO_STATUS.json',data)
 lines=['# 自动进度快照','', '更新时间：'+data['updated_at_shanghai'],'当前阶段：'+str(data['phase'])+'；状态：'+str(data['status']),'','|阶段|完整病例|病例分母|预测记录|','|---|---:|---:|---:|']
 for phase,v in data['phases'].items():lines.append(f'|{phase}|{v["complete_cases"]}|{v["planned_cases"]}|{v["unique_prediction_records"]}|')
 lines+=['','预计剩余时间：'+json.dumps(data.get('eta',{}),ensure_ascii=False),'','本文件只发布汇总进度，不包含病例、提示或模型原文。实验由服务器原调度器运行，本发布程序不启动或重启实验。']
 (public/'AUTO_STATUS.zh.md').write_text('\n'.join(lines)+'\n')
 files=['experiments/aster_fullscale_20261029/AUTO_STATUS.json','experiments/aster_fullscale_20261029/AUTO_STATUS.zh.md']
 for cmd in [['git','add','--',*files],['git','commit','-m','Update aggregate experiment monitoring snapshot','--',*files],['git','push','origin','codex/aster-fullscale-plan-20261029']]:
  result=subprocess.run(cmd,cwd=repo,capture_output=True,text=True,timeout=120)
  if result.returncode:raise RuntimeError((result.stderr or result.stdout)[-1200:])
 write(root/'status_publisher_status.json',{'status':'published','updated_at_utc':data['updated_at_utc'],'phase':data['phase'],'pid':os.getpid()})
 with (root/'monitor_snapshots.jsonl').open('a') as f:f.write(json.dumps(data,ensure_ascii=False)+'\n')
last_phase=None;last_time=0.
while True:
 state=read(root/'serial_status.json');phase=(state.get('phase'),state.get('status'))
 if a.once or phase!=last_phase or time.time()-last_time>=a.interval:
  try:
   data=snapshot();publish(data);print(json.dumps({'published':True,'phase':data['phase'],'complete_cases_total':data['complete_cases_total'],'eta':data.get('eta')},ensure_ascii=False),flush=True);last_phase=phase;last_time=time.time()
   if data['status']=='complete' and data['phase']=='complete':break
  except Exception as ex:
   write(root/'status_publisher_status.json',{'status':'publication_failed','reason':str(ex),'pid':os.getpid()});print('status publication failed: '+str(ex),flush=True)
   if a.once:raise
 if a.once:break
 time.sleep(60)
