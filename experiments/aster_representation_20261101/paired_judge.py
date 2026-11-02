"""Pair the frozen main and blind-readout predictions at the level of each judge prompt."""
import argparse, collections, concurrent.futures as cf, fcntl, json, os, time
from pathlib import Path
import numpy as np
from . import api_judge as judge, semantic as sem
from . import official
ROOT=judge.BASE/'paired_blind_readout'
INDEX=ROOT/'inputs.jsonl'
def prepare():
 ROOT.mkdir(exist_ok=True)
 if INDEX.exists():return
 cases={r['case_id']:r for r in official.all_cases()}
 main={r['case_id']:r for r in (json.loads(l) for l in (judge.BASE/'cases.jsonl').open())}
 records=[]
 for path in (official.OUT/'results').glob('*.jsonl'):
  for l in path.open():
   r=json.loads(l)
   if r['kind']!='blind_readout':continue
   assert r['status']=='complete'
   cid=r['case_id'];c=cases[cid];a=main[cid]
   text=sem.canonical(c,r['result']['text'])
   parsed=judge.official.parse_record(dict(id=cid,case_path_id=c['guideline_id'],path_index=c['path_id'],filename=c['filename'],language=c['language'],llm_output=text))
   steps=judge.official.pcc.get_candidate_steps(parsed);conclusion=judge.official.acc.candidate_conclusion(parsed)
   records.append(dict(case_id=cid,source=c['source_id'],main=a,blind=dict(a,candidate_steps=steps,candidate_conclusion=conclusion),same_path=steps==a['candidate_steps'],same_outcome=conclusion==a['candidate_conclusion']))
 assert len(records)==len({r['case_id'] for r in records})==7458
 temp=INDEX.with_suffix('.tmp');temp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records));temp.replace(INDEX)
 counts=collections.Counter((r['same_path'],r['same_outcome']) for r in records)
 judge.write(ROOT/'design.json',dict(n=len(records),same_both=counts[True,True],different_both=counts[False,False],only_path_differs=counts[False,True],only_outcome_differs=counts[True,False],additional_draws_before_refusals=5*sum(int(not a)+int(not b) for a,b in [(r['same_path'],r['same_outcome']) for r in records])))
def folder(root,cid):return root/'cases'/cid.replace(':','_')
def extract(root,stage):
 draws=[judge.read(root/f'{stage}_{i}.json') for i in range(1,6)]
 if not all(v and v.get('status')=='complete' and v['model']==judge.MODEL for v in draws):return None
 if stage=='outcome':return float(np.mean([(v['score']-1)/4 for v in draws]))
 m=np.mean([v['score'] for v in draws],axis=0)
 return float(judge.official.dynamic_time_warping((1-(m-1)/4).tolist())[1])
def pair(r):
 cid=r['case_id'];a=folder(judge.OUT,cid);b=folder(ROOT,cid)
 prior=judge.read(b/'pair.json')
 if prior and prior['status'] in ('complete','unscorable'):return prior
 main=judge.score(r['main'])
 shared={stage:a for stage in ('path','outcome') if r['same_'+stage]}
 blind=main if len(shared)==2 else judge.score(r['blind'],output_root=ROOT,shared=shared)
 scores={}
 for stage in ['path','outcome']:
  first=extract(a,stage);second=first if stage in shared else extract(b,stage)
  scores[stage]=[first,second]
 status='complete' if all(None not in v for v in scores.values()) else ('failed' if 'failed' in (main['status'],blind['status']) else 'unscorable')
 result=dict(case_id=cid,source=r['source'],same_path=r['same_path'],same_outcome=r['same_outcome'],scores=scores,status=status)
 judge.write(b/'pair.json',result);return result
def report(rows):
 result=dict(registered=7458,processed=len(rows),status_counts=dict(collections.Counter(r['status'] for r in rows)),judge=judge.MODEL,metrics={})
 for stage in ['path','outcome']:
  for subset in ['all','different_outputs','different_metric_input']:
   valid=[r for r in rows if None not in r['scores'][stage] and (subset=='all' or (subset=='different_outputs' and not (r['same_path'] and r['same_outcome'])) or (subset=='different_metric_input' and not r['same_'+stage]))]
   if not valid:continue
   x=np.asarray([r['scores'][stage] for r in valid]);diff=x[:,0]-x[:,1]
   stat=dict(n=len(x),main=float(x[:,0].mean()),blind=float(x[:,1].mean()),main_minus_blind=float(diff.mean()))
   if len(rows)==7458:
    sums=collections.defaultdict(list)
    for r,v in zip(valid,diff):sums[r['source']].append(v)
    arr=np.array([[sum(v),len(v)] for v in sums.values()]);rng=np.random.default_rng(20261102)
    boot=[(lambda y:y[0]/y[1])(arr[rng.integers(len(arr),size=len(arr))].sum(0)) for _ in range(2000)]
    stat['paired_source_ci95']=np.quantile(boot,[.025,.975]).tolist()
   result['metrics'][stage+'_'+subset]=stat
 judge.write(ROOT/'summary.json',result)
def write_final_report():
 summary=judge.read(ROOT/'summary.json')
 assert summary['processed']==7458
 rows=['### 5.6 主方法与盲表示分类器的配对语义评分','',
       '两种方法对同一病例的输出均与同一参考答案比较，使用 gpt-6.1-sol 裁判，每项五次独立评分。每项指标仅纳入两侧均有完整评分的病例。相同的裁判输入复用同一评分。','',
       '| 病例范围 | 指标 | 配对例数 | 主方法（%） | 盲表示分类器（%） | 配对差（百分点，95%来源簇区间） |',
       '|---|---|---:|---:|---:|---:|']
 for subset,label in [('all','全部病例中的有效配对'),('different_outputs','输出不同病例中的有效配对')]:
  for stage,metric in [('path','DTW-PCC↓'),('outcome','Acc↑')]:
   v=summary['metrics'].get(stage+'_'+subset)
   if v is None:continue
   lo,hi=v['paired_source_ci95']
   rows.append(f"| {label} | {metric} | {v['n']} | {100*v['main']:.2f} | {100*v['blind']:.2f} | {100*v['main_minus_blind']:+.2f}（{100*lo:+.2f}, {100*hi:+.2f}） |")
 rows+=['','配对差按主方法减分类器计算。拒评与未取得五次有效评分的输入保留为缺失。','']
 body='\n'.join(rows)
 (ROOT/'REPORT.zh.md').write_text(body)
 paper=Path('/home/gcx/TRIAGE-worktrees/aster-bottleneck-revision-20261031/docs/ASTER/manuscript.zh.md')
 text=paper.read_text()
 if '### 5.6 主方法与盲表示分类器的配对语义评分' not in text and '## 6 ' in text:
  paper.write_text(text.replace('## 6 ',body+'\n## 6 ',1))
def run():
 import sys
 lock=(judge.OUT/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 items=[json.loads(l) for l in INDEX.open()]
 def priority(r):
  same=r['same_path'] and r['same_outcome'];status=(folder(judge.OUT,r['case_id'])/'result.json').exists()
  return (same,not status)
 items.sort(key=priority)
 judge.write(ROOT/'launch.json',dict(pid=os.getpid(),status='running',workers=32,started=time.time()))
 with cf.ThreadPoolExecutor(32) as pool:
  futures=[pool.submit(pair,r) for r in items];rows=[]
  for f in cf.as_completed(futures):
   rows.append(f.result())
   if len(rows)%50==0:report(rows)
  for _ in range(2):
   missing={r['case_id'] for r in rows if r['status']=='failed'}
   if not missing:break
   updated={r['case_id']:r for r in pool.map(pair,[r for r in items if r['case_id'] in missing])}
   rows=[updated.get(r['case_id'],r) for r in rows]
 report(rows);judge.aggregate([r['main'] for r in items])
 write_final_report()
 judge.write(ROOT/'launch.json',dict(pid=os.getpid(),status='finished',workers=32,processed=len(rows)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');a=p.parse_args()
 prepare()
 if a.prepare:print((ROOT/'design.json').read_text())
 else:run()
