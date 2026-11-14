"""Export only numeric predictions/metadata, tree-paired CIs, plots and honest completion status."""
import argparse,collections,csv,gzip,hashlib,json,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
CORE=('NC','AC','IC','SW')
def read(path):
 rows=[]
 if path.exists():
  for line in path.read_text().splitlines():
   try:rows.append(json.loads(line))
   except json.JSONDecodeError:break # A running writer may have an incomplete final line.
 return rows

def paired(a,b):
 b={r['key']:r for r in b};tree=collections.defaultdict(lambda:[0,0])
 for r in a:
  if r['key'] in b:
   tree[r['tree']][0]+=int(r['correct'])-int(b[r['key']]['correct']);tree[r['tree']][1]+=1
 if not tree:return {'n':0}
 values=np.array(list(tree.values()),float);rng=np.random.default_rng(20261024)
 sums=values[rng.integers(0,len(values),(10000,len(values)))].sum(1);delta=sums[:,0]/sums[:,1]
 return {'n':int(values[:,1].sum()),'trees':len(values),'difference':float(values[:,0].sum()/values[:,1].sum()),'ci95':np.quantile(delta,[.025,.975]).tolist(),'bootstrap_replicates':10000,'unit':'tree; all variants/seeds of sampled trees stay together'}
def writecsv(path,rows):
 if not rows:return
 with path.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 meta={};tables={};summaries=[];cis=[];failures=[];e3=[]
 reference={f"{c['id']}|{c['seed']}|{c['variant']}":c for c in read(a.root.parent/'cases.jsonl')}
 whitelist=('key','id','tree','variant','seed','split','prediction','gold_option','correct','oracle','scores','seconds','reused','p')
 for model in ('qwen','llama','evidence'):
  d=a.root/model;meta[model]={}
  for name in ('status','locked','manifest','FAILED'):
   p=d/(name+'.json')
   if p.exists():meta[model][name]=json.loads(p.read_text())
  for path in sorted(d.glob('*/*.jsonl')):
   stage=path.parent.name;method=path.stem;rows=read(path);tables[model,stage,method]=rows
   export=a.out/'records'/model/stage/(method+'.jsonl.gz');export.parent.mkdir(parents=True,exist_ok=True)
   with export.open('wb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as f:
    for r in rows:f.write((json.dumps({k:r[k] for k in whitelist if k in r})+'\n').encode())
   if stage=='branches':
    valid=[r for r in rows if r['key'] in reference and r['split']=='dev']
    # All E3 comparisons use precisely the 50-case shared subset.
    subset={r['key'] for r in read(d/'branches'/'patient_first.jsonl')}
    valid=[r for r in valid if r['key'] in subset]
    if valid:e3.append({'model':model,'method':method,'rows':len(valid),'branch_accuracy':sum(max(range(len(r['p'])),key=lambda i:r['p'][i])==reference[r['key']]['gold_rule'] for r in valid)/len(valid)})
    continue
   for variant in (*CORE,'CS','CS-null','core'):
    group=[r for r in rows if r['variant'] in CORE] if variant=='core' else [r for r in rows if r['variant']==variant]
    if not group:continue
    seeds=collections.defaultdict(list)
    for r in group:seeds[r['seed']].append(int(r['correct']))
    means=[np.mean(x) for x in seeds.values()]
    summaries.append({'model':model,'stage':stage,'method':method,'variant':variant,'rows':len(group),'cases':len({r['id'] for r in group}),'seeds':len(seeds),'accuracy':float(np.mean([r['correct'] for r in group])),'seed_mean':float(np.mean(means)),'seed_std':float(np.std(means,ddof=1)) if len(means)>1 else None,'oracle':bool(group[0]['oracle'])})
 for (model,stage,method),rows in tables.items():
  if stage not in ('E2','E5','E7'):continue
  for baseline in (('none','two_stage_crop','direct_map') if stage=='E5' else ('none',)):
   if method==baseline:continue
   base=tables.get((model,stage,baseline),[])
   for variant in (*CORE,'CS','CS-null','core'):
    filt=lambda data:[r for r in data if r['variant'] in CORE] if variant=='core' else [r for r in data if r['variant']==variant]
    result=paired(filt(rows),filt(base))
    if result['n']:cis.append(dict(model=model,stage=stage,method=method,baseline=baseline,variant=variant,**result))
 for model in ('qwen','llama'):
  baseline={r['key']:r for r in tables.get((model,'E5','none'),[])}
  for method in ('ASTER_self','ASTER_full','two_stage_crop','direct_map'):
   for r in tables.get((model,'E5',method),[]):
    if r['key'] not in baseline:continue
    b=baseline[r['key']];kind='regression' if b['correct'] and not r['correct'] else 'recovery' if r['correct'] and not b['correct'] else 'shared_error' if not r['correct'] else None
    if kind:failures.append({k:r[k] for k in ('id','tree','variant','seed','prediction','gold_option')}|{'model':model,'method':method,'kind':kind})
 # E8 comparisons use exactly the common completed test subset.
 for (model,stage,method),rows in tables.items():
  if stage=='E8':
   base=tables.get((model,'E5','ASTER_self'),[]);result=paired(rows,base)
   if result['n']:cis.append(dict(model=model,stage=stage,method=method,baseline='ASTER_self',variant='core',**result))
  if stage=='E6' and method.startswith(('gate_','direct_')):
   base=tables.get((model,'E6','crop_'+method.split('_',1)[1]),[]);result=paired(rows,base)
   if result['n']:cis.append(dict(model=model,stage=stage,method=method,baseline='matched_crop',variant='core',**result))
 costs=[]
 for (model,stage,method),rows in tables.items():
  times=[r['seconds'] for r in rows if not r.get('reused',False)]
  if times:costs.append({'model':model,'stage':stage,'method':method,'measured_calls':len(times),'total_seconds':sum(times),'mean_seconds':float(np.mean(times)),'p95_seconds':float(np.quantile(times,.95))})
 writecsv(a.out/'costs.csv',costs)
 fig,axes=plt.subplots(2,2,figsize=(11,8))
 for row,model in enumerate(('qwen','llama')):
  for col,kind in enumerate(('noise','coverage')):
   ax=axes[row,col]
   for method in ('crop','gate','direct'):
    vals=[r for r in summaries if r['model']==model and r['stage']=='E6' and r['variant']=='core' and r['method'].startswith(method+'_'+kind+'_')]
    vals.sort(key=lambda r:float(r['method'].rsplit('_',1)[1]))
    if vals:ax.plot([float(r['method'].rsplit('_',1)[1]) for r in vals],[r['accuracy'] for r in vals],marker='o',label=method)
   ax.set_title(model+' '+kind);ax.set_ylim(0,1);ax.set_xlabel(kind);ax.set_ylabel('Accuracy')
   if ax.lines:ax.legend()
 fig.tight_layout();fig.savefig(a.out/'E6.png',dpi=150);plt.close(fig)
 writecsv(a.out/'accuracy.csv',summaries);writecsv(a.out/'E3.csv',e3);writecsv(a.out/'failure_metadata.csv',failures)
 (a.out/'paired_intervals.json').write_text(json.dumps(cis,indent=2));(a.out/'status.json').write_text(json.dumps(meta,indent=2))
 for stage in ('E2','E5','E7'):
  fig,axs=plt.subplots(1,2 if stage!='E7' else 1,figsize=(14,6),squeeze=False)
  for ax,model in zip(axs[0],('qwen','llama') if stage!='E7' else ('evidence',)):
   vals=[r for r in summaries if r['stage']==stage and r['model']==model and r['variant']=='core']
   ax.barh([r['method'] for r in vals],[r['accuracy'] for r in vals]);ax.set_xlim(0,1);ax.set_xlabel('Accuracy');ax.set_title(f'{model}: {stage} (observed rows only)')
  fig.tight_layout();fig.savefig(a.out/(stage+'.png'),dpi=150);plt.close(fig)
 lines=['# ASTER 后续实验实际结果','',f'导出时间：{time.strftime("%Y-%m-%d %H:%M:%S UTC",time.gmtime())}。只报告已写出的预测；运行中的结果属于中间结果。','','## 执行状态','']
 for model,obj in meta.items():
  s=obj.get('status',{});lines.append(f"- {model}: {s.get('stage','尚无进度')}，{s.get('completed_rows',0)}/{s.get('total_rows','?')}；峰值已分配显存 {s.get('peak_allocated_GiB',0):.2f} GiB，峰值保留显存 {s.get('peak_reserved_GiB',0):.2f} GiB；本进程 GPU 占用 {s.get('gpu_hours_this_process',0):.3f} 小时。")
  if 'FAILED' in obj:lines.append(f"  错误：{obj['FAILED']['error']}")
 lines+=['','GPU 小时按每个工作进程占用一张卡的墙钟时间计算，包括等待辅助预测；逐例 seconds 是复用后的增量耗时，不代表每种方法独立运行成本。断点重启时 status 的时间只计当前进程。模型全程复用加载，干预之间不复用 KV。','','## 实际准确率（NC/AC/IC/SW 合并）','','|模型|阶段|方法|已完成行数|准确率|种子数|','|---|---|---|---:|---:|---:|']
 for r in summaries:
  if r['variant']=='core' and r['stage'] in ('E2','E5','E7'):lines.append(f"|{r['model']}|{r['stage']}|{r['method']}|{r['rows']}|{r['accuracy']:.3%}|{r['seeds']}|")
 lines+=['','## 解释与复现','','- E0 部分完成、E1 两模型负结果保留在上一级目录，不改写为通过。旧 20 例 T2/T3 性能门槛不再控制后续运行。','- E2 使用金标，只比较通道。默认不写入金标规则前缀；gold_prompt 是显式提示对照。stage_hard_blind_options 专门保留盲判和选项共同干预的对照。','- E3 用开发集固定 50 例的四卷比较原顺序、病人优先、隐藏处置/选项、重复判断指令三次及盲判。重复指令不是重复采样投票。','- 每个主模型使用全部 200 例开发集四卷 seed=0，选择 self/full 的 λ=1/2/4、全部层或后半层，以及置信裁剪阈值 0.5/0.7/0.9；locked.json 在测试前落盘。','- ASTER-self 使用本模型盲判分布。ASTER-full 使用本模型盲判与 Qwen3-1.7B 隐藏处置/选项的分布等权平均，再进行仅指南处置的软门控。两者的回答阶段默认不再盲判、不干预选项。','- E5 预测函数只收到 patient/rules/options；金标仅由评估和明确标记 oracle 的上限组读取。explicit_branch 写出本模型普通分支判断；two_stage_crop/confidence_crop/direct_map 使用盲判分布。','- E5 全量为每模型 1000 例 × 四卷 × 三种子，加 CS/CS-null 各48例 × 三种子；8种方法。CS 属于测试集，不用于开发选参。','- E6 固定100个测试病例四卷 seed=0；向金标分支分布注入0/10/20/40%错误，或保留100/75/50%病例证据。缺证据时回退无干预。属于金标控制诊断，不能称为真实证据员错误率。','- E7 是 Qwen3-1.7B 的完整1000例 seed=0测试和48例 CS 补充，仅做额外模型泛化，不宣称新任务泛化。','- E8 在固定100个测试病例四卷上比较普通判断、硬门控、选项干预和回答阶段盲判；失败文件只含编号、类别及预测，不含临床原文。E4 仅采用本轮工程检查，不声称完成机制定位。','- 配对区间按树重采样10000次，保留同树的病例、卷子和种子；三个种子的标准差不是独立病例置信区间。中间结果只对共同完成的行配对。','- 所有完整输入和日志留在服务器 private。公开 records 仅白名单元数据和数值预测。','','复现：在115服务器激活 /home/gcx/gcx261014 后，在对应工作目录执行 `bash experiments/aster_e0_e8_20261024/launch_continuation.sh`。运行锁避免重复启动；相同配置跳过已完成记录。报告命令：`python experiments/aster_e0_e8_20261024/continuation_report.py --root /home/gcx/aster_e0_e8_20261024/private/continuation_v2 --out experiments/aster_e0_e8_20261024/results/continuation_v2`。']
 (a.out/'报告.zh.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps({'rows_exported':sum(len(v) for v in tables.values()),'summary_rows':len(summaries),'models':{k:v.get('status',{}).get('stage') for k,v in meta.items()}}))
if __name__=='__main__':main()
