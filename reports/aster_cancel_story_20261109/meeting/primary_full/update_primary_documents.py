
"""Insert full frozen-method task results into existing documents.

Called by the primary supervisor after each stage. Missing stage summaries are
shown as pending. This script never runs models, trains, or calls git.
"""
from pathlib import Path
import argparse,json,re,shutil,subprocess,sys

DEFAULT_PUBLISH=Path('/home/gcx/aster-real-completion-publish-20261109')
DEFAULT_STANDALONE=Path('/home/gcx/aster_cancel_story_20261109')
DEFAULT_RESULTS=Path('/home/gcx/aster_next_20261109/primary_results')
REPORT=Path('reports/aster_cancel_story_20261109')
FILES=[Path('paper_story_latest.zh.md'),Path('meeting/meeting_brief.zh.md'),Path('README.md')]
PACKET=Path('meeting/primary_full')
LABELS={
 'raw':'原模型','old_linear':'原16维线性方法','fit_bias_only':'仅fit校准',
 'optimized_balanced_fit_calibration':'平衡配置（fit校准）',
 'optimized_strict_fit_calibration':'较强保护配置（fit校准）',
 'trace_only_balanced_fit_calibration':'仅痕迹平衡配置（fit校准）'}
GROUPS={
 'fit':'原fit来源','dev':'原dev来源','old_test':'原806条测试来源',
 'oldtest':'原806条测试来源','remaining_train_val':'其余train_val来源','extra':'其余train_val来源',
 'NLI4CT-2024:train':'原train声明（零样本迁移）',
 'NLI4CT-2024:dev':'原dev声明（零样本迁移）',
 'NLI4CT-2024:practice_test':'practice_test声明（零样本迁移）',
 'NLI4CT-2024:official_test':'官方test声明（零样本迁移）',
 'NLI4CT-2024:practice':'practice声明（零样本迁移）',
 'NLI4CT-2024:test':'官方test声明（零样本迁移）'}

def load(path,default=None):
 p=Path(path);return json.loads(p.read_text(encoding='utf-8')) if p.exists() else default
def count(x):return '—' if x is None else f'{int(x):,}'
def num(x):return '—' if x is None else f'{float(x)*100:.2f}'
def interval(x):
 if not x:return '—'
 if isinstance(x[0],list):x=x[0]
 return '['+', '.join(num(v) for v in x)+']'
def table(headers,rows):
 return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']+['---:']*(len(headers)-1))+'|']+
  ['| '+' | '.join(str(x).replace('|','／').replace('\n',' ') for x in row)+' |' for row in rows])+'\n'
def marked(tag,body):return '<!-- '+tag+':start -->\n'+body.rstrip()+'\n<!-- '+tag+':end -->'
def insert(text,tag,body,anchor=None):
 block=marked(tag,body);pattern=re.compile('<!-- '+re.escape(tag)+':start -->.*?<!-- '+re.escape(tag)+':end -->',re.S)
 if pattern.search(text):return pattern.sub(lambda m:block,text)
 if anchor:
  m=re.search(anchor,text,re.M)
  if not m:raise ValueError('Document anchor missing: '+anchor)
  return text[:m.start()]+block+'\n\n'+text[m.start():]
 return text.rstrip()+'\n\n'+block+'\n'
def metric_values(item):
 correct=item.get('path_correct',item.get('correct'))
 evaluated=item.get('evaluated',item.get('n'))
 accuracy=item.get('path_accuracy',item.get('accuracy'))
 return correct,evaluated,accuracy
def method_names(methods,plan):
 order=plan.get('methods',[])
 return sorted(set(methods)|set(order),key=lambda x:(order.index(x) if x in order else len(order),x))
def display(name):
 if name.startswith(('condition_only','condition_prompt')):return '当前条件提示（对照）'
 return LABELS.get(name,name)
def collect(plan,root):
 stages=[]
 for spec in plan.get('stages',[]):
  path=root/spec['name']/'summary/metrics.json';metrics=load(path)
  overall=(metrics or {}).get('groups',{}).get('all',{})
  ready=bool(overall.get('methods'))
  if not ready:state='待完成'
  elif any(m.get('missing',0) or m.get('failed',0) for m in overall['methods'].values()):state='已汇总，含不可用项'
  else:state='已汇总'
  stages.append(dict(spec=spec,path=path,metrics=metrics,overall=overall,ready=ready,state=state))
 return stages
def result_table(overall,plan,compact=False):
 rows=[]
 for name in method_names(overall.get('methods',{}),plan):
  item=overall.get('methods',{}).get(name)
  if item is None:
   rows.append([display(name),'缺少方法结果','—'] if compact else [display(name),'缺少方法结果','—','—','—','—','—'])
   continue
  correct,n,accuracy=metric_values(item);paired=item.get('paired_vs_raw',{})
  if compact:rows.append([display(name),count(correct)+'/'+count(n),num(accuracy)])
  else:
   change='—' if not paired else count(paired.get('fixed'))+'/'+count(paired.get('broken'))
   rows.append([display(name),count(correct)+'/'+count(n),num(accuracy),interval(item.get('source_bootstrap_95ci')),
    change,num(paired.get('accuracy_difference')),interval(paired.get('source_bootstrap_95ci'))])
 headers=['方法','正确／已评价','正确率%'] if compact else ['方法','正确／已评价','正确率%','来源95%区间（%）','纠正／新增错误','相对原模型差（百分点）','差值95%区间（百分点）']
 return table(headers,rows)
def relevant_groups(stage):
 groups=(stage['metrics'] or {}).get('groups',{})
 kind=stage['spec'].get('kind')
 if kind=='binary':
  selected=[g for g in groups if g.startswith('NLI4CT-2024:')]
 elif stage['spec']['name']=='mega_train_val':
  selected=[g for g in groups if g in ('fit','dev','old_test','oldtest','remaining_train_val','extra')]
 else:selected=[]
 return selected

def atomic_text(path,text):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temporary=path.with_suffix(path.suffix+'.tmp')
 temporary.write_text(text,encoding='utf-8');temporary.replace(path)

def nli_pair_summary(plan,stage,root,manifest_path,execute=False):
 output=root/stage['spec']['name']/'pair_summary'
 metrics=output/'metrics.json';error=output/'pair_analysis_error.json'
 try:
  if metrics.exists():return load(metrics)
  if not execute or error.exists():return None
  base=manifest_path.parent
  script=base/'analysis/summarize_nli_pairs.py'
  if not script.exists():return None
  cases=Path(stage['spec']['cases'])
  pairs=base/'inventory/nli4ct_native_pairs.jsonl'
  nodes=sorted((root/stage['spec']['name']).glob('shard*/nodes.jsonl'))
  if not pairs.exists() or not nodes:return None
  output.mkdir(parents=True,exist_ok=True)
  frozen=Path(plan.get('frozen_manifest',str(base/'frozen_methods.json')))
  command=[sys.executable,str(script),'--cases',str(cases),'--pairs',str(pairs),'--nodes',*[str(p) for p in nodes],'--manifest',str(frozen),'--out',str(output)]
  with (output/'postprocess.log').open('w') as log:
   subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
  if not metrics.exists():raise RuntimeError('NLI pair postprocessor did not produce metrics.json')
  return load(metrics)
 except Exception as exc:
  if execute:
   atomic_text(error,json.dumps({'state':'failed','error':str(exc),'effect':'Main task results remain available; optional pair analysis omitted.'},ensure_ascii=False,indent=2)+'\n')
  return None

def focus_methods(methods,plan):
 names=[name for name in ['raw','old_linear','optimized_balanced_fit_calibration'] if name in methods]
 return names or method_names(methods,plan)

def nli_pair_text(data,plan):
 if not data:return '原生配对后处理尚无可用汇总，主分类结果照常报告。\n\n'
 text='**NLI 原生变体分项。** 保持或改变关系按原生配对两端 Label 的同异确定，不根据干预名称推断标签翻转。配对后处理区间按基础声明及其派生变体分组，总体分类区间按试验编号组合分组，两种口径分别报告。\n\n'
 subsets=data.get('case_subsets',{})
 rows=[]
 for key in ['test:original','test:preserving','test:changing']:
  item=subsets.get(key)
  if not item:continue
  for name in focus_methods(item.get('methods',{}),plan):
   result=item['methods'][name]
   rows.append([key,count(item.get('expected')),display(name),count(result.get('correct'))+'/'+count(result.get('n')),num(result.get('accuracy'))])
 if rows:text+=table(['官方test子组','预期条数','方法','正确／已评价','正确率%'],rows)+'\n'
 rows=[]
 for key,item in data.get('groups',{}).items():
  if str(item.get('split','')).lower() not in ('test','official_test') and not key.startswith('test:'):continue
  for name in focus_methods(item.get('methods',{}),plan):
   result=item['methods'][name]
   rms=lambda x:'—' if x is None else f'{float(x):.4f}'
   rows.append([key,count(item.get('expected_pairs')),display(name),rms(result.get('difference_rms')),rms(result.get('raw_difference_rms')),rms(result.get('correction_rms')),num(result.get('paired_accuracy'))])
 if rows:text+=table(['原生干预／关系','预期配对','方法','差分RMS','原模型差分RMS','修正RMS','配对全对率%'],rows)+'\n'
 text+='所有方法的原生变体指标及可用配对分母见同阶段 pair_summary/metrics.json；未将这些原生关系重命名为 MEGA 的 A/P/C/F。\n\n'
 return text


def section(plan,stages):
 rows=[[s['spec']['dataset'],s['spec'].get('split','—'),'完整路径' if s['spec'].get('kind')=='path' else '二元声明',count(s['spec'].get('expected')),s['state']] for s in stages]
 text=('### 8.12 主方法的全量真实任务评价\n\n'
  '本轮优先评价已冻结的主方法和必要对照，执行顺序为 MEGA-CDP 官方 test、NLI4CT-2024、MEGA-CDP train_val。'
  '全部阶段沿用既有模型、修正函数与校准参数，没有新增训练或按这些扩展结果重新选型。'
  '这里的“全部任务”按论文的三类任务界定：MEGA-CDP、NLI4CT及三段论有效性；服务器其他QA、数值和多选数据不并入本轮。\n\n')
 text+=table(['数据集','原始划分','评价单位','计划条数','状态'],rows)+'\n'
 text+=('MEGA-CDP 的完整路径正确与 NLI 的二元分类正确分别计算，不混合成跨任务总体正确率。'
  'NLI 使用在 MEGA-CDP 上拟合的修正函数进行零样本迁移，并未在 NLI 的 train 或 dev 上重新拟合。'
  'NLI 的9,542条为跨原始划分记录，对应9,342个不同原始ID；200条dev也在practice中，按原始划分分项报告。'
  'NLI 的来源区间以试验编号组合为重采样单位；不同组合可能共享试验，其独立性未得到保证，也不代表独立患者样本。'
  'MEGA-CDP train_val 包含此前 fit、dev及原留出测试，分组结果用于描述，不称作新增独立测试。\n\n')
 for stage in stages:
  spec=stage['spec'];unit='完整路径' if spec.get('kind')=='path' else '二元声明'
  text+='**'+spec['dataset']+'：'+spec.get('split',spec['name'])+'。** '
  if not stage['ready']:
   text+='计划 '+count(spec.get('expected'))+' 条'+unit+'，待完成；尚无汇总结果，不填写正确数或正确率。\n\n'
   continue
  overall=stage['overall']
  text+='预期 '+count(overall.get('case_count',spec.get('expected')))+' 条，覆盖 '+count(overall.get('source_count'))+' 个来源分组。\n\n'
  text+=result_table(overall,plan)+'\n'
  issues=[]
  for name,item in overall['methods'].items():
   if item.get('missing',0) or item.get('failed',0):
    issues.append(display(name)+'缺项'+count(item.get('missing',0))+'、失败'+count(item.get('failed',0)))
  if issues:text+='覆盖说明：'+'；'.join(issues)+'。正确率分母为表内已评价条数。\n\n'
  subset_rows=[]
  for group_name in relevant_groups(stage):
   group=stage['metrics']['groups'][group_name]
   for name in method_names(group.get('methods',{}),plan):
    item=group.get('methods',{}).get(name)
    if item is None:
     subset_rows.append([GROUPS.get(group_name,group_name),count(group.get('case_count')),display(name),'缺少方法结果','—'])
    else:
     correct,n,accuracy=metric_values(item)
     subset_rows.append([GROUPS.get(group_name,group_name),count(group.get('case_count')),display(name),count(correct)+'/'+count(n),num(accuracy)])
  if subset_rows:text+=table(['分组','预期条数','方法','正确／已评价','正确率%'],subset_rows)+'\n'
  if spec.get('kind')=='binary':
   text+=nli_pair_text(stage.get('pair_metrics'),plan)
  text+='完整来源分组、失败及配对差值见 [primary_full/'+spec['name']+'/metrics.json](meeting/primary_full/'+spec['name']+'/metrics.json)。\n\n'
 if plan.get('missing_data'):
  text+='**尚无真实数据的任务。**\n\n'
  text+=table(['任务','状态'],[[item.get('dataset','未知任务'),item.get('reason','尚无可用真实数据')] for item in plan['missing_data']])+'\n'
  text+='三段论的既有模拟继续标为模拟，未列为完成的真实实验。\n\n'
 text+='各阶段状态和预计规模来自 primary_plan.json；尚未完成的阶段在后续汇总时更新本节。\n'
 return text
def short_text(plan,stages):
 rows=[]
 for stage in stages:
  spec=stage['spec']
  if not stage['ready']:rows.append([spec['dataset'],spec.get('split','—'),count(spec.get('expected')),'待完成']);continue
  parts=[]
  for method in ['raw','old_linear','optimized_balanced_fit_calibration']:
   item=stage['overall'].get('methods',{}).get(method)
   if item:
    correct,n,accuracy=metric_values(item)
    parts.append(display(method)+' '+count(correct)+'/'+count(n)+'（'+num(accuracy)+'%）')
   else:parts.append(display(method)+'：缺少方法结果')
  result='；'.join(parts)
  rows.append([spec['dataset'],spec.get('split','—'),count(spec.get('expected')),result])
 return ('### 主方法全量真实任务进度\n\n'+table(['数据集','原始划分','计划条数','主方法结果／状态'],rows)+
  '\nNLI为既有MEGA修正函数的零样本迁移：9,542条跨split记录、9,342个原始ID，200条dev也在practice中，按原split分别报告。'
  'NLI来源区间按试验编号组合，不代表独立患者；MEGA train_val的fit、dev、原测试和其余来源分项报告。'
  '三段论尚无真实数据，既有模拟不算完成的真实测试。完整结果见正文第8.12节。\n')
def packet_copy(plan_path,plan,root,report_root):
 dest=report_root/PACKET;dest.mkdir(parents=True,exist_ok=True)
 shutil.copy2(plan_path,dest/'primary_plan.json')
 shutil.copy2(Path(__file__),dest/'update_primary_documents.py')
 base=plan_path.parent
 for name in ['frozen_methods.json','inventory/all_primary_inventory.json','inventory/available_datasets.json','inventory/nli4ct_inventory.json']:
  p=base/name
  if p.exists():
   target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
 code=dest/'code';code.mkdir(exist_ok=True)
 for name in ['inference/run.py','inference/run_primary.py','inference/frozen.py','analysis/summarize_nli_pairs.py']:
  source=base/name
  if source.exists():shutil.copy2(source,code/source.name)
 for stage in plan.get('stages',[]):
  source=root/stage['name']/'summary'
  if not source.exists():continue
  target=dest/stage['name'];target.mkdir(exist_ok=True)
  for p in source.iterdir():
   if p.is_file() and p.suffix in ('.json','.csv'):shutil.copy2(p,target/p.name)
  pair_source=root/stage['name']/'pair_summary'
  if pair_source.exists():
   pair_target=target/'pair_summary';pair_target.mkdir(exist_ok=True)
   for p in pair_source.iterdir():
    if p.is_file() and p.suffix in ('.json','.csv'):shutil.copy2(p,pair_target/p.name)
def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--manifest',type=Path,required=True)
 parser.add_argument('--results-root',type=Path,default=DEFAULT_RESULTS)
 parser.add_argument('--publish-worktree',type=Path,default=DEFAULT_PUBLISH)
 parser.add_argument('--standalone-root',type=Path,default=DEFAULT_STANDALONE)
 parser.add_argument('--dry-run',action='store_true')
 args=parser.parse_args();plan=load(args.manifest)
 if not isinstance(plan,dict) or not isinstance(plan.get('stages'),list):raise ValueError('Manifest requires stages list')
 stages=collect(plan,args.results_root)
 for stage in stages:
  if stage['ready'] and stage['spec'].get('kind')=='binary':
   stage['pair_metrics']=nli_pair_summary(plan,stage,args.results_root,args.manifest,execute=not args.dry_run)
 report_root=args.publish_worktree/REPORT
 originals={p:(report_root/p).read_text(encoding='utf-8') for p in FILES}
 texts={
  FILES[0]:insert(originals[FILES[0]],'aster-primary-full-section',section(plan,stages),r'^## 9\.[^\n]*'),
  FILES[1]:insert(originals[FILES[1]],'aster-primary-full-brief',short_text(plan,stages),r'^## 4\.[^\n]*'),
  FILES[2]:insert(originals[FILES[2]],'aster-primary-full-readme','主方法全量真实任务的已完成结果与待完成阶段见正文第8.12节及 [primary_full](meeting/primary_full/)。NLI为零样本迁移，MEGA train_val保留既有fit/dev/原测试分组；缺少真实数据的三段论仅保留模拟。')}
 if args.dry_run:
  print(json.dumps(dict(dry_run=True,stages={s['spec']['name']:s['state'] for s in stages},targets=[str(report_root/p) for p in FILES]),ensure_ascii=False));return
 for rel,value in texts.items():
  atomic_text(report_root/rel,value)
  target=args.standalone_root/rel;atomic_text(target,value)
 packet_copy(args.manifest,plan,args.results_root,report_root)
 print(json.dumps(dict(state='complete',stages={s['spec']['name']:s['state'] for s in stages},git_actions=False),ensure_ascii=False))
if __name__=='__main__':main()

