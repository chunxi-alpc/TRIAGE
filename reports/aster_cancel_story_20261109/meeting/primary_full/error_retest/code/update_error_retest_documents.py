
"""Publish conditional repair rates on the old-linear error pool; no scoring or git."""
from pathlib import Path
import argparse,json,re,shutil

BASE=Path('/home/gcx/aster_next_20261109')
PUB=Path('/home/gcx/aster-real-completion-publish-20261109')
STAND=Path('/home/gcx/aster_cancel_story_20261109')
REL=Path('reports/aster_cancel_story_20261109')
LABELS={'raw':'原模型','old_linear':'原16维线性方法','fit_bias_only':'仅fit校准',
 'optimized_balanced_fit_calibration':'优化平衡MLP',
 'optimized_strict_fit_calibration':'较强保护MLP',
 'trace_only_balanced_fit_calibration':'仅痕迹MLP'}
ORDER=['old_linear','raw','fit_bias_only','optimized_strict_fit_calibration','optimized_balanced_fit_calibration','trace_only_balanced_fit_calibration']
def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def label(name):return LABELS.get(name,name)
def num(value):return '—' if value is None else f'{int(value):,}'
def percent(value):return '—' if value is None else f'{float(value)*100:.2f}'
def table(headers,rows):
 return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']+['---:']*(len(headers)-1))+'|']+
  ['| '+' | '.join(str(v).replace('|','／').replace('\n',' ') for v in row)+' |' for row in rows])+'\n'
def ordered(methods):return sorted(methods,key=lambda n:(ORDER.index(n) if n in ORDER else len(ORDER),n))
def atomic(path,text):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
 tmp.write_text(text,encoding='utf-8');tmp.replace(path)
def insert(text,tag,body,anchor):
 block='<!-- '+tag+':start -->\n'+body.rstrip()+'\n<!-- '+tag+':end -->'
 pattern=re.compile('<!-- '+re.escape(tag)+':start -->.*?<!-- '+re.escape(tag)+':end -->',re.S)
 if pattern.search(text):return pattern.sub(lambda m:block,text)
 match=re.search(anchor,text,re.M)
 if not match:raise ValueError('Missing document anchor: '+anchor)
 return text[:match.start()]+block+'\n\n'+text[match.start():]
def dataset_title(name,stage):
 if 'mega_official' in name.lower():return 'MEGA-CDP 官方test'
 if 'nli' in name.lower():return 'NLI4CT-2024 全部原始split'
 if stage.get('dataset'):return stage['dataset']
 if 'mega' in name.lower():return 'MEGA-CDP'
 return name
def all_methods(stage):
 group=stage.get('groups',{}).get('all',{})
 return group.get('methods',stage.get('methods',{}))
def group_table(group):
 rows=[]
 for name,item in ((name,group.get('methods',{})[name]) for name in ordered(group.get('methods',{}))):
  rows.append([label(name),num(item.get('expected',group.get('pool_size'))),num(item.get('evaluated')),
   num(item.get('repaired')),num(item.get('still_wrong')),num(item.get('missing')),percent(item.get('repair_rate'))])
 return table(['方法','原错例池','已复测','修复','仍错','缺项','条件修复率%'],rows)


def error_pair_section(data):
 if not data:return ''
 rows=[]
 method='optimized_balanced_fit_calibration'
 keys=['MEGA_previous_controlled:A','MEGA_previous_controlled:CF',
       'NLI4CT-2024:test:all:preserving','NLI4CT-2024:test:all:changing']
 for key in keys:
  group=data.get('groups',{}).get(key,{})
  item=group.get('methods',{}).get(method)
  if not item:continue
  er=item.get('mean_evidence_retention',item.get('mean_signed_response_retention'))
  context=group.get('full_population_changes',{}).get(method,{})
  rows.append([key,num(group.get('error_pair_count')),num(item.get('paired_correct'))+'/'+num(item.get('n')),
   percent(item.get('variance_reduction')),percent(er),
   num(context.get('new_errors_on_originally_correct'))+'/'+num(context.get('originally_correct_pairs'))])
 if not rows:return ''
 return ('**原错误配对上的行为指标。** 配对池只按原16维方法先前“两端未全对”选择，'
  '不按优化后结果选样本。下表列优化平衡MLP的核心条件指标；'
  '旧MEGA为既用留出配对，NLI维持原生标签关系。\n\n'+
  table(['原错配对组','原错对','修复／已复测','VR%','ER%','新增错／原正确对背景'],rows)+'\n'
  '旧MEGA的P、F0均无原错误对，写为“无旧错对”，不补填VR为0。'
  '这些是原错误条件下的点估计，未新增置信区间。五种方法、所有分项与原配对总体背景见结果包。\n\n')

def pair_failure_facts(data):
 if not data:return ''
 method='optimized_balanced_fit_calibration';parts=[]
 group=data.get('groups',{}).get('MEGA_previous_controlled:A',{})
 point=group.get('methods',{}).get(method)
 if point:
  parts.append('旧MEGA留出A原错对'+num(group.get('error_pair_count'))+'个，平衡MLP的VR为'+
   percent(point.get('variance_reduction'))+'%，修复'+num(point.get('paired_correct'))+'/'+num(point.get('n'))+'。')
 group=data.get('groups',{}).get('NLI4CT-2024:test:all:changing',{})
 point=group.get('methods',{}).get(method)
 if point:
  fmt=lambda x:'—' if x is None else f'{float(x):.4f}'
  text='NLI官方真实翻转原错对'+num(group.get('error_pair_count'))+'个，平均有向响应'+fmt(point.get('raw_signed_response_mean'))+'→'+fmt(point.get('signed_response_mean'))+'，ER为'+percent(point.get('mean_signed_response_retention'))+'%'
  path=BASE/'faithfulness/metrics.json'
  if path.exists():
   overall=load(path).get('nli',{}).get('groups',{}).get('test:all:changing',{}).get('methods',{}).get(method,{})
   if overall.get('mean_signed_response_retention') is not None:text+='（全体翻转对'+percent(overall['mean_signed_response_retention'])+'%）'
  text+='，表明错误子集平均响应保留较低；正确方向数'+num(point.get('raw_direction_positive'))+'→'+num(point.get('direction_positive'))+'，修复'+num(point.get('paired_correct'))+'/'+num(point.get('n'))+'。'
  parts.append(text)
 return ''.join(parts)


def section(data,pairs=None):
 text=('### 8.14 优化方法对原错误案例的复测\n\n'
  '本轮按最新安排停止继续全量扩测，仅对原16维线性方法（old_linear）判断错误的已有案例重放冻结优化模型。'
  '使用现有缓存和原参考标签，没有新增训练、主干前向或按错误池调参。'
  '这些数字是“原方法已经出错”这一条件下的修复率，不是新的全量准确率，也不是独立新增测试集。'
  '此前已完成的官方MEGA及NLI全量结果继续保留，不能用下表替换。\n\n'
  '条件修复率=修复案例数／有可用缓存并完成重放的原错例数；原错误池规模、已复测数与缺失数分别列出。'
  'MEGA按方法自己的动态路径重放，NLI按冻结二元判断重放。\n\n')
 for name,stage in data.get('stages',{}).items():
  title=dataset_title(name,stage);pool=stage.get('pool',{})
  text+='**'+title+'：原错误池。** 原池'+num(pool.get('expected'))+'条，有缓存'+num(pool.get('available'))+'条。\n\n'
  group=stage.get('groups',{}).get('all')
  if group is None:group={'pool_size':pool.get('expected'),'methods':stage.get('methods',{})}
  text+=group_table(group)+'\n'
  rows=[]
  for group_name,group in stage.get('groups',{}).items():
   if group_name=='all':continue
   for method in ordered(group.get('methods',{})):
    item=group['methods'][method]
    rows.append([group_name,num(group.get('pool_size')),label(method),num(item.get('evaluated')),num(item.get('repaired')),percent(item.get('repair_rate')),num(item.get('missing'))])
  if rows:text+=table(['原始分组','原错例池','方法','已复测','修复','条件修复率%','缺项'],rows)+'\n'
  background=stage.get('cached_background',{})
  if not background:
   background=data.get('cached_background',{}).get(name,{})
  bm=background.get('methods',background.get('groups',{}).get('all',{}).get('methods',{}))
  if bm:
   rows=[]
   for method in ordered(bm):
    item=bm[method]
    rows.append([label(method),num(item.get('original_correct_pool')),num(item.get('cached_available')),num(item.get('cached_new_errors'))])
   text+='**既有缓存内的新增错误背景。** 下表单列原方法正确的已有案例，作为判断修复代价的背景；不把它们并入原错例修复率的分母，也不据此生成新全量准确率。\n\n'
   text+=table(['方法','既有原正确池','有缓存','缓存内新增错误'],rows)+'\n'
 text+=error_pair_section(pairs)
 facts=pair_failure_facts(pairs)
 if facts:text+=facts+'\n\n'
 text+=('NLI按原始split保留分项；200条dev记录也出现在practice中，不将跨split记录视为独立新增样本。'
  '逐例结果保留原首次错误、新首次错误及缓存一致性记录，以区分整条路径被修复和错误位置改变。'
  '错误池来自原方法，比较不等于在该池上重新训练；本轮未自动恢复已停止的全量train_val或官方配对扩测。\n\n'
  '完整错误池覆盖、冻结模型及逐例修复／新增错误见 [error_retest](meeting/primary_full/error_retest/)。\n')
 return text

def brief(data,pairs=None):
 rows=[]
 for name,stage in data.get('stages',{}).items():
  group_key='all'
  if 'nli' in name.lower():
   group_key=next((key for key in stage.get('groups',{}) if key in
                  ('NLI4CT-2024:official_test','NLI4CT-2024:test')),None)
   if group_key is None:continue
  group=stage.get('groups',{}).get(group_key,{})
  methods=group.get('methods',{})
  background=stage.get('cached_background',{})
  bm=background.get('groups',{}).get(group_key,{}).get('methods',background.get('methods',{}))
  title='NLI4CT-2024 官方test' if 'nli' in name.lower() else dataset_title(name,stage)
  for method in ordered(methods):
   item=methods[method];context=bm.get(method,{})
   rows.append([title,label(method),num(item.get('repaired'))+'/'+num(item.get('evaluated')),
    percent(item.get('repair_rate')),
    num(context.get('cached_new_errors'))+'/'+num(context.get('cached_available'))])
 return ('### 原错误案例复测\n\n'
  '简表只列两个官方test的原错误池：MEGA 424条、NLI 1,469条。'
  '仅在原16维线性方法出错的已有案例中比较冻结优化方法的修复；这不是新的全量准确率。'
  '新增错误背景来自既有原正确案例缓存，分母独立列出。\n\n'+
  table(['数据集','方法','修复／已复测原错例','条件修复率%','新增错／原正确缓存'],rows)+
  '\n'+pair_failure_facts(pairs)+'\n\nNLI其他原split细项与2554条跨split错例记录见正文8.14节及配套结果；共享原ID不视为独立样本。'
  '本轮仅CPU缓存重放，不新增训练或主干前向，已停止的扩测不自动续跑。\n')


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--metrics',type=Path,default=BASE/'error_retest/metrics.json')
 p.add_argument('--pair-metrics',type=Path,default=BASE/'error_retest/pair_metrics.json')
 p.add_argument('--publish-worktree',type=Path,default=PUB)
 p.add_argument('--standalone-root',type=Path,default=STAND)
 p.add_argument('--dry-run',action='store_true')
 args=p.parse_args();data=load(args.metrics);pairs=load(args.pair_metrics) if args.pair_metrics.exists() else None
 if data.get('baseline','old_linear')!='old_linear':raise ValueError('Expected old_linear error-pool baseline')
 if not isinstance(data.get('stages'),dict):raise ValueError('Error-retest metrics require stages dictionary')
 report=args.publish_worktree/REL
 paper=report/'paper_story_latest.zh.md';meeting=report/'meeting/meeting_brief.zh.md'
 values={paper:insert(paper.read_text(encoding='utf-8'),'aster-error-retest-main',section(data,pairs),r'^## 9\.[^\n]*'),
  meeting:insert(meeting.read_text(encoding='utf-8'),'aster-error-retest-brief',brief(data,pairs),r'^## 4\.[^\n]*')}
 if args.dry_run:
  print(json.dumps({'dry_run':True,'targets':[str(p) for p in values],'baseline':data.get('baseline','old_linear')},ensure_ascii=False));return
 for path,text in values.items():
  atomic(path,text);atomic(args.standalone_root/path.relative_to(report),text)
 packet=report/'meeting/primary_full/error_retest';packet.mkdir(parents=True,exist_ok=True)
 for source in args.metrics.parent.iterdir():
  if source.is_file() and source.suffix in ('.json','.csv','.jsonl'):shutil.copy2(source,packet/source.name)
 code=packet/'code';code.mkdir(exist_ok=True)
 for name in ['update_error_retest_documents.py','retest_errors.py','error_retest.py','summarize_error_pairs.py']:
  source=BASE/'analysis'/name
  if source.exists():shutil.copy2(source,code/name)
 print(json.dumps({'state':'complete','updated':[str(p) for p in values],'git_actions':False},ensure_ascii=False))
if __name__=='__main__':main()

