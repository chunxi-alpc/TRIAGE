
"""Write existing behavioral-fidelity indicators into section 8.13; no git/model work."""
from pathlib import Path
import argparse,json,re,shutil
BASE=Path('/home/gcx/aster_next_20261109')
PUB=Path('/home/gcx/aster-real-completion-publish-20261109')
STAND=Path('/home/gcx/aster_cancel_story_20261109')
REL=Path('reports/aster_cancel_story_20261109')
LABELS={'raw':'原模型','linear_output':'原16维线性方法','old_linear':'原16维线性方法',
 'optimized_strict_fit_calibration':'较强保护MLP','optimized_balanced_fit_calibration':'优化平衡MLP',
 'trace_only_balanced_fit_calibration':'仅痕迹MLP'}
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def f(x,d=2,scale=1):return '—' if x is None else f'{float(x)*scale:.{d}f}'
def pc(x):return f(x,scale=100)
def n(x):return '—' if x is None else str(int(x))
def ci(x):return '—' if not x else '['+', '.join(pc(v) for v in x)+']'
def lab(x):return LABELS.get(x,x)
def table(headers,rows):
 return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']+['---:']*(len(headers)-1))+'|']+
  ['| '+' | '.join(str(x).replace('|','／').replace('\n',' ') for x in row)+' |' for row in rows])+'\n'
def marked(text,tag,body,anchor=None):
 block='<!-- '+tag+':start -->\n'+body.rstrip()+'\n<!-- '+tag+':end -->'
 pattern=re.compile('<!-- '+re.escape(tag)+':start -->.*?<!-- '+re.escape(tag)+':end -->',re.S)
 if pattern.search(text):return pattern.sub(lambda m:block,text)
 if anchor is None:return text.rstrip()+'\n\n'+block+'\n'
 if anchor is None:return text.rstrip()+'\n\n'+block+'\n'
 m=re.search(anchor,text,re.M)
 if not m:raise ValueError('Document anchor missing '+anchor)
 return text[:m.start()]+block+'\n\n'+text[m.start():]
def atomic(p,s):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(s,encoding='utf-8');tmp.replace(p)
def mega_table(methods,groups):
 rows=[]
 for kind in groups:
  for name,r in methods.items():
   m=r['groups'][kind]
   rows.append([kind,lab(name),n(m.get('n')),f(m.get('difference_rms'),4),pc(m.get('variance_reduction')),
    pc(m.get('prediction_r2')),pc(m.get('mean_evidence_retention')),f(m.get('correction_rms'),4),
    pc(m.get('correction_rms_over_ap_rms')),n(m.get('response_direction_degraded')) if kind in ('C','F','CF') else '—',
    pc(m.get('paired_accuracy'))])
 return table(['类型','方法','配对数','分数差RMS','VR%','R²%','ER%','修正RMS','修正／原A/P RMS%','方向退化','配对全对率%'],rows)
def official_section(official):
 text='**官方 MEGA-CDP 的新增配对范围。** '
 if official.get('status')!='available_partial_scope':
  plan=official.get('planned_collection',{});selected=plan.get('selected',{})
  pairs=selected.get('pairs',{});sources=selected.get('pair_sources',{})
  def amount(value,default):return f'{int(default if value is None else value):,}'
  return text+(
   '此项已进入固定队列，结果尚未汇总，不填写实测指标。范围为'+
   amount(selected.get('original_endpoints'),7458)+'个官方病例的首个gold原端点，对应'+
   amount(selected.get('independent_tree_nodes'),440)+'个树首节点；原端点均已命中现有缓存。'
   '计划A '+amount(pairs.get('A'),22330)+'对（'+amount(sources.get('A'),417)+'个来源），P '+
   amount(pairs.get('P'),123)+'对（'+amount(sources.get('P'),32)+'个来源），共'+
   amount(plan.get('expected_views'),29911)+'个视图，新增前向上界为'+
   amount(plan.get('new_forward_upper_bound'),22453)+'。'
   'C/F与F0尚无合格既有观测，因此官方完整忠实性评价尚未完成。'
   '已有全量路径正确率另报，不代替这些配对指标。\n\n')
 text+=('新增结果限每条官方参考路径的首个gold决策节点，按表内实际配对覆盖报告；'
  '它不是路径内全部节点或完整反事实路径的评价。\n\n')
 rows=[]
 for kind in ['A','P','AP']:
  g=official.get('groups',{}).get(kind,{})
  if not g.get('expected_pairs'):
   rows.append([kind,'全部方法','未构造','—','—','—','—','—']);continue
  for name,m in g.get('methods',{}).items():
   rows.append([kind,lab(name),n(m.get('n'))+'/'+n(g.get('expected_pairs')),f(m.get('difference_rms'),4),
    pc(m.get('variance_reduction')),pc(m.get('prediction_r2')),f(m.get('correction_rms'),4),pc(m.get('paired_accuracy'))])
 text+=table(['类型','方法','已评价／预期','分数差RMS','VR%','R²%','修正RMS','配对全对率%'],rows)+'\n'
 missing=official.get('missing_evidence_groups',[])
 if missing:text+='C/F证据响应与F0缺项仍明确保留：'+', '.join(missing)+'。不据A/P结果宣称完整证据保护。\n\n'
 return text
def nli_section(data):
 if data.get('status')!='complete':return '**NLI原生配对。** 结果尚未产生；保留缺项，不补填0。\n\n'
 text=('**NLI原生鲁棒性与响应。** NLI使用冻结MEGA修正函数进行零样本迁移。'
  '保持／改变依据两端真实Label是否相同，而不是按干预名称猜测。'
  '同标签不自动等于MEGA的相同证据条件，下面的中心化VR和分数差仅描述原生变体变化；'
  '真实翻转对另报沿标签方向的平均响应保留率。区间以同一基础声明及其派生变体成组抽样；'
  '不同组仍可能共享试验材料，不视为独立患者。\n\n')
 rows=[]
 for key in ['test:all:preserving','test:all:changing']:
  group=data.get('groups',{}).get(key)
  if not group:continue
  for name,m in group['methods'].items():
   rows.append([group['relation'],lab(name),n(m['n']),pc(m.get('variance_reduction')),ci(m.get('variance_reduction_ci')),
    f(m['difference_rms'],4),pc(m.get('mean_signed_response_retention')),
    n(m.get('raw_gold_direction_positive'))+'/'+n(m.get('gold_direction_positive')) if group['relation']=='changing' else '—',
    n(m.get('gold_direction_degraded')) if group['relation']=='changing' else '—',pc(m.get('paired_accuracy'))])
 text+=table(['原生关系','方法','配对数','中心VR%','VR来源区间%','差分RMS','翻转响应ER%','原／新正确方向数','新增方向退化','配对全对率%'],rows)+'\n'
 rows=[]
 for key,group in data.get('groups',{}).items():
  if group['split']!='test' or group['intervention']=='all':continue
  for name in ['raw','old_linear','optimized_balanced_fit_calibration']:
   if name not in group['methods']:continue
   m=group['methods'][name]
   rows.append([group['intervention'],group['relation'],lab(name),n(m['n']),f(m['difference_rms'],4),
    pc(m.get('variance_reduction')),pc(m.get('mean_signed_response_retention')),f(m['correction_rms'],4),
    pc(m['paired_accuracy'])])
 if rows:text+=table(['原生干预','实际标签关系','方法','配对数','差分RMS','中心VR%','翻转响应ER%','修正RMS','配对全对率%'],rows)+'\n'
 warnings=[]
 for key in ['test:all:preserving','test:all:changing']:
  group=data.get('groups',{}).get(key,{})
  for name,m in group.get('methods',{}).items():
   if name=='raw':continue
   if m.get('mean_and_variance_both_reduced'):
    warnings.append([group.get('relation'),lab(name),f(m['raw_difference_mean'],4)+' → '+f(m['difference_mean'],4),
     f(m['raw_difference_variance'],4)+' → '+f(m['difference_variance'],4),pc(m.get('mean_signed_response_retention')),pc(m['paired_accuracy'])])
 if warnings:
  text+='**同步收缩的实测现象。** 下列组的分数差均值绝对值与方差都下降；这一事实本身不能区分有选择的干扰降低与整体响应减弱。\n\n'
  text+=table(['原生关系','方法','差分均值原→新','差分方差原→新','翻转响应ER%','配对全对率%'],warnings)+'\n'
 text+=('ER接近100%表示相对于原响应的平均比例，不表示原模型方向本来正确；'
  '因此同时列原／新正确方向数、方向退化和全对率。'
  '全部五方法、practice分项及各干预的family区间见配套metrics.json和summary.csv。\n\n')
 return text
def section(data):
 mega=data['mega'];methods=mega['methods']
 text=('### 8.13 忠实性主指标、覆盖与局限\n\n'
  '行为忠实性为主要评价目标；准确率和路径正确率用于检验任务能力保留。VR须与证据响应、方向退化和全对率联合解释，不能单独按越高越好排序。\n\n'
  '本节按第3.3节的原定义补报干扰VR、配对预测R²、证据ER、证据修正RMS及其干扰单位、'
  '方向退化、配对全对率和F0判断不翻转率，各项分别报告。'
  '这些指标衡量指定输入改变后的输出依赖，属于行为层面的忠实性；'
  '它们不证明内部表示、隐含推理或显式思维链使用了正确证据。\n\n'
  '记原／修正后的配对分数差为d₀、d，VR=1−Var(d)/Var(d₀)，'
  'R²=1−mean(d²)/Var(d₀)。前者使用中心化方差，后者还计入残差均值，不能混用。'
  'C/F的ER按真实标签方向的平均响应求修正前后之比。证据修正为d−d₀，'
  '干扰单位使用原A/P分数差RMS。F0不翻转率只判断两端预测是否相同，与两端是否都正确分别报告。\n\n'
  '**MEGA既有留出配对。** 以下仍是3,100输入、806条路径对应的旧train_val留出集合，'
  '此前用于旧版及优化方案评价，不是新官方测试。全部五种方法沿用冻结权重与截距，'
  '本次仅复算已有结果，无新增拟合或主干前向。\n\n')
 cov=mega['coverage']
 text+=table(['类型','配对数','来源数','缺项'],[[k,n(v.get('n')),n(v.get('sources')),n(v.get('missing'))] for k,v in cov.items()])+'\n'
 text+=mega_table(methods,['A','P','AP','C','F','CF','F0'])+'\n'
 rows=[]
 for name,r in methods.items():
  a=r['groups']['AP'];c=r['groups']['CF'];z=r['groups']['F0']
  rows.append([lab(name),ci(a.get('variance_reduction_ci')),ci(c.get('mean_evidence_retention_ci')),
   pc(z.get('prediction_nonflip_rate')),ci(z.get('prediction_nonflip_rate_ci')),pc(z.get('paired_accuracy'))])
 text+=table(['方法','A/P VR区间%','C/F ER区间%','F0不翻转率%','F0不翻转区间%','F0全对率%'],rows)+'\n'
 text+=('旧留出P仅10对、F仅7对、F0仅6对，来源分别为5、2、2个。'
  '这些项的覆盖有限，F0六对全不翻转不等于已验证更广泛场景。MEGA区间沿用按指南来源成组的2,000次重采样，种子11。\n\n')
 text+=official_section(data.get('official_mega_pair_evaluation',{}))
 text+=nli_section(data['nli'])
 text+=('**方法能够与尚不能解决的问题。** 原16维线性实例只对135对fit C/F施加逐对硬预算，'
  '这一保证不自动推广到新输入。优化MLP采用证据均方惩罚和开发集修正RMS档，'
  '同时读取证据子空间、残差与原分数，不能称作原硬约束的等价实现，'
  '其收益也不能全部归于原16维干扰痕迹。\n\n'
  '现有分数修正不能补读痕迹中没有可辨识信息的证据，'
  '也不保证修复变量、时间点、单位或逻辑条件的高置信错误。'
  '开发集36条原模型错误路径中34条首次错误的原分数绝对值超过5，19条集中于同一来源PMA条件；'
  '输入语义核对与输出端干扰降低需要分别验证。fit A/P覆盖也不均，A有4,869对而P只有18对。'
  '减少干扰并不必然提高节点或路径正确率，截距校准与真实纠错应分别报告。\n\n'
  '可继续在fit/dev上检验三个方向：补充P及时间点、变量和单位相关的真实可核验配对；'
  '对MLP比较证据误差尾部及方向保护，并沿VR与证据改动的权衡选择配置；'
  '对语义错误另比较证据核对或二次读取模块，保留额外前向成本和信息范围。'
  '这些为后续优化方向，本轮未修改权重或使用官方测试重新调参。\n\n'
  '独立主指标汇总与复算代码见 [faithfulness](meeting/primary_full/faithfulness/)。\n')
 return text

def brief(data):
 rows=[]
 for name,r in data['mega']['methods'].items():
  ap=r['groups']['AP'];cf=r['groups']['CF'];f0=r['groups']['F0']
  rows.append([lab(name),pc(ap.get('variance_reduction')),pc(ap.get('prediction_r2')),pc(cf.get('mean_evidence_retention')),f(cf.get('correction_rms'),4),pc(f0.get('prediction_nonflip_rate'))])
 text=('### 忠实性主指标补报\n\n'
  '行为忠实性为主要评价目标；准确率和路径正确率检验任务能力保留。'
  'VR须与证据响应、方向和全对率联合解释。下表是既用MEGA留出配对，'
  'P/F/F0分别仅10/7/6对，不与新官方路径测试混同。\n\n'+
  table(['方法','A/P VR%','R²%','C/F ER%','证据修正RMS','F0不翻转率%'],rows)+'\n')
 nli=data.get('nli',{})
 if nli.get('status')=='complete':
  preserving=nli.get('groups',{}).get('test:all:preserving',{}).get('methods',{})
  changing=nli.get('groups',{}).get('test:all:changing',{}).get('methods',{})
  nr=[]
  for name in ['raw','old_linear','optimized_balanced_fit_calibration']:
   if name not in preserving or name not in changing:continue
   a=preserving[name];b=changing[name]
   nr.append([lab(name),f(a.get('difference_rms'),4),pc(b.get('mean_signed_response_retention')),
    pc(b.get('paired_accuracy')),n(b.get('gold_direction_positive'))+'/'+n(b.get('n'))])
  if nr:
   text+='**NLI官方test原生变体。** 同标签4,136对，真实翻转864对。\n\n'
   text+=table(['方法','同标签RMS','翻转ER%','翻转全对率%','翻转正确方向／总对数'],nr)+'\n'
 else:text+='NLI原生配对指标尚未汇总。\n\n'
 return text+('NLI按原生干预和真实标签关系报告鲁棒性与响应；'
  '均值和方差同步下降不能单独当作忠实性改善，需结合翻转响应、正确方向数及全对率。'
  '官方MEGA新增首节点A/P按可用结果更新，C/F及F0缺项继续保留。完整结果和局限见第8.13节。\n')


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--metrics',type=Path,default=BASE/'faithfulness/metrics.json')
 p.add_argument('--publish-worktree',type=Path,default=PUB)
 p.add_argument('--standalone-root',type=Path,default=STAND)
 p.add_argument('--dry-run',action='store_true')
 a=p.parse_args();data=load(a.metrics);report=a.publish_worktree/REL
 paper=report/'paper_story_latest.zh.md';meeting=report/'meeting/meeting_brief.zh.md'
 texts={paper:marked(paper.read_text(encoding='utf-8'),'aster-faithfulness-main',section(data),r'^## 9\.[^\n]*'),
  meeting:marked(meeting.read_text(encoding='utf-8'),'aster-faithfulness-brief',brief(data),r'^## 4\.[^\n]*'),
  report/'README.md':marked((report/'README.md').read_text(encoding='utf-8'),'aster-faithfulness-readme',
   '行为忠实性主指标、配对覆盖与方法局限见正文第8.13节及 [faithfulness](meeting/primary_full/faithfulness/)。既有MEGA配对与NLI原生变体分别报告；官方MEGA首节点A/P按实际覆盖更新，C/F和F0缺项明确保留。')}
 if a.dry_run:
  print(json.dumps(dict(dry_run=True,targets=[str(p) for p in texts],official_scope=data.get('official_mega_pair_evaluation',{}).get('status')),ensure_ascii=False));return
 for path,value in texts.items():
  atomic(path,value);atomic(a.standalone_root/path.relative_to(report),value)
 packet=report/'meeting/primary_full/faithfulness';packet.mkdir(parents=True,exist_ok=True)
 for path in a.metrics.parent.iterdir():
  if path.is_file() and path.suffix in ('.json','.csv'):shutil.copy2(path,packet/path.name)
 code=packet/'code';code.mkdir(exist_ok=True)
 for name in ['summarize_faithfulness.py','update_faithfulness_documents.py','summarize_mega_pairs.py','summarize_nli_pairs.py']:
  source=BASE/'analysis'/name
  if source.exists():shutil.copy2(source,code/name)
 shutil.copytree(packet,a.standalone_root/'meeting/primary_full/faithfulness',dirs_exist_ok=True)
 print(json.dumps(dict(state='complete',updated=[str(p) for p in texts],packet=str(packet),git_actions=False),ensure_ascii=False))
if __name__=='__main__':main()

