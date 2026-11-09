"""Update section 8.15 from frozen hard-error optimization results; no scoring or git."""
from pathlib import Path
import argparse,json,re,shutil,gzip

ROOT=Path('/home/gcx/aster_hard_optimization_20261109')
PUBLISH=Path('/home/gcx/aster-real-completion-publish-20261109')
STANDALONE=Path('/home/gcx/aster_cancel_story_20261109')
REPORT=Path('reports/aster_cancel_story_20261109')
NEW=['hard_tail_pair_protected_fit_calibration','hard_tail_robust_protected_fit_calibration']
BASE=['raw','old_linear','optimized_balanced_fit_calibration']
NLI_NEW=['nli_supervised_residual_adaptation','old_linear_NLI_intercept','balanced_NLI_intercept']
LABELS={'raw':'原模型','old_linear':'原16维线性（MEGA fit校准）',
 'old_balanced':'旧平衡MLP（MEGA fit校准）','optimized_balanced_fit_calibration':'旧平衡MLP（MEGA fit校准）',
 'hard_tail_pair_fit_calibration':'难配对MLP，幅度1','hard_tail_robust_fit_calibration':'稳健损失MLP，幅度1',
 NEW[0]:'难配对MLP（dev保护幅度）',NEW[1]:'稳健损失MLP（dev保护幅度）',
 NLI_NEW[0]:'NLI监督残差权重适配',NLI_NEW[1]:'原16维线性＋NLI train截距',NLI_NEW[2]:'旧平衡MLP＋NLI train截距'}

def load(path):
 p=Path(path)
 return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
def num(v):return '待完成' if v is None else f'{int(v):,}'
def pct(v):return '—' if v is None else f'{float(v)*100:.2f}'
def dec(v):return '—' if v is None else f'{float(v):.4f}'
def label(v):return LABELS.get(v,v)
def table(headers,rows):
 return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']+['---:']*(len(headers)-1))+'|']+
  ['| '+' | '.join(str(v).replace('|','／').replace('\n',' ') for v in row)+' |' for row in rows])+'\n'
def ratio(a,b):return '待完成' if a is None or b is None else num(a)+'/'+num(b)
def atomic(path,text):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_name(path.name+'.tmp');temp.write_text(text,encoding='utf-8');temp.replace(path)
def insert(text,tag,body,anchor=None):
 block='<!-- '+tag+':start -->\n'+body.rstrip()+'\n<!-- '+tag+':end -->'
 pattern=re.compile('<!-- '+re.escape(tag)+':start -->.*?<!-- '+re.escape(tag)+':end -->',re.S)
 if pattern.search(text):return pattern.sub(lambda m:block,text)
 if anchor:
  match=re.search(anchor,text,re.M)
  if not match:raise ValueError('Missing document anchor '+anchor)
  return text[:match.start()]+block+'\n\n'+text[match.start():]
 return text.rstrip()+'\n\n'+block+'\n'
def collect(root):
 names={'dev':'dev_metrics.json','protected':'protected_dev_metrics.json','plan':'plan.json',
  'ensembles':'protected_ensembles.json','augmentation':'fit_augmentation/inventory.json',
  'linear':'linear/selected.json','pairs':'evaluation/error_pair_metrics.json',
  'supervised':'nli_adaptation/error_retest/metrics.json','supervised_pairs':'nli_adaptation/error_retest/pair_metrics.json',
  'supervised_selected':'nli_adaptation/selected.json','intercept':'nli_intercept_calibration/error_retest/metrics.json',
  'intercept_pairs':'nli_intercept_calibration/error_retest/pair_metrics.json',
  'mega':'evaluation/results/mega_official_test/summary/metrics.json',
  'nli':'evaluation/results/nli4ct_all/summary/metrics.json'}
 return {name:load(root/path) for name,path in names.items()}
def official_group(data,dataset):
 groups=data.get('groups',{})
 keys=['NLI4CT-2024:official_test','NLI4CT-2024:test'] if dataset=='nli' else ['official_test','all']
 return next((groups[k] for k in keys if k in groups),{})
def case_item(data,dataset,method):
 if method==NLI_NEW[0]:return data['supervised'].get('case_metrics_by_split',{}).get('test',{})
 if method in NLI_NEW[1:]:return data['intercept'].get('case_metrics_by_method',{}).get(method,{}).get('test',{})
 x=official_group(data[dataset],dataset).get('methods',{}).get(method,{})
 if not x:return {}
 n=x.get('evaluated');fixed=x.get('path_correct',x.get('correct'))
 return dict(evaluated=n,repaired=fixed,still_wrong=n-fixed if n is not None and fixed is not None else None,missing=x.get('missing'))
def pairs_for(data,dataset,category,method):
 if dataset=='mega':keys=['MEGA_previous_controlled:'+category];sources=['pairs']
 else:
  keys=['NLI4CT-2024:test:all:'+category,'test:all:'+category]
  sources=['pairs','supervised_pairs','intercept_pairs']
 for source in sources:
  for key in keys:
   group=data[source].get('groups',{}).get(key,{})
   if method in group.get('methods',{}):return group['methods'][method],group
 return {},{}
def retained(item):return item.get('mean_evidence_retention',item.get('mean_signed_response_retention',item.get('signed_response_retention')))
def direction(item):return item.get('response_direction_degraded',item.get('gold_direction_degraded'))
def main_table(data,dataset,compact=False):
 methods=BASE+NEW+(NLI_NEW if dataset=='nli' else [])
 rows=[]
 for method in methods:
  case=case_item(data,dataset,method)
  if not case and method not in NEW:continue
  n=case.get('evaluated',case.get('old_error_cases'));fixed=case.get('repaired')
  preserving,_=pairs_for(data,dataset,'A' if dataset=='mega' else 'preserving',method)
  changing,_=pairs_for(data,dataset,'CF' if dataset=='mega' else 'changing',method)
  if compact and dataset=='mega' and method in ['raw','old_linear']:continue
  if compact and dataset=='nli' and method=='old_linear':continue
  rows.append([label(method),ratio(fixed,n),num(case.get('still_wrong')),pct(preserving.get('variance_reduction')),
   pct(retained(changing)),num(direction(changing)) if changing else '待完成',ratio(changing.get('paired_correct'),changing.get('n'))])
 headers=['方法','修复／已复测原错例','仍错',('旧A错对VR%' if dataset=='mega' else '原生保持错对VR%'),
  ('旧C/F错对ER%' if dataset=='mega' else '原生翻转错对ER%'),'方向退化','翻转／证据错对全对']
 return table(headers,rows)
def dev_table(data):
 rows=[]
 for source,methods in [('protected',['old_linear','old_balanced']),('dev',['hard_tail_pair_fit_calibration','hard_tail_robust_fit_calibration']),('protected',NEW)]:
  for method in methods:
   item=data[source].get('methods',{}).get(method,{})
   if not item:continue
   dev=item.get('splits',{}).get('dev',{});cf=dev.get('CF',{});path=item.get('dev_path',{})
   gain=data['ensembles'].get(method,{}).get('gain',1.0)
   rows.append([label(method),dec(gain),ratio(path.get('correct'),path.get('n')),pct(dev.get('AP',{}).get('variance_reduction')),
    pct(cf.get('mean_evidence_retention')),dec(cf.get('correction_rms_over_ap_rms')),num(cf.get('response_direction_degraded'))])
 return table(['方法','追加幅度','dev路径全对','A/P VR%','C/F ER%','C/F修正／原A/P RMS','方向退化'],rows)
def nli_control_text(data):
 sup=case_item(data,'nli',NLI_NEW[0]);old=case_item(data,'nli',NLI_NEW[1]);bal=case_item(data,'nli',NLI_NEW[2])
 point,_=pairs_for(data,'nli','changing',NLI_NEW[0]);oldpoint,_=pairs_for(data,'nli','changing',NLI_NEW[1]);balpoint,_=pairs_for(data,'nli','changing',NLI_NEW[2])
 if not sup:return 'NLI独立监督适配及标量校准结果尚未完成。\n'
 return ('NLI监督残差适配在官方原错池修复'+ratio(sup.get('repaired'),sup.get('evaluated',sup.get('old_error_cases')))+
  '，但真实翻转错对的平均有向ER仅'+pct(retained(point))+'%，修正RMS为'+dec(point.get('correction_rms'))+
  '，方向退化'+num(direction(point))+'对。原16维方法和旧平衡MLP仅在NLI train拟合一个截距，分别修复'+
  num(old.get('repaired'))+'和'+num(bal.get('repaired'))+'例，ER分别保持'+pct(retained(oldpoint))+'%和'+pct(retained(balpoint))+
  '%，与各自校准前函数一致。截距在配对两端抵消，因此改变分类阈值而不改善配对响应。'
  '这说明监督权重适配的条件修复并未显示优于简单截距对照，且存在更大的响应保护代价；'
  '两类方法的修复集合并不完全重合，不能把权重适配修复的案例全部归因于截距。\n')
def scope_text():
 return ('所有复测池事先固定为原16维线性方法（old_linear）的原错误案例；MEGA官方test为424条动态路径，NLI官方test为1,469条二元记录。'
  '本轮新模型及新截距只复测这些原错例，原正确全集没有重新评价，因此新增错误和全量净收益未知。'
  '8.14节中旧模型的32／30例新增错误背景不适用于本轮新模型。此前官方全量结果和8.14旧冻结模型复测均保留。')
def section(data):
 a=data['augmentation'];p=data['plan']
 text=('### 8.15 针对错误原因的新训练与原错例复测\n\n'
  '行为忠实性仍为主要评价目标；条件修复率用于检查在既有错误上是否有用，不能单独证明忠实性提高或全量任务能力提升。'+scope_text()+'\n\n'
  '**训练数据与冻结流程。** 从原fit缓存gold节点中去掉与原761个自然节点重复的96个case／节点后，新增'+num(a.get('new_supervised_nodes',1720))+
  '个带标签节点，来自原80个fit来源中的'+num(a.get('cached_fit_sources',12))+'个来源，没有新增dev或test来源。'
  '这些节点不是1,720个独立病例，也不代表80个fit来源均获得同等覆盖。原761节点与新增节点各占自然监督损失的0.5，组内按来源均衡。'
  '两种预先指定损失（难配对平方损失、稳健损失）各用11／22／33三个种子，共6次训练，每次'+num(p.get('steps',1200))+
  '步；仍用4,887个fit A/P配对和135个fit C/F配对，模型是原平衡MLP的延续训练。训练后按原fit逐对C/F预算δ=0.2923349缩放输出，'
  '再只用原761节点交叉熵校准截距。三种子输出等权平均；种子波动与论文第9、10节的模拟生成波动分开解释。\n\n'
  '**开发集结果与失败尝试。** 幅度1时两个新集成均仅381／420条路径全对，并损伤dev证据响应。'
  '随后对每个既定集成仅解析计算一个幅度 g=min(1,0.1×原dev A/P RMS／该模型dev C/F修正RMS)，'
  '将C/F修正RMS控制在原dev A/P RMS的0.1；不对官方test选择幅度，截距仍只按原fit校准。'
  '这里的dev均方保护与原方法在fit上的逐对硬预算应分别理解。\n\n'+dev_table(data)+'\n')
 linear=data['linear']
 text+=('另一个CPU对照在旧平衡输出上拟合32维线性残差，使用fit难A/P权重、来源均衡自然监督和逐对C/F硬约束。'
  '预定幅度0／0.25／0.5／0.75／1经dev选择为'+dec(linear.get('selected_amplitude'))+
  '：所有正幅度均超出dev的0.1保护档，路径也未改善，因此未将非零线性残差用于官方复测。\n\n'
  '**MEGA官方原错误路径。** 下表路径修复分母为424；忠实性列另取此前已使用的MEGA受控留出原错误配对（A为27对、C/F为21对），'
  '它们不是这424条官方路径的新配对指标。P与F0没有原错误对；官方MEGA C/F和F0仍缺项，不能把本轮写成完整官方忠实性验证。\n\n'+
  main_table(data,'mega')+'\n'
  '**NLI官方原错误记录与原生错误配对。** 单例修复分母为1,469；配对忠实性使用实际标签保持的1,606个原错对和真实翻转的336个原错对，'
  '两种分母不可互换。NLI原生变体不是MEGA A/P/C/F。原16维及旧平衡方法采用MEGA fit校准，两个新MLP直接零样本迁移；'
  '下面的NLI监督适配与NLI train截距校准则使用NLI train的1,700条标签，单列为任务内监督对照。\n\n'+main_table(data,'nli')+'\n'+nli_control_text(data)+'\n')
 sel=data['supervised_selected'];fd=sel.get('selected_fit_dev_metrics',{})
 if fd:
  text+=('监督残差适配在NLI dev200上选择16维、λ=0.01；train为'+ratio(fd.get('train',{}).get('correct'),fd.get('train',{}).get('n'))+
   '，dev为'+ratio(fd.get('dev',{}).get('correct'),fd.get('dev',{}).get('n'))+'。两个截距对照仅在train1700最小化交叉熵，dev只检查、不扫阈值。'
   '监督残差适配没有加入NLI原生配对保护约束；截距对照保持各自原修正函数。所有拟合都未使用官方错例标签。\n\n')
 text+=('本轮结果应按各自训练条件与分母解读：dev响应保护不自动保证跨来源或跨任务的难例响应，'
  '分类阈值校准带来的修复也不是证据响应改善。所有新模型、新截距的原正确全集均未测，不能推算全量准确率或净收益。'
  'NLI其他split细项保留在结果包；practice包含200条dev原记录及共享family，不作为独立新增测试证据。'
  '完整配置、逐例原错复测和配对分项见 [hard_optimization](meeting/primary_full/hard_optimization/)。\n')
 return text

def brief(data):
 return ('### 针对错因重新训练后的复测\n\n'+scope_text()+'\n\n'
  '新增1,720个原fit节点仅覆盖原80来源中的12个。两损失×三种子训练，幅度1的集成都退至381／420条dev路径；'
  '按dev证据修正0.1档解析缩放后为385／383条，非零线性残差未入选。\n\n'
  '**MEGA官方原错路径；忠实性列来自旧受控原错配对（A27、C/F21），非官方路径配对。**\n\n'+main_table(data,'mega',True)+'\n'
  '**NLI官方原错记录；原生保持／翻转错对分别1,606／336。**\n\n'+main_table(data,'nli',True)+'\n'+nli_control_text(data)+'\n'
  'NLI train截距改变阈值但保持各自配对响应；条件修复较多不等于忠实性提高。'
  '两种新MEGA MLP未优于旧平衡模型的原错例修复；NLI监督残差也未显示优于单一截距对照，不能称全量提升。详细训练条件和分项见正文8.15及结果包。\n')

def copy_packet(root,packet):
 paths=['plan.json','data.json','protected_frozen_methods.json','protected_ensembles.json','ensembles.json',
  'dev_metrics.json','protected_dev_metrics.json','validation.json','protected_validation.json',
  'fit_augmentation/inventory.json','linear/selected.json','linear/training.json','linear/dev_amplitudes.json',
  'evaluation/evaluation_manifest.json','evaluation/plan.json','evaluation/status.json']
 for pattern in ['evaluation/error_pair*','evaluation/results/*/summary/*.json','evaluation/results/*/summary/*.csv',
  'nli_adaptation/selected.json','nli_adaptation/plan.json','nli_adaptation/fit_dev_summary.json',
  'nli_adaptation/error_retest/*.json','nli_adaptation/error_retest/*.csv',
  'nli_intercept_calibration/selected.json','nli_intercept_calibration/plan.json','nli_intercept_calibration/fit_dev_summary.json',
  'nli_intercept_calibration/old_linear/selected.json','nli_intercept_calibration/old_linear/fit_dev_summary.json',
  'nli_intercept_calibration/error_retest/*.json','nli_intercept_calibration/error_retest/*.csv']:
  paths += [str(p.relative_to(root)) for p in root.glob(pattern) if p.is_file() and p.suffix in ['.json','.csv']]
 for rel in sorted(set(paths)):
  source=root/rel
  if source.exists():
   target=packet/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
 # Save only the newly trained small readouts and their costs, not feature arrays.
 for pattern in ['tail_pair/*.pt','tail_robust/*.pt','tail_pair/*.json','tail_robust/*.json',
                 'nli_adaptation/selected_model.npz','linear/frozen_model.npz',
                 'nli_intercept_calibration/calibration.npz','nli_intercept_calibration/old_linear/calibration.npz',
                 'evaluation/results/*/shard*/done.json']:
  for source in root.glob(pattern):
   target=packet/source.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
 for source in root.rglob('*.py'):
  if any(part.startswith('_writer') for part in source.relative_to(root).parts):continue
  if source.is_relative_to(packet):continue
  target=packet/'code'/source.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
 # Only final case decisions are compressed; nodes, prompts, patients and feature caches are excluded.
 for source in root.glob('evaluation/results/*/shard*/results.jsonl'):
  target=packet/source.relative_to(root);target=target.with_suffix('.jsonl.gz');target.parent.mkdir(parents=True,exist_ok=True)
  with source.open('rb') as src,gzip.open(target,'wb') as dst:shutil.copyfileobj(src,dst)


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--root',type=Path,default=ROOT)
 p.add_argument('--publish-worktree',type=Path,default=PUBLISH)
 p.add_argument('--standalone-root',type=Path,default=STANDALONE)
 p.add_argument('--dry-run',action='store_true')
 args=p.parse_args();data=collect(args.root)
 if not data['plan']:raise ValueError('Missing hard-optimization plan.json')
 report=args.publish_worktree/REPORT
 paper=report/'paper_story_latest.zh.md';meeting=report/'meeting/meeting_brief.zh.md';readme=report/'README.md'
 values={paper:insert(paper.read_text(encoding='utf-8'),'aster-hard-optimization-main',section(data),r'^## 9\.[^\n]*'),
  meeting:insert(meeting.read_text(encoding='utf-8'),'aster-hard-optimization-brief',brief(data),r'^## 4\.[^\n]*'),
  readme:insert(readme.read_text(encoding='utf-8'),'aster-hard-optimization-readme',
   '正文8.15及简版补充了针对错因的新训练和原错例条件复测；原正确全集未评价，不能据此推算全量净收益。配置、分项、逐例结果及代码见 [hard_optimization](meeting/primary_full/hard_optimization/)。')}
 if args.dry_run:
  print(json.dumps({'dry_run':True,'targets':[str(x) for x in values],'official_summaries':{k:bool(data[k]) for k in ['mega','nli']},'new_pair_metrics':bool(data['pairs'])},ensure_ascii=False));return
 for path,text in values.items():atomic(path,text);atomic(args.standalone_root/path.relative_to(report),text)
 packet=report/'meeting/primary_full/hard_optimization';packet.mkdir(parents=True,exist_ok=True);copy_packet(args.root,packet)
 for source in packet.rglob('*'):
  if source.is_file():
   target=args.standalone_root/source.relative_to(report);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
 print(json.dumps({'state':'complete','updated':[str(x) for x in values],'packet':str(packet),'git_actions':False},ensure_ascii=False))
if __name__=='__main__':main()
