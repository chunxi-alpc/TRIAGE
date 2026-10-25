"""Publish aggregate evidence and counts only; patient text stays on the server."""
import collections,csv,gzip,json,time,hashlib
from pathlib import Path
from analysis import BASE,ROOT,OUT,HERE,read,save,sha

def main():
 audit={'snapshot_unix':time.time(),'stages':{},'notes':[]}
 for model in ('qwen','llama','evidence'):
  d=BASE/'continuation_v2'/model;st={}
  for stage in ('E2','branches','dev_select','E5','E6','E7','E8'):
   st[stage]={}
   for p in sorted((d/stage).glob('*.jsonl')):
    # A live append can leave an incomplete trailing record; ignore only that tail.
    lines=p.read_bytes().splitlines();rr=[]
    for i,line in enumerate(lines):
     try:rr.append(json.loads(line))
     except json.JSONDecodeError:
      if i!=len(lines)-1:raise
    st[stage][p.stem]={'rows':len(rr),'unique_keys':len({r['key'] for r in rr}),'by_split_seed_variant':dict(collections.Counter(f"{r['split']}|{r['seed']}|{r['variant']}" for r in rr))}
  st['status']=json.loads((d/'status.json').read_text()) if (d/'status.json').exists() else None
  st['locked']= (d/'locked.json').exists();audit['stages'][model]=st
 audit['notes']=['Root STATUS.json is stale; actual files supersede its not_run flags.','E4 is development selection/locked.json in current continuation, not a separately verified complete stage.','E1 earlier engineering failures remain historical negative results; later controls do not erase them.','E2/3 and E5-8 counts are method-specific; nonempty files do not establish completion.','Existing repeat3 repeats the instruction sentence, NOT the whole prompt or full guideline. It does not establish the full-prompt-repetition baseline.','No independent original-plan-completeness claim: initial preregistration and subsequent continuation have different experiment definitions.']
 audit['human_review']={}
 for name in ('A','B'):
  p=BASE/f'annotation_{name}.csv';rr=list(csv.DictReader(p.open(encoding='utf-8-sig')))
  audit['human_review'][name]={'rows':len(rr),'completed_all_four':sum(all(r[k].strip() for k in ('来源一致','条件适用','构造有效','答案正确')) for r in rr)}
 save('legacy_audit.json',audit)
 workload={}
 for phase in ('mismatch','test'):
  workload[phase]={}
  for model in ('qwen','llama'):
   d=ROOT/phase/model
   if (d/'status.json').exists():workload[phase][model]=json.loads((d/'status.json').read_text())
   if (d/'manifest.json').exists():save(f'{phase}_{model}_manifest.json',json.loads((d/'manifest.json').read_text()))
   if phase=='mismatch' and (d/'predictions.jsonl').exists():
    with gzip.open(OUT/f'mismatch_{model}.jsonl.gz','wt') as f:
     for r in read(d/'predictions.jsonl'):f.write(json.dumps(r)+'\n')
 save('workload.json',workload)
 p2=json.loads((OUT/'p2_cross_validation.json').read_text());p0=list(csv.DictReader((OUT/'p0_controls.csv').open()));p1=list(csv.DictReader((OUT/'p1_strata.csv').open()))
 lines=['# ASTER P0–P3：真实执行记录','', '日期：2026-10-25。旧预测仅用于明确标注的复分析；错配和新测试来自服务器实际 GPU 推理。原预测、旧预注册保持不变。','', '## 当前完成状态','', '- P0：温度、均匀概率、动作频数、sum/max/mean、规则重复与去重已完成无 GPU 复分析；错配状态见下表。','- P1：双模型各 800 条开发记录完成逐病例纠正/伤害四格及分树、变体、置信度、规则数分析。','- P2：4 棵开发树留一树交叉验证完成；同一选择器在两模型之间共享参数。','- P3：是否完成以完整计数为准；人工审核未完成时不得称整个阶段完成。','', '## 开发树交叉验证','', '| 方法 | 模型 | 准确率 | 全部病例伤害率 | 覆盖率 |','|---|---|---:|---:|---:|']
 for r in p2['results']:
  lines.append(f"|{r['method']}|{r['model']}|{r['accuracy']:.2%}|{r['harm_all']:.2%}|{r['coverage']:.2%}|")
 lines+=['','伤害率为原无干预答对、干预后答错的记录数/全部记录数；另外提供以原答对记录和改答记录为分母的比例。覆盖率表示采用映射的比例，不等于改答率。','开发集仅 4 棵树，簇 bootstrap 区间不稳定。观察到零伤害不能推出总体风险为零。选择性覆盖降低伤害，但未同时超过两模型的直接映射准确率，不能宣称全面胜出。','', '## P0 关键对照','', '| 模型 | 方法 | 温度 | 聚合 | 准确率 | 并列最高比例 |','|---|---|---:|---|---:|---:|']
 for r in p0:
  if r['variant']=='ALL' and float(r['temperature'])==1 and r['method'] in ('uniform_options','uniform_rules','action_frequency','blind','condition'):
   lines.append(f"|{r['model']}|{r['method']}|{r['temperature']}|{r['aggregation']}|{float(r['accuracy']):.2%}|{float(r['all_tie_rate']):.2%}|")
 lines+=['','均匀规则概率经 sum 映射等价于动作出现频数。温度是对已存概率的事后变换，不是新增采样推理；无法恢复下溢到零的原 logits。并列时首选项硬判决和随机解并列的期望准确率分别报告。','错配在同一树内替换为另一位病人的文本，不使用金标挑选替换病例。条件判断在 NC 上实测，再映射到四种动作版本；旧盲判错配仅比较 NC，保持动作和选项不变。','']
 if (OUT/'p0_mismatch.json').exists():
  for r in json.loads((OUT/'p0_mismatch.json').read_text()):lines.append(f"- 错配 {r['model']}/{r['source']}/{r['aggregation']}：{r['unique_patients']} 位病人；匹配 {r['matched']['accuracy']:.2%}，错配 {r['mismatched']['accuracy']:.2%}。")
 lines+=['','## 冻结与独立测试','']
 if (OUT/'frozen.json').exists():
  lock=json.loads((OUT/'frozen.json').read_text());lines+=['唯一候选已冻结：`'+json.dumps(lock['candidate'],ensure_ascii=False)+'`。','冻结文件 SHA256：`'+sha(OUT/'frozen.json')+'`。测试前固定，后续不得依据测试结果换候选。']
 else:lines+=['唯一候选尚未冻结；尚未启动本阶段测试推理。']
 if (OUT/'p3_status.json').exists():lines+=['','测试记录计数：`'+(OUT/'p3_status.json').read_text().replace('\n',' ')+'`。']
 if (OUT/'p3_metrics.json').exists():
  p3=json.loads((OUT/'p3_metrics.json').read_text());lines+=['','以下是'+('完整' if p3['complete'] else '尚不完整、不可作最终结论的')+'测试结果：','', '|方法|模型|准确率|伤害率|相对无干预差值95%簇区间|','|---|---|---:|---:|---|']
  for r in p3['results']:lines.append(f"|{r['method']}|{r['model']}|{r['accuracy']:.2%}|{r['harm_all']:.2%}|{r['cluster_inference']['delta_ci95']}|")
 lines+=['','NC/AC/IC/SW 每模型每种子各 1000 条，CS/CS-null 各 48 条，共每模型 12,288 条、双模型 24,576 条。三种子是数据构造随机种子；模型本身采用确定性评分，不是三次独立训练。按树抽样 10,000 次，病例的全部版本与种子一起保留；模型分别报告。CS 数量很小，单独报告而不掩盖在核心四卷均值中。','', '## E0–E8 与复杂模块','']
 for m,st in audit['stages'].items():
  s=st['status'];lines.append(f"- {m}：实际状态 {s['stage'] if s else 'unknown'}，{s.get('completed_rows') if s else '?'}/{s.get('total_rows') if s else '?'}；逐方法覆盖见 legacy_audit.json。")
 lines+=['- E0 已有树隔离划分与六类数据；CS 每类只有 48 例，人工审核尚缺。E1 旧负结果保留。E2/E3 有实际记录；E4 按当前代码是开发选择锁定，并非独立完成证明。E5/E6/E8 有实际记录，但必须逐方法检查覆盖；E7 辅助模型完成状态单列。','- 旧 repeat3 实际只重复一句判断指令，不能冒充完整提示重复基线。原计划与续跑实现并不完全相同，因此不声称原 E0–E8 全部完成。','- FlexAttention 已有算子数值验证和极小模型样例一致性；没有吞吐或显存优势证据，本轮不引入。','- SEKA 现有代码依赖模型匹配投影，已有工作不能证明其提升本轮条件判定和伤害控制；本轮没有必要新增其计算。','- GRPO 没有本轮训练结果。当前主要问题可由条件判断与动作计数对照直接检验，本轮不加入强化学习训练。','', '## 可靠性与工作量','', '- 提供 Brier 分数与 10 个等宽箱的 ECE。选择性方法回退时使用无干预概率，覆盖时使用映射概率；这些不是已校准的临床风险。','- 开发树 OOF 对照和独立测试严格分开。三分类条件支持属于模型证据判断，不是人工验证的引文抽取。','- 模型未读取测试金标做决策。新推理明确记录精确输入的条件缓存复用；其复用记录不是额外独立模型调用。','- 错配/测试进程耗时、峰值显存与调用单位见 workload.json；调用单位包括每条规则条件判定及每次回答/分支评分，分支评分内部可能有多次 forward。','- 全部推理在原研究服务器执行，仅 GPU 0/1。运行时长乘设备数是分配 GPU 小时，因与旧任务共享设备，不能等同独占 GPU 时间。','- 两份 150 例人工表目前完成数：'+json.dumps(audit['human_review'],ensure_ascii=False)+'。这部分必须由人工填写，不能由模型代签。','- 私有病历、规则原文和日志不提交；公开输出仅记录标识、概率、预测、聚合统计和文件指纹。','']
 (OUT/'报告.zh.md').write_text('\n'.join(lines))
 print(json.dumps({'workload':workload,'human':audit['human_review']}))
if __name__=='__main__':main()
