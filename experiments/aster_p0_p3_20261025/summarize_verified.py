"""Render independently verified P3 evidence without changing the frozen candidate."""
import json
from pathlib import Path
H=Path(__file__).resolve().parent;R=H/'results';O=R/'verification_20261025';v=json.loads((O/'verified.json').read_text());human=json.loads((R/'human_audit_status.json').read_text())
labels={'none':'无干预','candidate':'冻结候选','threshold':'简单阈值（0.4）','direct':'新条件直接映射','blind_direct':'旧盲判直接映射'}
s=['# P3 独立测试核对（2026-10-25）','','P3 自动推理和统计已完成：两个模型各 12,288 条，合计 24,576 条；不存在缺失或重复键，执行退出码为 0。人工审核仍未完成，因此不将包含人工审核的整个阶段称为全部完成。','','核对从服务器原始预测、条件缓存及私有病例的键集合出发，独立重算每个方法的预测和统计。冻结代码、选择报告、数据指纹匹配，冻结时间早于测试进程启动。原始预测和预注册均未改写；此次只有 CPU 核对，没有重新运行 GPU 或依据测试结果调参。','','## 主要结果：NC/AC/IC/SW','','每模型 1,000 病例 × 4 变体 × 3 数据种子 = 12,000 条，来自与开发树不重叠的 22 棵测试树。','','|模型|方法|准确率|伤害率|覆盖率|纠正数|伤害数|','|---|---|---:|---:|---:|---:|---:|']
for m,z in v['models'].items():
 for method,r in z['metrics']['core'].items():s.append(f"|{m}|{labels[method]}|{r['accuracy']:.3%}|{r['harm_rate']:.3%}|{r['coverage']:.3%}|{r['corrected']}|{r['harmed']}|")
s+=['','伤害率统一定义为“无干预答对、该方法答错”的数量除以全部记录数。覆盖率是选择规则映射的比例，未覆盖仍由无干预回答，并非不回答。原本答对记录中的伤害比例：Qwen 候选 1.976%，Llama 候选 3.876%。多种子和变体重复同一病例，不作为独立样本。','','## 相比阈值和直接映射有没有优势','','- Qwen：候选比阈值低 16.233 个百分点，比新条件直接映射低 18.417 个百分点；准确率差距的簇级区间均低于零。伤害率分别降低 4.750、9.650 个百分点。','- Llama：候选比阈值低 2.142 个百分点，比新条件直接映射低 0.550 个百分点；准确率差距区间均跨零，不能证明优于它们，也不能把不显著当作等效。伤害率分别降低 4.567、18.133 个百分点。','- 相比无干预，候选准确率提升 Qwen 8.450、Llama 4.508 个百分点，两个模型的簇级区间均高于零。','- 结论是降低伤害与降低覆盖/纠正机会之间的权衡。当前结果不支持“候选准确率最佳”或“整体支配简单基线”。也尚无相同覆盖率/相同伤害预算下的对照，不能把低伤害全部归因于证据约束机制。Llama 旧盲判直接映射准确率 61.850%，还高于新条件直接映射 57.300%。','','## 按树配对推断','','整树抽样 10,000 次，保留同一病例的全部版本与种子；模型分别计算。下表为候选减比较方法，单位为百分点；伤害差对双方都以原无干预为参照。区间是点对点 95% 百分位区间，未作多重比较校正。','','|模型|比较对象|准确率差|95% 区间|伤害率差|95% 区间|','|---|---|---:|---|---:|---|']
for m,z in v['models'].items():
 for other,r in z['paired']['core'].items():
  a,b=r['accuracy_delta_ci95'];c,d=r['harm_delta_ci95'];s.append(f"|{m}|{labels[other]}|{r['accuracy_delta']*100:.3f}|[{a*100:.3f}, {b*100:.3f}]|{r['harm_delta']*100:.3f}|[{c*100:.3f}, {d*100:.3f}]|")
s+=['','本次独立复算用等价的整树有放回抽样实现，随机数取法不同导致区间端点与旧文件略有 Monte Carlo 差异；点估计逐项完全一致。原 p3_pairwise.json 的 harm_all 使用比较方法作参照，表示比较方法答对而候选答错，不能当作本表的无干预伤害率；统一口径见本目录 verified.json。','','## CS / CS-null 单独报告','','每模型每类 48 病例 × 3 种子 = 144 条；样本少。','','|模型|变体|方法|准确率|伤害率|覆盖率|','|---|---|---|---:|---:|---:|']
for m,z in v['models'].items():
 for variant in ('CS','CS-null'):
  for method in ('none','candidate','threshold','direct'):
   r=z['metrics'][variant][method];s.append(f"|{m}|{variant}|{labels[method]}|{r['accuracy']:.3%}|{r['harm_rate']:.3%}|{r['coverage']:.3%}|")
s+=['','候选在 CS 上没有产生纠正；在 CS-null 上 Qwen 无新增伤害，Llama 有 3 条伤害、无纠正。不能把核心四卷的收益推广到条件漂移场景。','','## 六类合计（补充，不代替核心结果）','','|模型|方法|准确率|伤害率|覆盖率|','|---|---|---:|---:|---:|']
for m,z in v['models'].items():
 for method in ('candidate','threshold','direct'):
  r=z['metrics']['all'][method];s.append(f"|{m}|{labels[method]}|{r['accuracy']:.3%}|{r['harm_rate']:.3%}|{r['coverage']:.3%}|")
s+=['','## 工作量与未完成事项','']
for m,z in v['models'].items():
 st=z['status'];s.append(f"- {m}：墙钟 {st['elapsed_seconds']/3600:.3f} 小时，峰值模型分配显存 {st['peak_GiB']:.3f} GiB；{z['unique_condition_inputs']} 个独立条件输入，{z['condition_reused_rows']} 条记录明确复用了完全相同输入的条件概率；并未把缓存复用计作额外 GPU 推理。")
s+=['- 两模型累计分配时间约 7.171 GPU 小时，与旧任务共享设备，不能解释为独占 GPU 计算量。','- 新人工审核表完成数：'+json.dumps(human['completed'],ensure_ascii=False)+'，每人应审 150 条，尚待人工填写与分歧裁定。','- ECE/Brier 原始结果已发布，但概率不是经验证的临床风险；反事实任务伤害表示答题由对变错，不等同实际患者伤害。','- 分种子完整结果、原始文件哈希、逐项检查见 verified.json；原分树/变体/置信度文件保持不变。']
(O/'P3核对.zh.md').write_text('\n'.join(s)+'\n')
p=R/'报告.zh.md';text=p.read_text();note='> 2026-10-25 独立核对：P3 自动评估已全部完成，人工审核仍待完成。核心四卷候选准确率 Qwen 55.258%、Llama 56.750%，伤害率 0.925% / 2.025%。相比简单阈值和直接映射主要优势是降低伤害，没有准确率优势。详见 [P3 独立测试核对](verification_20261025/P3核对.zh.md)。\n\n'
if not text.startswith('> 2026-10-25 独立核对：'):p.write_text(note+text)
