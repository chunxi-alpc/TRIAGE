import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
NAMES={'static_fusion':'静态概率融合','dynamic_fusion':'动态概率融合','legacy_attention_residual':'原全层动作注意力残差','dynamic_midlate4':'新稀疏动态引导（主配置）','dynamic_half':'新稀疏动态引导（半强度）','dynamic_late4':'新稀疏动态引导（更后层）','dynamic_midlate2':'新稀疏动态引导（2头）','static_sparse':'同位置静态注意力偏置','misaligned':'动作支持错位','reverse':'反向动态偏置','sign_flip':'患者固定符号翻转','zero_strength':'零强度残差','numeric_drift':'仅数值漂移','no_residual':'去掉残差'}
def main():
 r=json.loads((HERE/'results.json').read_text());p=r['methods']['dynamic_midlate4'];pair=p['matched_pair']
 lines=['# TRIAGE / ASTER：Qwen3-8B seed=0 稀疏动态注意力实验','',
 '**状态：800 条真实 GPU 前向已完成并核对；新实验共 11 种配置、8,800 次前向。旧的无干预及原动作偏置结果复用 PR #33，不重复整套计算。所有结果均为已研究开发集上的探索性结果。**','',
 f"主配置正确 {p['metrics']['correct']}/800（{p['metrics']['accuracy']:.3%}）。相对本配置相同融合参数的无残差对照，纠正 {pair['corrected_vs_comparator']} 条、退化 {pair['regressed_vs_comparator']} 条，净变化 {pair['corrected_vs_comparator']-pair['regressed_vs_comparator']:+d} 条。",
 f"逐树描述性 95% 区间为 [{pair['tree_cluster_percentile_ci95']['accuracy_delta'][0]*100:+.3f}, {pair['tree_cluster_percentile_ci95']['accuracy_delta'][1]*100:+.3f}] 个百分点。只有 4 棵已反复研究的开发树，该区间不计方法设计选择造成的偏差，不能视为新的独立确认。",'',
 f"不加概率融合时，新动态主配置为 {r['raw']['dynamic_midlate4']['correct']}/800，同路径零强度为 {r['raw']['zero_midlate4']['correct']}/800，配对纠正 {r['raw_paired_vs_matched_zero']['dynamic_midlate4']['corrected_vs_comparator']}、退化 {r['raw_paired_vs_matched_zero']['dynamic_midlate4']['regressed_vs_comparator']}。但加入既有概率融合后，主配置与同参数无残差持平，因此本次没有证明新的注意力残差带来稳定额外收益。",'',
 f"反向偏置单独选参为 {r['methods']['reverse']['metrics']['correct']}/800，但使用主配置相同参数时只有 {r['matched_primary_controls']['reverse']['metrics']['correct']}/800。不得挑选前者宣称反向方法已被独立证实，也不能把不同调参结果的差异全部归因于注意力方向。",'',
 '## 数据、来源与公平配置','',
 '- 基于 PR #33 `a8c1e49c0fbda941282daf25908004407eb0b1bf` 和 PR #34 `72acfa00ad1f4a4041e8cf6eeaa5341f6184522d`；新分支保留全部旧负结果。',
 '- 200 名开发患者，每名 NC/AC/IC/SW 四卷，共 800 条；按患者分到 GPU 0、1、6，所有卷都保留。只读取原开发病例和其旧数值记录，没有读取 P3 测试标签。',
 '- 对全部样本核对条件输入键、条件概率、规则到动作映射、选项到动作映射、动作支持及标签；模型配置和病例文件哈希与旧运行一致。权重使用同一服务器路径，当前权重文件哈希另存；旧运行未保存权重内容哈希，不能事后声称完成跨运行权重哈希一致性证明。',
 '- 每个融合/残差方法统一 27 组参数：beta={0.5,0.75,1}、T0={1,2,4}、TS={0.25,0.5,1}。每次留一开发树，在其余树按树平均准确率选参，平局取固定网格首项。所有配置、全体预测和各卷结果均保留。',
 '- 主配置固定为中后层4头、强度1，不在所有结构消融里挑最好者冒充主结果。其他结构各自保留完整27组探索结果；另以主配置每折参数计算全部配对消融。',
 '- 各融合方法使用相同原模型概率与动作支持；残差使用各自同路径零强度输出，不能直接减去另一计算路径的旧原始 logits。原全层残差使用 PR #33 对应 `zero_all`。',
 '- 结果是规则遵循反事实标签上的准确率，未进行新增临床有效性验证。','',
 '## 新注意力算子','',
 '仅在 `Answer:` 回答后缀对应的查询位置干预，前文查询和未选头保持原 SDPA 路径。主配置层索引为 18、22、26、30，头索引为 0、8、16、24（均从0计数）；后层消融为 28、30、32、34；2头消融为 0、16。Qwen3-8B 为36层、32个查询头、8个KV头，按查询头所属KV组正确展开。','',
 '每个选中头/查询先计算当前注意力，将同动作的所有规则动作片段与选项片段合并为组。令 m 为可见动作组当前注意力的归一化分布，s 为条件动作支持在可见组上的归一化分布，给组内每个可见键加入 `strength * clip(log(max(s,1e-8))-log(max(m,1e-8)), -2, 2)`；没有规则对应的选项与非动作键不加偏置。支持总量为0时不干预。规则支持按同动作求和，重复选项属于同一组，不重复生成独立动作支持。','',
 '偏置后的 softmax 可改变动作与非动作的总注意力比例；本方法不声称保持动作总质量。显式取因果遮挡与输入遮挡交集，禁止向未来或被屏蔽键分配注意力。选中位置在零强度时也执行相同FP32重算；其余位置仍用原路径。当前实现仅支持无缓存的完整前向，缓存解码输入会明确报错。','',
 '输出有界残差为 `0.25*tanh((干预选项logits-匹配零强度选项logits)/2)`，加入融合概率的对数后取最大项。每个选项偏移绝对值不超过0.25。错位在动作组支持上循环移动；反向直接反转GPU内部偏置；符号翻转按患者标识的固定哈希决定并对四卷保持一致，均为真实新GPU前向。','',
 '## 完整开发树留一结果','',
 '| 方法 | 正确/800 | 总体 | NC | AC | IC | SW | 相对原模型纠正/退化 |','|---|---:|---:|---:|---:|---:|---:|---:|']
 for k,v in r['methods'].items():
  m=v['metrics'];vs=m['by_variant'];lines.append(f"| {NAMES[k]} | {m['correct']} | {m['accuracy']:.3%} | "+' | '.join(f"{vs[x]['accuracy']:.1%}" for x in ('NC','AC','IC','SW'))+f" | {m['corrected']}/{m['regressed']} |")
 lines+=['','纠正/退化均相对全部800条的原模型结果；各卷计数详见 `results.json`。同患者四卷不是独立样本。','',
 '## 主配置相同参数的消融','',
 '| 对照 | 正确/800 | 主配置相对该对照纠正/退化 | 主配置净增 |','|---|---:|---:|---:|']
 for k,v in r['matched_primary_controls'].items():
  d=v['paired_primary_vs_control'];lines.append(f"| {NAMES[k]} | {v['metrics']['correct']} | {d['corrected_vs_comparator']}/{d['regressed_vs_comparator']} | {d['corrected_vs_comparator']-d['regressed_vs_comparator']:+d} |")
 lines+=['','零强度残差与去掉残差理论上相同；单独的“仅数值漂移”使用零路径减旧原路径，不能将它解释成注意力有效性。','',
 '## 主配置相对同参数无残差：逐树配对','',
 '| 开发树 | 样本数 | 纠正 | 退化 | 准确率变化 |','|---|---:|---:|---:|---:|']
 for t,d in pair['by_tree'].items():lines.append(f"| {t} | {d['n']} | {d['corrected_vs_comparator']} | {d['regressed_vs_comparator']} | {d['accuracy_delta']*100:+.3f} 个百分点 |")
 lines+=['','## 不加概率融合的原始前向结果','',
 '| 前向 | 正确/800 | NC | AC | IC | SW |','|---|---:|---:|---:|---:|---:|']
 for k,m in r['raw'].items():lines.append(f"| {k} | {m['correct']} | "+' | '.join(f"{m['by_variant'][v]['accuracy']:.1%}" for v in ('NC','AC','IC','SW'))+' |')
 lines+=['','## 数值漂移和覆盖率','',
 '| 零强度路径 | 相对旧无干预改答数 | 最大选项logit差 | 平均绝对差 |','|---|---:|---:|---:|']
 for k,v in r['numerical_drift'].items():lines.append(f"| {k} | {v['changed']} | {v['max_abs']:.8g} | {v['mean_abs']:.8g} |")
 c=r['coverage'];lines+=['',f"母集800条、200名患者。动作支持覆盖 {c['covered']}/800；多规则合并到同一动作的样本 {c['rule_action_many_to_one']} 条；重复动作选项样本 {c['option_action_many_to_one']} 条；有未对应规则选项的样本 {c['orphan_option_rows']} 条。上述样本均未剔除。",'',
 '## 实际性能成本','',
 '| GPU | 样本 | 前向次数 | 墙钟分钟 | 峰值分配显存GiB | 峰值保留显存GiB |','|---|---:|---:|---:|---:|---:|']
 gpu_seconds=0
 for gpu,v in r['resources'].items():
  s=v['status'];gpu_seconds+=s['elapsed_seconds'];lines.append(f"| {gpu} | {s['total']} | {s['forward_calls']} | {s['elapsed_seconds']/60:.2f} | {s['peak_allocated_GiB']:.3f} | {s['peak_reserved_GiB']:.3f} |")
 lines+=['',f"全量任务合计 {gpu_seconds/3600:.3f} GPU小时（含模型加载与写盘，不含此前旧实验）。另有4条×11配置的小试，14.06秒、44次前向。计费单价未知。",'',
 '| 配置 | 单条平均秒 | P95秒 | 相对新零强度路径耗时 |','|---|---:|---:|---:|']
 for k,v in r['performance'].items():lines.append(f"| {k} | {v['mean_seconds']:.4f} | {v['p95_seconds']:.4f} | {v['relative_to_new_zero']:.3f} |")
 lines+=['','以上是同次运行中各完整前向的成本。生成一个残差需要干预及相应零路径；多配置可以共享同几何零路径。条件概率本次缓存复用，其历史计算成本不在新增GPU小时内。','',
 '## 数值测试、产物和限制','',
 '- 19项CPU测试通过，三卡BF16测试通过；351组融合候选全部核对，匿名数值CPU复算与结果逐项完全一致（见 `audit.json`）。CPU测试覆盖因果遮挡、全屏蔽行、GQA、选择头/查询隔离、组重编号不变性、规则与选项多对一、无支持回退、动态偏置方向及边界、长序列、非法输入，以及开发树选参不读取留出树标签。每个GPU分片还运行BF16算子测试。详见 `logs/` 与 `results.json`。',
 '- `numeric/predictions.jsonl`：800条匿名数值、完整新旧分数、动作映射、支持与逐配置耗时；不包含病例、规则或条件原文。`predictions.csv`：各方法留树预测。`results.json`：所有27项候选、逐卷/逐树计数、折参数、匹配对照和资源信息。',
 '- `README.zh.md`：服务器命令与离线复算说明；`run_gpu.py`、`sparse_attention.py`、`sparse_engine.py`：可复用实现。',
 '- Qwen一个种子的本次范围已完成。主配置未在既有概率融合上取得额外净收益，暂不建议直接扩展其他模型。没有新独立标注集，因此不声称独立确认。尤其不得将已查看的P3重新包装成新测试。',
 '- 保留原负结果及所有消融，无有利子集筛选。即使本开发集某些配置更高，也不足以证明稳定泛化优势。','']
 validation=json.loads((HERE/'runtime_validation.json').read_text())
 lines += ['## 缓存与运行版本抽查','',f"固定等距抽取8条、16次补充前向核对原模型缓存和新零路径；原模型缓存最大绝对差 {max(x['original_cache_max_abs'] for x in validation['results']):.8g}，答案一致 {sum(x['original_cache_same_answer'] for x in validation['results'])}/8。抽查不依赖标签，没有整套重跑旧800条。核心GPU源码哈希与三个运行清单全部一致。补充核对耗时 {validation['elapsed_seconds']:.2f} 秒（不含模型初始化）。",'']
 (HERE/'REPORT.zh.md').write_text('\n'.join(lines))
if __name__=='__main__':main()
