from .common import *
LABELS={'raw':'原始 Qwen3-8B','aster':'ASTER-CR E1 16维线性','leace':'LEACE（历史检查点）','lofit':'LoFiT（历史检查点）','lora':'LoRA（历史检查点）','cad':'CAD（双前向）'}
MODES={'output_patch':'输出修补','output_qk_frozen':'输出修补＋QK冻结','ov_values_qk_frozen':'OV值路径＋QK冻结','routing_only_qk_frozen':'仅替换注意力权重','projection_native':'原输出投影（加法）','projection_swap':'随机头输出投影（加法）'}
def e9():
 info=read(OUT/'e9_statistics.json');cfg=read(OUT/'configs/e9.json');stats=info['summary'];contrasts=info['contrasts']
 def cell(mode,group,kind='all'):return next(r for r in stats if r['group']==group and r['intervention']==mode and r['kind']==kind)
 def contrast(name,kind='all',metric='abs_score_change_difference'):return next(r for r in contrasts if r['contrast']==name and r['kind']==kind and r['metric']==metric)
 head=contrast('head_selection');proj=contrast('output_projection_selected')
 aphead=contrast('head_selection','AP');cfhead=contrast('head_selection','CF');vr=contrast('value_vs_route_selected')
 qk=contrast('qk_dependence_selected',metric='absolute_between_conditions_score_difference')
 text=f"""# E9：冻结 E1 的 M1 机制分析

## 完成范围

本次新增 {cfg['pair_count']} 对开发集样本，来自 {info['unique_sources']} 个来源；A/P/C/F 分别为 40/3/18/10 对。每个来源、每类按 pair_id 取前两对，双向干预。没有依据标签、分数或结果筛选样本、头或路径。每个头组 6 个条件，共 {info['prediction_rows']} 条干预预测。

模型为冻结的 Qwen3-8B，BF16、SDPA、批量 1；先在GPU 0运行，之后仅将未完成记录续跑到GPU 5，两卡均为A100-SXM4-80GB。M1 固定使用 E1 的 51 个头，M2/M3 固定使用 E1 的 16 维参数。原模型分数是最终判断位置 FP32“是”减“否”logit；另报告 E1 最终分数 ell + b + ((h − mu)/scale) N w。h 使用干预实际生效后的头输出。干预作用于最终判断 token，前缀保持接收样本的原始 KV Cache。没有训练或重新选择参数。

## 已有证据复用

已检查 experiments/aster_cr/mechanisms.py、历史选中/随机头分析、h2/h3、历史激活修补与随机子空间结果。历史头集合与 E1 不同，旧数据保留为历史证据，不混入当前 E1 统计。

- 50 种子随机 51 头：平均 R²=0.0182；50 种子随机16维子空间：平均 R²=0.3115。直接复用[历史痕迹验证报告](../aster_cancel_story_20261109/meeting/trace_validation/REPORT.zh.md)及其统计，没有重跑。
- 历史 h2：16维干扰预测 R²=0.4061，证据泄漏比例约1.03%。
- 历史 h3：选中头在错误节点上的准确率约9.90%，置换对照约58.42%；不支持把可解码信号直接解释成任务收益。
- 本次只有一个固定种子的层匹配随机组：每层头数与E1一致，排除选中头。它补充当前 E1 缺失的干预条件，不替代50种子的历史分布。

复用文件与差异记录保存在 [reused/](reused/) 和 [reuse_manifest.json](reuse_manifest.json)。

## 干预定义

| 条件 | 操作 |
|---|---|
| 输出修补 | 将选定头的最终位置输出替换为配对供体输出，后续层正常计算 |
| 输出修补＋QK冻结 | 同上；所有层最终位置Q/K固定为接收样本原值，前缀K固定，注意力权重不随干预改变 |
| OV值路径＋QK冻结 | 保留接收注意力权重，在对齐位置替换供体V，经原O投影传播；未对齐V保留 |
| 仅替换注意力权重 | 对齐位置使用供体权重，其他位置保留接收权重并归一化；V保留接收值，经原O投影传播，其他QK冻结 |
| 原输出投影（加法） | 将固定的供体−接收头输出差，经原O块加到当前注意力输出 |
| 随机头输出投影（加法） | 同一差向量改经同层随机头O块，按Frobenius范数匹配；其他操作与上一行相同 |

V与注意力权重根据完全相同token的顺序对齐交换，平均覆盖率 {info['alignment']['mean']:.2%}；未对齐token不交换。这是对齐位置范围内的路径干预，不能代表未对齐证据token的全部影响。共享KV按query head展开，仅修改指定query head的贡献。

输出投影的受控比较是两种“加法”条件，不将它们与覆盖式输出修补当作单因素比较。QK冻结控制检验输出修补效果对后续注意力路由的依赖；这些效果并非可以相加的因果贡献比例。

## 最终分数与判断

表中为逐干预事件的平均绝对分数变化。“纠正/新增错误”相对于同一接收样本的原判断；双向接收样本可能重复，因此不把事件计数当作独立任务样本准确率。

| 条件 | 选中头平均绝对分数变化 | 随机头平均绝对分数变化 | 选中头纠正/新增错误 | 随机头纠正/新增错误 |
|---|---:|---:|---:|---:|
"""
 for mode in cfg['interventions']:
  a=cell(mode,'selected');b=cell(mode,'random')
  text+=f"| {MODES[mode]} | {a['raw_abs_delta_mean']:.4f} | {b['raw_abs_delta_mean']:.4f} | {a['raw_corrected']}/{a['raw_new_errors']} | {b['raw_corrected']}/{b['raw_new_errors']} |\n"
 text+="""
标签保持与证据改变分别统计：

| 条件 | 选中头A/P | 随机头A/P | 选中头C/F | 随机头C/F |
|---|---:|---:|---:|---:|
"""
 for mode in cfg['interventions']:
  a=cell(mode,'selected','AP');b=cell(mode,'random','AP');c=cell(mode,'selected','CF');d=cell(mode,'random','CF')
  text+=f"| {MODES[mode]} | {a['raw_abs_delta_mean']:.4f} | {b['raw_abs_delta_mean']:.4f} | {c['raw_abs_delta_mean']:.4f} | {d['raw_abs_delta_mean']:.4f} |\n"
 text+=f"""
C/F供体与接收样本真值不同；判断随供体改变属于机制响应，不能直接视为接收任务改进。

按来源等权、2,000次来源簇重采样：

- 选中头减随机头的输出修补绝对分数变化：**{head['mean']:.4f}**，95%区间 **[{head['ci95_low']:.4f}, {head['ci95_high']:.4f}]**。
- 选中头原O块减随机O块的绝对分数变化：**{proj['mean']:.4f}**，95%区间 **[{proj['ci95_low']:.4f}, {proj['ci95_high']:.4f}]**。
- 输出修补在QK自由与冻结条件下的最终分数平均绝对差：**{qk['mean']:.4f}**，95%区间 **[{qk['ci95_low']:.4f}, {qk['ci95_high']:.4f}]**。

A/P的选中头减随机头差值为 {aphead['mean']:.4f}，95%区间[{aphead['ci95_low']:.4f}, {aphead['ci95_high']:.4f}]；C/F差值为 {cfhead['mean']:.4f}，95%区间[{cfhead['ci95_low']:.4f}, {cfhead['ci95_high']:.4f}]。

在对齐位置范围内，仅替换注意力权重的绝对分数变化大于替换V，来源等权差值为 {-vr['mean']:.4f}，95%区间[{-vr['ci95_high']:.4f}, {-vr['ci95_low']:.4f}]。QK冻结后输出修补仍有较大影响；输出映射与注意力路由都影响最终分数，不能只归因于其中一项。

输出修补前原模型正确130/142个接收事件，干预后98/142；E1修正分数对应130/142与99/142。A/P的86个接收事件均未发生正确性变化；原模型新增32个错误都位于C/F供体交换条件。这里是供体交换诊断，不是用于改善预测的干预策略。

上述来源等权结果与表中逐事件均值的权重不同。E1修正后的全部分数、A/P、C/F及逐类的纠正/新增错误见 [e9_summary.csv](e9_summary.csv)；来源区间见 [e9_paired_contrasts.csv](e9_paired_contrasts.csv)。逐例干预前后分数见 [e9_predictions.jsonl](e9_predictions.jsonl)。

## 结论

"""
 text+=("选中头对最终判断分数的影响大于层匹配随机头，本次来源区间不跨零。\n\n" if head['ci95_low']>0 else "本次来源区间尚不能确立选中头对最终分数的影响稳定大于层匹配随机头。\n\n")
 text+=("原输出投影相较匹配随机投影产生更大的绝对分数变化，支持输出映射对干预效果有贡献。\n\n" if proj['ci95_low']>0 else "输出投影比较尚未显示原投影的稳定优势，不能仅凭局部头输出变化推断输出映射的特异性。\n\n")
 text+=f"""局部激活变化、最终分数变化幅度和供体信息传递不直接等于任务性能改善。本报告分别列出纠正与新增错误；未开展新的完整路径评价，不声称干预提高整体任务性能。P类只有3对、2个来源；单个随机头组也不能确立跨种子的稳定性。

自我修补控制的最大logit差为 {info['identity_max_abs_difference']:.8f}。实现核对记录和初版输出保留在logs/，正式统计只使用当前有效输出。冻结E1参数、样本和头索引均已保存。

## 复现

服务器工作区为 /home/gcx/aster-e9-e11-20261112。运行 bash reports/aster_e9_e11_20261112/run_e9.sh，再运行 python -m experiments.aster_e9_e11_20261112.summarize。

配置和输入在configs/，参数快照在checkpoints/，运行日志在logs/。脚本按已保存键跳过完成记录；新独立重复需另设输出目录。没有运行已有50种子随机实验、任何禁止重复的内部消融或E10，也没有修改论文正文。
"""
 (OUT/'E9_REPORT.zh.md').write_text(text,encoding='utf-8')
def e11():
 cfg=read(OUT/'configs/e11.json');result=read(OUT/'e11_statistics.json');stats=result['summary'];infos=read(OUT/'e11_method_info.json')
 text=f"""# E11：统一模型推理效率

## 完成情况

已实测原始Qwen3-8B、冻结E1的ASTER-CR 16维线性、LEACE、LoFiT、LoRA及CAD，共 {result['measurements']} 次计时。**E11尚待通过有效性审计的ITI，以及新增方法完成后的代表方法接入。** Codex00/01尚未完成；Codex03尚无已完成新增方法。本轮没有以未经审计ITI或未完成方法填表。

LEACE、LoFiT、LoRA使用历史已保存检查点，本报告只比较推理成本，不将它们标为当前E1数据有效性审计已通过。没有重新训练方法，没有重跑已有特征提取测试或CPU后处理微基准。

## 统一协议

- 同一服务器GPU 1：NVIDIA A100-SXM4-80GB；同一Qwen3-8B、BF16、SDPA、批量1。
- 固定 {cfg['inputs']} 个开发prompt，各方法使用完全相同的末尾512或2,048个token。截断用于固定长度工作负载，不据此评价任务准确率。
- 每种方法每档预热 {cfg['warmup']} 次，随后 {cfg['repeats']} 轮、每轮 {cfg['inputs']} 个输入，共每档20次。所有方法每轮输入次序相同。
- 端到端包括模板、分词、CPU→GPU、预填充、最终判断前向、方法修正与分数返回CPU；不包含模型/适配器加载。预填充与端到端计时均使用CUDA同步。
- 输出为单次“是/否”分数判断，没有长文本生成。1次逻辑前向包括预填充和最终token两次底层模型调用。
- CAD使用相同主输入及原实现的无规则参考输入；两次逻辑前向、参考分词和全部预填充均计入。参考输入按相同上限截断，实际长度另列。
- LoFiT、LoRA前缀在相应适配器下计算；LEACE按历史实现作用于最终判断位置。ASTER只收集选中头并计算16维线性修正，无额外参考前向。
- GPU峰值allocated包含模型、活动方法与临时张量；reserved另存。KV表记录逻辑缓存tensor字节数，前缀回放副本和临时分配计入实测显存峰值。
- 所有方法推理时的可训练参数均为0。方法间固定基础模型；LoRA保留未合并形式。

## 时间与显存

时间为均值±样本标准差；吞吐量为各轮均值±轮间标准差。CSV同时保存总体吞吐量，不以最快单次代表方法。

| 方法 | 主输入token | 预填充ms | 单样本端到端ms | 吞吐量 样本/s | 峰值显存GiB |
|---|---:|---:|---:|---:|---:|
"""
 for r in stats:
  text+=f"| {LABELS[r['method']]} | {r['length']} | {r['prefill_ms_mean']:.2f} ± {r['prefill_ms_std']:.2f} | {r['latency_ms_mean']:.2f} ± {r['latency_ms_std']:.2f} | {r['throughput_repeat_mean']:.2f} ± {r['throughput_repeat_std']:.2f} | {r['peak_allocated_gib_max']:.3f} |\n"
 text+="""
## 参数、存储与前向

训练参数指拟合时经优化器训练的数目；闭式求解另有保存矩阵。部署tensor统计推理所需数值及头索引，检查点可能另含训练元数据；本次实际加载精度与文件字节数保留在统计中。

| 方法 | 训练参数 | 部署数值KiB | 保存检查点KiB | 逻辑前向/底层调用 |
|---|---:|---:|---:|---:|
"""
 for name,info in infos.items():
  text+=f"| {LABELS[name]} | {info['trainable_parameters']:,} | {info['deployment_tensor_bytes']/1024:.3f} | {info['storage_bytes']/1024:.3f} | {info['logical_forwards']}/{2*info['logical_forwards']} |\n"
 text+="""
LoFiT第一阶段另有147,456个头选择参数，最终偏移参数为128个；推理不需要保留选择变量。LEACE训练参数记0表示闭式求解，不表示没有拟合或没有额外状态。ASTER线性修正为16个权重及1个截距；冻结M1/M2的存储另计。

## KV Cache

| 方法 | 主输入token | 主输入KV MiB | 参考输入平均token | 参考KV MiB |
|---|---:|---:|---:|---:|
"""
 for r in stats:
  text+=f"| {LABELS[r['method']]} | {r['length']} | {r['kv_cache_bytes_mean']/2**20:.3f} | {r['reference_tokens_mean']:.1f} | {r['reference_kv_cache_bytes_mean']/2**20:.3f} |\n"
 text+="""
CAD顺序执行两个输入，同时存活的逻辑KV取较大值；实际峰值仍由显存计量给出。预填充时间也已包含参考输入。

## 直接复用的既有性能结果

[extraction_performance.json](reused/extraction_performance.json)：单进程3.8228输入/s，双进程合计3.7932输入/s，峰值约15.91 GiB。历史测试在GPU 6、约2,273 token输入上进行，不与本次GPU 1固定长度结果混算。

[evaluation_performance.json](reused/evaluation_performance.json)：配对统计加速3.5917倍、路径统计加速10.5983倍。这些是CPU后处理数据，未再次计时，也不代表模型推理加速。

## 本次观察

"""
 for length in cfg['lengths']:
  a=next(r for r in stats if r['method']=='aster' and r['length']==length);b=next(r for r in stats if r['method']=='raw' and r['length']==length);c=next(r for r in stats if r['method']=='cad' and r['length']==length)
  text+=f"- {length} token：ASTER平均端到端延迟为原模型的 {a['latency_ms_mean']/b['latency_ms_mean']:.3f} 倍；CAD为 {c['latency_ms_mean']/b['latency_ms_mean']:.3f} 倍。\n"
 text+="""
上述比值只对应当前固定输入和实现，结合波动阅读，不据小幅差异推断稳定加速。ASTER增加选中头收集和线性修正，不增加模型逻辑前向；CAD的额外参考前向已计入，参考实际长度也已记录。

## 复现与产物

在 /home/gcx/aster-e9-e11-20261112 运行 bash reports/aster_e9_e11_20261112/run_e11.sh，再运行 python -m experiments.aster_e9_e11_20261112.summarize。

全部分数和重复测量见 [e11_measurements.jsonl](e11_measurements.jsonl)，汇总见 [e11_summary.csv](e11_summary.csv)、[e11_repeat_summary.csv](e11_repeat_summary.csv) 和 [e11_statistics.json](e11_statistics.json)。配置与固定输入在configs/，参数快照在checkpoints/，日志在logs/。软硬件信息见 [e11_environment.json](e11_environment.json) 与 configs/software_versions.json。

脚本按方法、长度、轮次和输入ID跳过已完成记录；新独立重复须另设输出目录。待审计ITI与新增代表方法到位后，只按同一协议补缺行。论文正文未修改。
"""
 (OUT/'E11_REPORT.zh.md').write_text(text,encoding='utf-8')
def main():
 if (OUT/'e9_statistics.json').exists():e9()
 if (OUT/'e11_statistics.json').exists():e11()
if __name__=='__main__':main()
