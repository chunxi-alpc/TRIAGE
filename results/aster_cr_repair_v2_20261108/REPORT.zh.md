# ASTER-CR 新实验报告

状态：运行中；未完成阶段不作为结果。

执行位置：/home/gcx/aster_cr_repair_v2_run_20261108

GPU：0、1、5、6。使用 repair-v2 数据；历史结果保留。

## 数据

来源数：{"fit": 80, "dev": 20, "test": 40}；独立节点数：{"fit": 1623, "dev": 398, "test": 761}。
原始病例路径：{"fit": 1712, "dev": 420, "test": 806}。
MEGA 原始病例是基准提供的病例，不能解释为医院病例的自然患病或错误比例。历史使用记录见 data/history_usage.json。

各类配对数、独立节点数、来源数分别见 data/variants.json；P/C/F 独立节点仍少，不能用配对总数推断证据充足。
规则抽取与原标签不一致的记录归入抽取复核，不自动重标。新 test 评分前须完成各模型的 frozen.json。

## 阶段

- prepare：pending
- preparation：complete
- qwen：pending
- medgemma：pending
- pipeline：failed

## qwen

状态：in_progress。
主配置：cr_r16_l0.01。

| 方法 | 四类 PA 宏平均 | 路径正确数 / 总数 | 相对 Full 非劣性 |
|---|---:|---:|---|

H2*：{"dev_interference_r2": 0.188792884349823, "signal_leakage_rms": 0.3641678527570859, "true_signal_rms": 42.19247286721427, "leakage_ratio": 0.008631109484935241, "nuisance_enabled": true, "by_rank": {"4": {"dev_interference_r2": 0.05617803335189819, "signal_leakage_rms": 0.20042275727724554, "true_signal_rms": 42.19247286721427, "leakage_ratio": 0.004750201722188803, "nuisance_enabled": true}, "8": {"dev_interference_r2": 0.188792884349823, "signal_leakage_rms": 0.3641678527570859, "true_signal_rms": 42.19247286721427, "leakage_ratio": 0.008631109484935241, "nuisance_enabled": true}, "16": {"dev_interference_r2": 0.40614253282546997, "signal_leakage_rms": 0.43507787192106795, "true_signal_rms": 42.19247286721427, "leakage_ratio": 0.010311741463704203, "nuisance_enabled": true}, "full": {"dev_interference_r2": 0.5479006767272949, "signal_leakage_rms": 0.9462098208194706, "true_signal_rms": 42.19247286721427, "leakage_ratio": 0.022426033757190003, "nuisance_enabled": true}}, "criterion": "For each rank: R2 > 0 and leakage RMS / true signal RMS <= 0.1", "dev_pairs": 1258, "corrected": 0, "new_errors": 0}

H3a 错误子集证据：sufficient。

## medgemma

状态：pending。

## 方法与成本说明

ASTER-CR 的信号基、干扰基、中心化与标准化仅用 fit；干扰投影先移除信号空间。一次未编辑前向之后修正二元分数。
LoReFT 使用官方 pyreft 组件；LoFiT 保留官方两阶段流程，通过注意力输出钩子适配 Qwen3/Gemma3；LoRA 使用 PEFT。
四个训练对照均使用相同 fit 输入和三个学习率，每阶段一轮；LoFiT 额外选头阶段单独计成本。MedGemma 仅采用 Qwen 选定的学习率和方法。
LoReFT、LoFiT、LoRA 采用最接近的可实现参数规模，实际参数和全部阶段耗时见 baselines/*_training.json。
普通读出与相应 ASTER-CR 的可训练系数数量一致；bias-only 保留单参数。
ITI、SEA 的二元任务适配以 fit 中正确/错误答案补全提取特征；旧闭式参数不搜索，LEACE 与闭式方法仅在新 fit 必要重拟合。
CAD、参考融合、参考版旧 M3 计入额外参考前向；对称输出平均计两次前向。推理耗时包含实际执行，诊断与缓存准备成本另报。
所有成本记录保存在各模型 cost/、baselines/*_training.json 和 comparators/*_metrics.json。特征预缓存会计算额外节点，不能将其总耗时等同于单病例自主执行耗时。
配对统计按指南来源进行 2,000 次 bootstrap；方法差值直接重采样。非劣性仅在路径差值 95% 区间下界不低于 -0.5 个百分点时成立。

## 结果解释

未超过校准和普通读出时，收束为规约诊断与负结果；未超过 LoReFT/LoFiT 时，不主张完整方法优于这些训练方法。最终解释以实际区间和消融为准。

官方实现：[LoReFT](https://github.com/stanfordnlp/pyreft)、[LoFiT](https://github.com/fc2869/lo-fit)、[ITI](https://github.com/likenneth/honest_llama)、[SEA](https://github.com/yfqiu-nlp/sea-llm)。
