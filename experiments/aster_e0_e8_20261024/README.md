## FlexAttention 工程验证

`flex_engine.py` 提供独立的 `FlexEngine`，用 `score_mod` 实现软分数调整，用 `create_block_mask`/`block_mask` 实现因果、盲判和硬门控访问控制；GQA使用原生 `enable_gqa`，不展开KV头。当前范围为原Engine使用的单条、无填充、完整序列评分；未接入KV缓存或批量填充解码。不会替换正在运行的Engine。

`run_flex_validation.sh` 只使用GPU6，在实验目录提取Python开发头文件，未修改系统环境。`validate_flex.py` 比较FP32/BF16、GQA和129词元跨块边界、分数/访问控制、三层间接传播，以及两8B模型的4种模式。模型接口对照每模型仅1条开发样本，不是性能实验；实际差异保留在结果文件中。SEKA和头选择尚未加入。

## 新增：seed=0 开发集改进实验

`launch_improvement.sh` 在空闲GPU6顺序运行两个8B模型。`improve_branch.py` 添加不含处置的逐条条件三分类判断、全层门控、跨树选择融合权重和判断一致才启用干预的对照。四种分支来源均比较直接映射、裁剪、全层软/硬门控；共22种答案配置，部分复用此前同配置结果。只用200例开发集四卷 seed=0，不更改原测试锁定配置。结果位于 `results/improvement_seed0_v1/`，每个模型结束后自动汇总并推送本PR。

# ASTER 后续真实实验（2026-10-24）

本分支继续 PR #30。下面的历史说明保留 E0/E1 原结果；关于 E1 阻挡扩跑的要求已由用户的新执行要求替代。

- `engine.py` 不读取金标；裁剪/写提示必须显式传入 selected。条件、处置、选项分别标记；盲判的相位/层与门控的相位/层独立。
- `continue_run.py` 在开发集200例四卷 seed=0比较12种 E2配置、最小E3和预测方法参数；写入 locked.json 后运行 E5 的1000例四卷三种子及既有CS补充。
- ASTER-full 的辅助证据为 GPU6 上 Qwen3-1.7B 隐藏处置/选项得到的分支分布，与本模型盲判等权平均；并非人工证据或金标。
- `launch_continuation.sh` 仅启动 GPU0 Qwen3-8B、GPU1 Llama3.1-8B、GPU6 Qwen3-1.7B。单模型持续加载，配置独立输出，逐例断点续跑；每50病例构成顺序处理的一组，配置文件内按病例和种子唯一保存。
- `continuation_report.py` 导出白名单数值记录、按树配对区间、图表、实际显存/GPU小时、成本和失败元数据。全部进程结束后自动导出并提交本PR分支；中途结果标为运行中。
- 主结果在 `results/continuation_v2/`。12项最小工程测试通过；没有运行旧20例2百分点性能验收来阻挡后续实验。

复现入口：`bash experiments/aster_e0_e8_20261024/launch_continuation.sh`。服务器私有输出：`/home/gcx/aster_e0_e8_20261024/private/continuation_v2`。运行锁防止重复占卡；不要在进程运行期间改变engine或runner。

---

## 历史记录（以下是此前执行状态，不代表后续实验当前状态）

# ASTER E0–E8：独立执行版本

本目录保留2026-10-24实际执行的E0构建、E1开发集门槛测试，以及未启动的E2核心入口。不能将这些工程结果写成E0–E8完成。

工作目录：`/home/gcx/TRIAGE-worktrees/aster-e0-e8-20261024`。
环境：`/home/gcx/gcx261014`。仅允许物理GPU 0、1、6；每个进程显式只看一张卡。
私有输入、完整日志、标注表及SEKA投影：`/home/gcx/aster_e0_e8_20261024/private`。

## 数据与协议

`results/preregistered.json`在首次GPU推理前写入。`results/split.json`固定10个开发树条目，其余测试条目；实际选择200/1000病例。MedGUIDE disease条目作为树标识，不宣称每条都是独立完整NCCN树。六卷沿用同一病例、同一种子的选项集合，三个种子分别打乱。选项集合为受控扩展版本，与原始MCQA有区别。随机打乱不保证有限样本字母数量精确相等。

CS构造复用历史分支`codex/medguide-expand-cs-20261023`的阈值解析代码，复制为`threshold_source.py`保留原函数。已验证的CS数量不足400；阈值解析只覆盖部分显式数值/时间表达式，不能声称所有语义类型穷尽。反事实标签仅是实验标签。

`prepare.py`产生私有数据和逐条排除原因，公开目录只保存选择、哈希、数量和划分。完整原子标注、闭卷先验、旧800病例复现、人工κ仍待完成。人工审核单独进行，用户明确授权继续工程验证。

## E1实现

`engine.py`在每一层注意力计算前施加4D加性遮挡，使用PyTorch SDPA；覆盖全部判断阶段的非处置查询，处置段和整条选项行均标记ACT。多token规则编号使用完整串对数概率。答案前缀使用oracle规则，仅用于T3通道比较；不将此分支标签计作预测准确率。对数门控依据规则概率，硬门控使用-1e4。无匹配规则的孤儿选项不强行赋予规则概率。未实现主实验的预测分支作答链。

不复用KV缓存：不同干预下不能直接共享。预填充后端为已核实的Llama/Qwen注意力接口；MedGemma不在此接口验证范围。`strict_diagnostic`额外对所有查询切断非选中动作，仅用于排查间接泄漏，不代替正式阶段门控。T1检查追加病历事实后共同前缀各层K不变，比完整的逐事实最小对检验范围更窄。

`green.py`调用本机官方SEKA实现的attach_projection，在Llama k_proj输出、RoPE之前编辑key；Qwen官方会选k_norm。投影必须与模型匹配，不提供随机替代。`build_green.sh`在GPU6用100对官方合成样本构建Llama末10层投影；并非SEKA论文数值复现。绿笔接口烟雾测试不等于组10效果实验。

## 复现

以下命令在115服务器运行。输出目录若已存在会拒绝覆盖；复跑请换新名字。

```bash
source /home/gcx/gcx261014/bin/activate
cd /home/gcx/TRIAGE-worktrees/aster-e0-e8-20261024
python experiments/aster_e0_e8_20261024/prepare.py
python experiments/aster_e0_e8_20261024/annotation.py
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s experiments/aster_e0_e8_20261024 -p 'test_*.py'
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python experiments/aster_e0_e8_20261024/run_e1.py --model /home/gcx/TRIAGE/models/modelscope/Meta-Llama-3.1-8B-Instruct --name llama_e1_reproduction --limit 20
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 python experiments/aster_e0_e8_20261024/run_e1.py --model /home/gcx/TRIAGE/models/modelscope/Qwen3-8B --name qwen_e1_reproduction --limit 20
python experiments/aster_e0_e8_20261024/analyze.py
```

`analyze.py`只读取本轮固定命名输出；复跑输出需要显式修改收集名单。所有结果按模型分别记录，逐树聚类bootstrap 10,000次。此轮仅seed=0工程试验，不能报告三种子均值±标准差。测试集尚未推理。

`run_e2.py`只提供组3、4、9开发集扫参入口，必须读入通过的E1 summary。其他E2组、注意力质量、正式测试及E3–E8未实现或未执行，详见中文报告。不要绕过E1开关直接扩跑。

## 标注

参见同目录`标注说明.txt`。服务器`annotation_A.csv`和`annotation_B.csv`分别交给两位标注者，150个不同病例、六卷分层抽样；表中不含模型预测。独立填写后运行`annotation.py --a ... --b ... --out ...`。逐项计算κ，分歧另存裁定，不改独立原表。
