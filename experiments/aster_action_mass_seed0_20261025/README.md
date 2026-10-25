# ASTER 动作级质量守恒：单种子开发实验

基于 PR #31 `cf3298d054d2681911c4c4925eee0bd922c17036`；已读取 PR #30 最新 head `1dae58a14a8242c620e8c5810be432ae198949cf` 的实验说明与引擎。保留两者及全部旧记录。

## 范围与资源

只使用原划分开发集 200 病例 × NC/AC/IC/SW，seed=0，共800条/模型。最初试跑 Qwen3-8B 使用115服务器 GPU0，Llama-3.1-8B 使用GPU1；正式实验两模型改用GPU0顺序运行。两卡均为A100 80GB。启动前预计每卡20–35GiB、2–6小时；4条短试跑实际约43–47秒、15.5–15.8GiB，随后GPU1出现功率限制，同样4条变为177秒；Llama换到GPU0的1条复核16.25秒（含模型加载和单测）。正式两模型GPU0顺序执行，预计总计约4小时，峰值预算20–35GiB，不占用GPU1。货币单价未知，成本报告GPU小时。

私有数据从既有 cases.jsonl 只提取 dev/seed0/四卷，另存 `/home/gcx/aster_action_mass_seed0_20261025/dev.jsonl`。推理与选参程序只打开此文件。既有测试集不运行、不用于选参；GitHub PR 元数据含旧测试摘要，但未用于本次参数选择。所有强度与门控候选在正式开发集运行前固定。

## 算子与公平对照

逐条件三分类判断只接受患者与条件，不含处置、选项或答案。沿用 PR31 的条件判断提示，支持概率归一化为 p(r)。同动作的所有规则概率相加得到 S(a)。所有方法共享每条记录的同一 p(r)、S(a)、规则、病例与选项。完全相同的条件输入只在本次运行内复用，并公开缓存键。

对每层、每个头、每个查询，令动作组原注意力为 M(a)，动作并集总质量为 M。新质量为 `(1-alpha)*M(a) + alpha*M*S_visible(a)`；S 仅在当前因果掩码允许访问的动作组中归一化。组内相对权重不变；非动作词元及孤儿选项注意力不变；总动作质量与整行总和不变。下溢到零的组使用可访问词元均匀分布；不存在可访问动作质量时不增加质量。保留因果与显式掩码，按256个查询分块，不截断上下文；GQA显式重复KV头，不复用跨配置KV缓存。

比较原答案、直接动作映射、同S输出概率融合（0.25/0.5/0.75）、动作裁剪、原分支软/硬门控、动作分数偏置、质量守恒、零强度数值对照。动作裁剪保留所有条件和原始选项，只把非胜出动作文本替换为占位符；同动作多条规则一起保留。直接映射及融合将S在原选项中归一化；不在选项中的动作质量单列报告。孤儿选项的S为零，融合仍保留原模型概率。原分支门控按旧实现的规则级max概率处理，是明确的历史对照，不假称动作级求和。

质量守恒强度0.25/0.5/1，全层或后半层；偏置强度1/2，同两种层范围；原分支软门控强度2、硬门控，两种层范围。另对质量守恒0.5按最大S>=0.6启用，其余回退原答案；此门控由同一预测离线组合，不增加模型推理。共23项固定配置。所有候选、全部病例均报告。按树留一选择质量守恒与简单基线，另列开发集同集最优；均不代表独立测试表现。

## 运行与复现

代码与测试在服务器运行。新工作目录：`/home/gcx/TRIAGE-worktrees/aster-action-mass-seed0-20261025`。

```bash
source /home/gcx/gcx261014/bin/activate
cd /home/gcx/TRIAGE-worktrees/aster-action-mass-seed0-20261025
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 python -m unittest discover -s experiments/aster_action_mass_seed0_20261025 -p 'test_*.py'
bash experiments/aster_action_mass_seed0_20261025/launch.sh
```

输出路径已存在时拒绝覆盖。复跑需为run.py传入新的 `--root`（包含同一dev.jsonl），分别设置CUDA_VISIBLE_DEVICES=0及--name qwen/llama，顺序执行。报告命令：

```bash
python experiments/aster_action_mass_seed0_20261025/report.py --root /home/gcx/aster_action_mass_seed0_20261025 --output experiments/aster_action_mass_seed0_20261025/results/final_v1
python experiments/aster_action_mass_seed0_20261025/verify.py --root /home/gcx/aster_action_mass_seed0_20261025 --report experiments/aster_action_mass_seed0_20261025/results/final_v1
```

报告需两模型800/800完整矩阵，缺失时不能发布完整结论。完整私有日志及失败试跑留在服务器。公开输出只含数值、已有病例ID、预测、指标、配置与指纹，不含病历、规则原文、权重或凭据。

## 验证与运行状态

CPU测试包括质量守恒、组质量目标、因果/显式掩码、全遮挡行、多规则同动作、孤儿选项、零信号、GQA与不同查询分块一致性、三层间接传播。GPU验证包括BF16与SDPA数值对照、4097词元上下文、真实模型全部层的非动作K盲判不变性和无遮挡阳性对照。零强度模型输出另列完整配置，以便发现数值差异。

首次短试跑在GPU数值单测发现掩码dtype不匹配，未进入病例推理；修复后两模型完成4条真实推理。失败目录和日志未删除。完整结果见完成后生成的 `results/final_v1/REPORT.zh.md`；该文件不存在时表示全量实验尚未完成。

正式launcher在完成后自动汇总、独立复算并提交推送本分支；失败时只发布失败状态，不发布完整结果。状态见results/run_state.json，只有complete_verified才表示全量完成且复算通过。稳定运行后不持续监控。
