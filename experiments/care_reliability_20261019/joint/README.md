# 完整上下文联合实验

本目录实现同一完整病例上下文下的结构来源与干预方式比较。**当前 CPU 核验不构成模型效果实验。** 实际模型执行状态保存于运行目录；没有 PyTorch、固定模型权重或已验证子空间资产时，相关实验明确标记为阻塞，不填入模型指标。

本集为模型前向执行前冻结的新合成机制测试。其 schema 和措辞曾用于开发阶段回归修复，包括亲属主体识别；因此，它不是对全部方法开发保持盲法的独立确证，也不是临床验证。确认家族和全部表面变体保持固定，不根据预测或干预效果删选。

## 比较设计

| 结构来源 | 仅结构化提示 | 直接有符号注意力分数偏置 | 学习子空间与查询校准 |
| --- | --- | --- | --- |
| V6 预测 | 保留全部规则与预测 T/F/U | 严格沿用 V6 实际门控 | 同门控与已验证 B4 资产 |
| V7 预测 | 保留全部规则与预测 T/F/U | 严格沿用 V7 实际门控 | 同门控与已验证 B4 资产 |
| 参考结构，仅正向门控 | 诊断 | 参考 T/F/U 保留，仅 T 开门 | 同正向策略的诊断 |
| 参考结构，双符号门控 | 诊断 | 已知 T/F 均可开门 | 双符号机制诊断 |

另运行不添加适用性表的完整上下文无干预基线。对于给定病例和结构来源，三种干预方式使用**字节一致的完整提示**，并保存提示摘要。不依据结构预测删去不适用动作；不调用旧流水线中的动作过滤步骤。

参考结构必须通过 `--include-reference-diagnostic` 显式启用。参考正向门控用于区分结构误差与门控策略的影响。参考双符号结果不能直接归因于预测结构准确性的提高。V6/V7 原有规则门控若不允许负向干预，本实验不会擅自开放。未校准的二值证据门控保存为 `gate_weight`，不会冒充校准概率；没有置信度时 `confidence` 保持空值。

头位置、分数扰动强度、查询校准和分组查询注意力检查复用相邻 `module2/protocol.json` 的冻结配置。新确认集不用于选择头、秩、强度或随机种子。学习子空间必须来自有训练与选择来源、模型版本和文件摘要的 B4 资产；缺失时该实验臂单独阻塞。

对多个互不重叠的规则动作片段，在同一层保存原始查询和键，逐片段计算校准编辑，再一次性修改该层键。局部符号检查覆盖最后提示位置的四个分组查询头。每片段按其门控权重匹配扰动；另报告合并后的实际均方根幅度。**这不是证据预算实现，也不保证后续生成过程的方向。** 真实模型注意力钩子的有效性必须在模型环境核验。

## 指标

- 病例准确率、适用与不适用条件准确率，以及二者的调和平均 H-Acc。
- 成对准确率：同一最小对比中的两例均答对。主要分析只用确认集的标准表述对；全部表面变体另报。
- 不适用目标动作误用率。缺失输出会使误用率无法完全识别，此时报告范围和缺失数，不能将缺失当作“未误用”。
- 保持率：完整上下文基线原本正确的病例中，干预后仍正确的比例。它不是“稳定闭卷一致病例”的临床保持率。
- 门控覆盖与错误状态风险：分母始终为全部期望病例乘全部规则。缺失、解析失败、无效状态分别保存，不从分母中删除；评分可以读取参考状态，预测器不能读取。
- 严格解析失败包括全部非 accepted 终态、生成截断和技术错误。数值比较器能够独立解决某个原子时，其结果仍按实际实现保留，但解析失败计数不会被抹去。

成对区间按完整家族重采样。预设的五项比较给出探索性的 95% 区间，未做多重比较校正，不据此声称确证性显著结果。零宽区间也不意味着总体效应严格为零。

## 不加载模型的预检

```bash
python experiments/care_reliability_20261019/joint/run_joint.py \
  --output /tmp/care_joint_preflight \
  --sources reference reference_positive v6 v7 \
  --include-reference-diagnostic --preflight-only
```

## 真实模型运行

```bash
python experiments/care_reliability_20261019/joint/run_joint.py \
  --model /path/to/Qwen3-4B-Base \
  --module1-base-model /path/to/phi-4 \
  --module1-adapter-model /path/to/trialmatchai-adapter \
  --module1-evidence-model /path/to/biolinkbert-mednli \
  --learned-assets experiments/care_reliability_20261019/module2/frozen_learned_assets.json \
  --device cuda:0 --module1-device cuda:0 \
  --output /path/to/new_isolated_joint_run \
  --sources reference reference_positive v6 v7 --include-reference-diagnostic
```

本地固定权重不可用时不自动下载其他模型、不改变模型版本。已有预测日志不可覆盖。各结构原始输出、规范化结构、完整提示摘要、实际模型概率、干预局部审计、阻塞原因和执行状态分别保存。`metrics.json` 仅在实际模型运行阶段后读取真实日志生成。

模型初始化核对原 4B manifest 的权重摘要及解析后的完整关键配置，包括 RoPE、全部注意力层类型、滑动窗口、分组查询维度和归一化；加载前后均复用模块二兼容检查器。版本、配置审计与分词器摘要保留在模型来源记录中。

每个实际病例与结构来源的完整提示均运行无编辑结构提示和空目标 `signed_bias` 两次真实前向，核对任务所用的四个候选 logits 和 A/B/C/D 概率是否逐值相同，只保存最大差异和检查状态。结构提示输出缓存为该实验臂答案。`zero_checks.jsonl` 与 `zero_checks.json` 记录所有期望病例/来源的覆盖、差异、错误及未运行原因。零控制不一致时，该病例的直接偏置臂标记失败，涉及该失败的汇总和对比不输出效果值；不能把掩码维度或内核路径的变化解释为非零干预效果。

学习资产清单需含：`artifact_type=learned_applicability_subspace`、`method=B4`、`source_repository_commit`、固定 `model_revision`、非空 `training_family_ids`、`selection_family_ids`、空 `confirmation_family_ids_used`；`folds` 的每一项含 `operator_path`、`operator_sha256`、可选 `operator_key` 和冻结的 `location`。路径相对于资产清单。载入时检查新确认家族未参与拟合或选择、算子摘要、形状、对称性和幂等性，不接受随机投影冒充学习资产。

复算已有真实日志：

```bash
python experiments/care_reliability_20261019/joint/evaluate.py \
  --cases experiments/care_reliability_20261019/data/mechanism_cases.jsonl \
  --predictions /path/to/joint_run/predictions.jsonl \
  --output /path/to/joint_run/recomputed_metrics.json
```

## 在原服务器分环境执行

现场核验发现原模块一实际使用 /home/gcx/n/.venv（transformers 4.57.1，
bitsandbytes 0.50.2），原 Qwen 使用 /home/gcx/gcx261014（transformers 4.57.6）。
新入口 run_server.py 默认先在 GPU 0、1、2、3、6、7 以原 Phi-4 环境生成结构，再在 GPU 0
以原 Qwen 环境完成矩阵；不安装依赖，不合并两个环境。

激活原环境后运行 python experiments/care_reliability_20261019/joint/run_server.py
--output /home/gcx/care_reliability_runs/joint_formal_new。
增加 --training-preflight 仅运行既定训练集前两例。确认集始终完整运行，不能限额。
结构阶段按病例交错分片（默认六片，各 32 例），复用完全相同 patient、criterion text、criterion kind
的真实生成结果。V6/V7 分别执行原数值、证据和门控逻辑，NLI 仍实际调用。
batch=8 和 max_new_tokens=2048 复用原 TrialMatchAI 接口，批处理可能改变浮点舍入，
不声称与逐条执行字节相同。两个版本从同一请求缓存读取以控制生成差异。

每片保存 request_manifest.json、原始 trialmatch_generations.jsonl、原子提交的
generation_batches、两版本原始图、精确输入摘要、源码摘要和模型全权重摘要。
immutable_provenance.json 不可变；--resume 先核对数据、源码、模型字节和软件版本，
拒绝后不会替换原记录。恢复保持原 batch 成员，仅复用完整批；旧 journal 保留。
COMPLETE.json 标记完整结束。Qwen 的 --structure-runs 在加载模型前核对全部
期望病例、原始图、规范化图、输入、源码、生成日志与不可变来源，任一缺失拒绝执行。

控制器只等待自己启动的子进程退出，不轮询 GPU；结构成功后自动运行 Qwen。
模块一子进程如失败可按其原命令增加 --resume 恢复，完成后以 --structure-runs
显式提交全部完成目录给 run_joint.py。已完成目录禁止覆盖或继续。

