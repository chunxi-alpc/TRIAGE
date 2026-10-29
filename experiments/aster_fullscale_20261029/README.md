# ASTER 全量实验工具

先读 [实验计划](PLAN.zh.md) 与冻结协议 protocol.json。此目录提供按指南树分组的拆分工具和单元测试，不会自动启动模型推理。

## 1. 检查数据字段

先检查私有服务器上的 JSONL schema，确认决策树/来源指南 ID 的实际字段名。不要根据本示例猜字段名，也不要把病例 ID 当作分组字段。原始病例、指南文本和模型长输出应留在服务器私有目录。

## 2. 建立 MEGA-CDP train_val 分组清单

对决策树文件创建 manifest；将下例中的 TREE_ID_FIELD 替换成经检查的真实字段：

~~~bash
python experiments/aster_fullscale_20261029/scripts/prepare_group_splits.py create \
  --input /absolute/private/path/train_val/decision_trees.jsonl \
  --group-key TREE_ID_FIELD \
  --dataset MEGA-CDP-train_val \
  --fractions fit=0.7,select=0.15,confirm=0.15 \
  --seed 20261029 \
  --output /absolute/private/path/aster_fullscale/mega_split_manifest.json
~~~

把同一分组映射应用到病例文件；病例文件必须有可映射到树的相同分组键：

~~~bash
python experiments/aster_fullscale_20261029/scripts/prepare_group_splits.py apply \
  --input /absolute/private/path/train_val/vignette_cases.jsonl \
  --group-key TREE_ID_FIELD \
  --manifest /absolute/private/path/aster_fullscale/mega_split_manifest.json \
  --output /absolute/private/path/aster_fullscale/mega_cases
~~~

若病例的分组字段与树文件字段不同，先生成一个只含 ID 到树 ID 的映射表并审计覆盖率；不能通过病例文本猜测归属。apply 报告中的 unknown_group_records 必须为 0，否则阻断实验。

## 3. MedGUIDE

用来源树/疾病树 ID 创建 60/20/20 manifest。由于 MedGUIDE 目前使用原始 train split，留出集只能称为按来源指南分组的内部外部验证，不能称为官方测试集。跨数据集来源重叠审计通过前，不得宣称完全独立的外部验证。

## 4. 运行测试

~~~bash
python -m unittest discover -s experiments/aster_fullscale_20261029/tests -v
~~~

## 状态

- [x] 固定数据分工、分母规则、对照和统计单位。
- [x] 实现确定性指南级拆分与应用工具。
- [x] 添加拆分工具单元测试。
- [ ] 核验服务器原始数据字段并生成真实 manifest。
- [ ] 接入最小对构造、ASTER 前向与官方路径评分。
- [ ] 在参数冻结后运行 MEGA-CDP 官方测试和 MedGUIDE 留出验证。

当前没有启动全量 GPU 推理，不能把规划或接口测试写成方法有效性结果。
