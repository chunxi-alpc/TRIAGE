# 数据与配对构造

所有数据准备与评分均在服务器运行。大文件根目录为 `/home/gcx/aster_trace_repair_20261110/data/`，本目录保留来源、病例索引、覆盖和复现代码。

来源保持为：原有fit80来源、1712病例；新增dev20来源、256病例；锁定final60来源、802病例。新来源从1105个无已记录项目使用的候选来源中，按语言和决策树节点数四分位分层抽取，种子20261110；每来源最多15个既有病例，不足时全取。没有按模型错误、分数或方法效果挑选病例。`source_selection.json` 保存后即复用，重跑不会改选来源。

每个病例含original、action_presence、action_neutral、action_swap四个有效版本，原始case_id不变，variant单独记录。A版本逐字复用 `repair_action_guides.py` 中旧repair的action_guides：保留条件、分支、叶节点ID、患者事实与定义；保护包含定义或被条件直接引用的叶文本。三种变化在同一病例的所有节点上采用同一棵修改后的树。

fit仅保存原gold路径上的46888个四版本节点任务。dev保存19776个任务，final保存63356个任务，覆盖每个已选病例整棵树的所有判断节点。gold路径上的label沿用发布标签，其余节点label=null；实际动态路径由各方法独立行走，不能沿gold路径强制执行。`data_checks.json` 记录来源分离、task唯一性、配对两端及标签连接检查。

`cases.jsonl` 以case_id和variant共同区分病例视图；`tasks_{fit,dev,final}.jsonl` 用case_id、gid和variant共同形成唯一id。`pairs.jsonl` 连接同病例同节点的原始与干扰视图。P/C/F属于节点配对诊断，不冒充新的自然病例路径。主任务文件不会追加诊断输入，后者单独放在 `tasks_diagnostic_{dev,final}.jsonl`，并包含与主任务相同id的原始左侧。

历史扫描包括原始历史记录、当前80/20/40来源输入、工作树和发布目录的reports，以及近两天新增实际评分节点/结果；未评分的inventory/catalog不算已用。旧1206个候选中101个来源已被额外评分使用，剩1105。完整文件记录和排除来源见 `history_scan.json`。这里的“未使用”限于扫描到的项目记录；不能证明Qwen预训练未接触MEGA。病例仍为MEGA既有生成病例，不是新患者数据。

## 证据与背景配对

固定在同一批已选病例中，每来源最多10个不同gold节点，dev184、final568个节点。第一版只对显式单数值条件作程序构造，范围较窄；未把无法直接解析误称为不存在可编辑证据。

增强构造逐字复用旧repair的source-only规则编译、事实提取、编辑规划及程序验证。规则阶段只看当前条件与定义，不看患者或答案；事实阶段只看所需变量与编号的原始患者句子；编辑阶段定位真实原文全部相关位置并处理受影响祖先。C/F先要求程序计算原始真值与发布标签一致，再计算修改后的真值；缺失事实、冲突、无法定位或祖先路径变化均保留失败原因。P只能修改真实可定位且与当前及祖先判断无关的背景，不插入通用末句凑数。

独立副本在 `repair_reference/`，把输出ROOT重定向到新annotations目录，并支持单个提取worker；缺失单位unit=null按原允许缺失语义规范为空字符串，不推断单位；数值编辑准备中的单条结构错误记录为该节点不可用并继续其他节点。不写旧实验ROOT。原始JSON生成、规则、观测句子定位、编辑位置与失败结果留在 `/home/gcx/aster_trace_repair_20261110/annotations/`。最终来源只为构造评价配对而进行规则/事实标注，不输出方法ell或最终判断，也不用于选修复方法。最终方法评分须在开发选择冻结后启动。

增强诊断完成后才写 `diagnostic_dev_ready.json` 与 `diagnostic_final_ready.json`，供pipeline继续；构建过程状态另存status文件。增强构造已完成：dev为C14/F4/P1对，覆盖6/2/1个来源；final为C29/F34/P15对，覆盖14/8/8个来源。配对共享来源和节点，不能把配对数当独立来源数。实际覆盖与未构造原因见 `diagnostic_coverage_{dev,final}.json`。

## 复现

在仓库根目录、gcx261014环境中执行：

```bash
python reports/aster_cancel_story_20261109/meeting/trace_repair_followup/data.py --fit
python reports/aster_cancel_story_20261109/meeting/trace_repair_followup/data.py --new
python reports/aster_cancel_story_20261109/meeting/trace_repair_followup/data/enhance_diagnostics.py --prepare
CUDA_VISIBLE_DEVICES=6 python reports/aster_cancel_story_20261109/meeting/trace_repair_followup/data/enhance_diagnostics.py --run --batch 6
python reports/aster_cancel_story_20261109/meeting/trace_repair_followup/data/data_checks.py
```

已有标注完成时，`enhance_diagnostics.py --run` 直接从缓存重新build/export，不再加载生成模型；也可使用 `--export`。普通 `data.py` 全流程会优先复用已完成增强标注。`data.py --diagnostics` 明确只运行首版程序数值构造，供诊断该阶段解析能力；完整复算应使用增强缓存。导出时通过原始group_id连接 `annotations/data/nodes.jsonl` 明确补齐case_id/gid/source_id/split，保留程序生成的左右真值标签；不从case_id字符串推断身份。缓存导出及连接检查结果见 `diagnostic_export_resume.json`、`pair_identity_normalization.json` 和 `data_checks.json`。

旧来源差异与病例观察见 `../trace_validation/exp3/README.zh.md` 和 `../trace_validation/exp3/full/source_metrics.csv`；这些旧结果仅用于提出诊断问题，不参与新来源抽样。
