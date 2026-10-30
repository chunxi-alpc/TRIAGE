# A：可运行的方向特异性诊断

所有命令在服务器 `gcx@10.29.128.115` 执行。先进入 `/home/gcx/gcx261014` 环境与本独立工作树。协议先于新增模型运行写入 `protocol.md`；不修改旧主稿/临床目录任何文件。

## 输入

- 默认基准路径：`/home/gcx/aster_manuscript_20261030/clinical01`，可通过`--base`更改。
- `analysis/expert.json`：`aster.manuscript.expert.v1`主稿配对键差分、谱基投影有向均值专家；保持冻结rank8、头集合。PR41查询谱投影产物不能混入。
- `review/accepted_pairs.jsonl`：已接受规则验证年龄三态对，`pair_id/source_id/split/guideline/target_rule_id/states/clinical_verification`。不表示医生金标。
- `collect0/status.json`、`collect1/status.json`与其`.pt`：已有真实Qwen3-8B的`offline[state][(layer,kv_head)]`张量，形状`目标词元数×128`；只读取status.complete记录，并与accepted pair_id交集。
- `analysis/statistics.json`：旧全候选头训练/留出和单次分半统计，仅用于报告对照，不选参。
- GPU额外需要固定本地`models/Qwen3-8B-2602814`和transformers4.57.6、torch2.8.0。
- 新候选构建只读取`source_clusters.json`和原MEGA-CDP train_val。官方test只使用已有来源隔离元数据，不读取test病例输出/标签。

## CPU与GPU命令

```bash
source /home/gcx/gcx261014/bin/activate
cd /home/gcx/TRIAGE-worktrees/aster-abc-diagnostics-20261030
python -m unittest experiments.aster_abc_diagnostics_20261030.a.test_a -v
python -m experiments.aster_abc_diagnostics_20261030.a.cpu --output /home/gcx/aster_abc_diagnostics_20261030/a/cpu_reproduce
python -m experiments.aster_abc_diagnostics_20261030.a.candidates --output /home/gcx/aster_abc_diagnostics_20261030/a/candidates_reproduce
CUDA_VISIBLE_DEVICES=0 python -m experiments.aster_abc_diagnostics_20261030.a.gpu_controls --output /home/gcx/aster_abc_diagnostics_20261030/a/gpu_reproduce
```

所有输出目录必须新建；上面是复现说明，不应在已有GPU诊断运行时重复启动。正式启动前应报告资源；本轮由协调代理负责GPU。单卡预计55–65GB、3–15分钟，4个既有年龄确认对/2个来源、6变体×3状态=72次前向，无训练/生成。

## 输出

- CPU：`summary.json`实际数目与范围；`per_head.json`每头能量曲线、谱均值余弦、20次来源分半、32固定随机方向、64随机子空间、来源bootstrap；`source_projection.json`每病例/头投影。
- GPU：每对一个`.pt`保留6变体三态键（只选4头）；`records.jsonl`逐对逐头的决定事实、无关事实和变体配对差；`summary.json`来源簇描述统计、技术失败、实际用时/峰值显存；`status.json`运行状态。
- 候选：`candidate_pairs.jsonl`6个非年龄候选，标签均null；`annotation_template.jsonl`两位独立裁定模板；`source_manifest.json`来源隔离和排除清单。数值替换可能残留定性表述，因此需要临床人员修正文案与裁定。

GPU只采集4个已有年龄确认对，不构成新的确认试验。所有2来源区间一律作为描述，不据其宣称总体特异性通过。记录号数字等长；改写、前置说明仍可能改变上游位置效应，输出保留词元数。处置互换改变目标词元内容/个数，只比较逐词元均值并明确限制。

## 人工裁定导入

复制模板给独立裁定者后，在单独文件中填写`reviewer_id/reviewer_type=human_clinician/role/labels/ancestors_reachable/evidence/corrected_patient_texts`。a/b必须一真一假，missing必须unknown。两位意见不一致时需要第三位独立adjudicator；没有人工裁定时不生成金标。

```bash
python -m experiments.aster_abc_diagnostics_20261030.a.candidates --annotations /path/to/completed_annotations.jsonl --output /home/gcx/aster_abc_diagnostics_20261030/a/candidates_adjudicated
```

输出`adjudicated_pairs.jsonl`保留所有完整已裁定三态记录；祖先不可达者保留但`eligible_for_direction=false`、`exclusion_reason=ancestor_not_reachable`，单列于`annotation_status.json`，不能计入可用三态金标。尚未裁定或契约不满足者保留为pending原因。模型辅助意见不能当医生签名。

## 裁定后非年龄实验入口（当前尚无可用人工标签，未运行）

旧临床采集器强制年龄谓词。新`adjudicated.py`使用实际人工裁定导出的主稿契约，不复用年龄验证器；导出时保留祖先不可达排除，不能凭脚本制造标签。

```bash
python -m experiments.aster_abc_diagnostics_20261030.a.adjudicated prepare --candidates /home/gcx/aster_abc_diagnostics_20261030/a/candidates/candidate_pairs.jsonl --adjudications /home/gcx/aster_abc_diagnostics_20261030/a/candidates_adjudicated/adjudicated_pairs.jsonl --output /home/gcx/aster_abc_diagnostics_20261030/a/nonage_inputs
CUDA_VISIBLE_DEVICES=0 python -m experiments.aster_abc_diagnostics_20261030.a.adjudicated collect --pairs /home/gcx/aster_abc_diagnostics_20261030/a/nonage_inputs/pairs.jsonl --split confirm --output /home/gcx/aster_abc_diagnostics_20261030/a/nonage_confirm
```

该采集固定原年龄专家/4头，检验在已裁定非年龄决定事实上的迁移，不重新选参。来源内两个互补规则一起报告；未通过主稿词元对齐者记录技术失败，不能偷偷改标签或丢分母。正式启动前再按实际可用临床样本规模报告GPU/显存/时间。

## 已有GPU产物的CPU词元分解与报告

```bash
python -m experiments.aster_abc_diagnostics_20261030.a.token_parts --gpu /home/gcx/aster_abc_diagnostics_20261030/a/gpu --output /home/gcx/aster_abc_diagnostics_20261030/a/gpu_token_parts_reproduce
python -m experiments.aster_abc_diagnostics_20261030.a.render_gpu_report --gpu /home/gcx/aster_abc_diagnostics_20261030/a/gpu --copy-to /home/gcx/aster_abc_diagnostics_20261030/a/audit.md
```

词元分解是对现有提取键的CPU事后定位，将规则编号与处置文字分开；未重跑GPU、未挑选头或增益。默认报告渲染器在相邻`gpu_token_parts/summary.json`存在时附上分解结果。它有助定位处置文本组成和上下文效应，不能消除上下文混杂。
