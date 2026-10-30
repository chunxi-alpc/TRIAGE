# PR42 A/B/C 独立诊断
所有运行在服务器 gcx@10.29.128.115:22。本目录基于PR42最新4fbc0c5，独立分支codex/aster-abc-diagnostics-20261030。原PR42冻结配置、运行与结果只读。主稿方法与PR41查询谱投影不混用。

先读protocol_D.md、hypothesis_matrix.csv及a/b/c各自协议；具体实验规模/种子以启动前登记的子协议、config及运行manifest为准。execution_amendment.md记录实际资源与协议间的差异。报告见REPORT.zh.md及a/audit.md、b/report.md、c/AUDIT.md。

## 运行CPU全流程
```bash
ssh -p 22 gcx@10.29.128.115
cd /home/gcx/TRIAGE-worktrees/aster-abc-diagnostics-20261030
bash experiments/aster_abc_diagnostics_20261030/run_cpu.sh /home/gcx/aster_abc_reproduction_new
```
必须给新的独立输出目录。run_cpu读取config.json，依次运行回归/诊断测试、来源审计、真实张量方向分析、非年龄候选、已保存查询分析、路径误差/覆盖风险复算。每步日志保存在新输出目录，失败立即退出，不改官方结果。无需GPU。

## GPU运行
启动前按用户要求汇报卡号、显存、规模、时长；本轮已汇报且获授权。每卡独立模型，只使用0/1，不做持续监控。以下命令均需新输出目录：
```bash
source /home/gcx/gcx261014/bin/activate
source /home/gcx/aster_manuscript_20261030/venv-overlay/bin/activate
cd /home/gcx/TRIAGE-worktrees/aster-abc-diagnostics-20261030
# A：72次前向，年龄控制
CUDA_VISIBLE_DEVICES=0 python -m experiments.aster_abc_diagnostics_20261030.a.gpu_controls --output /home/gcx/aster_abc_A_new
# B：60记录，金标根、同词元轨迹
CUDA_VISIBLE_DEVICES=0 python -m experiments.aster_abc_diagnostics_20261030.b.run_gpu --output /home/gcx/aster_abc_B_gold_new --route gold_root --max-sources 8 --steps 17
# B预测根对照（与上行串行）
CUDA_VISIBLE_DEVICES=0 python -m experiments.aster_abc_diagnostics_20261030.b.run_gpu --output /home/gcx/aster_abc_B_predicted_new --route predicted_root --max-sources 8 --steps 17
# C：43病例×2方法×2路由=172，另2次零编辑，合计174生成
CUDA_VISIBLE_DEVICES=1 python -m experiments.aster_abc_diagnostics_20261030.c.gpu_oracle --run /home/gcx/aster_manuscript_20261030/clinical01 --cpu-audit /home/gcx/aster_abc_diagnostics_20261030/c/cpu01 --output /home/gcx/aster_abc_C_new
```
B的--max-sources=8是上限；实际confirm只有2来源，未用训练来源补齐。B现5随机种子，最小经验p=1/6；A有32方向/20分半/64随机子空间。统计单位始终是来源簇。

B汇总：
```bash
python -m experiments.aster_abc_diagnostics_20261030.b.summarize_replay --run /home/gcx/aster_abc_B_gold_new --output /home/gcx/aster_abc_B_gold_summary_new
python -m experiments.aster_abc_diagnostics_20261030.b.audit_replay --run /home/gcx/aster_abc_B_gold_new --output /home/gcx/aster_abc_B_gold_audit_new
```

## 输入契约与环境
- 只读数据根：clinical01；需要review/accepted_pairs.jsonl、collect0/collect1张量、analysis/expert.json、frozen.json、classification和confirm记录；C还需要cases/confirm.jsonl与confirm0/confirm1/units。
- 固定权重：clinical01/models/Qwen3-8B-2602814，Qwen/Qwen3-8B修订26028140be3ee69b82b1d1450179ab71bb1121b9。以运行reproduction_manifest.json为准：Python3.12.3、PyTorch2.8.0、Transformers4.57.6、BF16/eager；不安装或替换公共环境。
- 预留单卡55–65GB；B实测33.97GB/65.42秒。A/C实测见最终报告。CPU线程4，真实张量分析不训练官方专家。
- 临床原文/候选/逐病例输出保留服务器；本仓库仅代码、配置和允许公开的汇总。缺少私有数据时不能声称端到端已复现。

## 非年龄候选裁定
当前候选6条，来源隔离，标签均null；见服务器a/candidates目录。需独立人工临床裁定，不能由模型预测填充。a/README.md及candidates.py说明输入契约。
```bash
python -m experiments.aster_abc_diagnostics_20261030.a.candidates --output /home/gcx/aster_abc_annotations_new --annotations /absolute/path/clinician_annotations.jsonl
```
祖先不可达记录保留但不纳入方向样本。尚无医生裁定三态临床对，不能称通用临床语义金标。

## 官方测试恢复
用户最新授权后曾暂停0/1/5/6的旧官方作业，暂停时498完整病例；新诊断结束后已恢复缺失单元，仍不是全量完成。当前监督器状态见服务器official_resume_20261031/status.json。official_test_launch.json属于旧启动器，可能仍显示running，应以official_pause.json及实际进程为准。所有已有方法/评分成功与失败不重跑。
若再次需要恢复，四卡空闲后用resume_official.sh及新的日志输出目录恢复缺失单元，沿用原冻结参数与原执行器；它不会重新启动旧自动发布进程，监督器仅等待进程退出并在全队列完成后运行原report模块汇总，不据中途成绩改参数。完整7,458病例×13方法及官方多次评审仍预计需数天，本轮小诊断不替代它。
