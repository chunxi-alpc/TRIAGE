# v2 修复、CPU 验证与预登记

v1 的 N1 GPU 输出不是有效科学结果，不能只修复选项计分后继续。额外确认的问题：N1/N2 提示缺少选项语义；N2 没有把 synthetic prior 提供给接收者；所谓 direct_bias 只修改输出分数；N3 追问属性判别把无关未知也算作可询问，且提示构造依赖 D/Q gold；N4 Q 行并非来自未知事实的合法补全；N5 原连接只读目标规则，没有从原始 A1 请求核对每条规则文本。旧产物不覆盖，旧 GPU CLI 禁用。

新可执行包：`src/care_m2_v2`。正式新数据：`data_v2_ready`。未投入 GPU 的 prefreeze 数据和初始 null group 失败保留服务器；对应失败日志随本报告提交。新 N1 仍沿用原先家族划分，明确标为已看过无效旧输出后的探索修复，不能宣称独立确认。N5 为历史固定预测迁移，不宣称逻辑家族独立的确认集。

CPU 检查：两个独立 Boolean evaluator；24 个家族在输入变量所有排列下真值表不同；N1 单属性四格；N2 显式 synthetic prior 和中性 receiver；N3 可决定性属性严格比较只翻转该属性的合法补全；N4 真实 review 未知产生 D/Q，目标规则 A 单独求值，实际交换规则文本顺序；N5 从 PR12 不可变 commit 8fb87d 的 A1 recorded request 核对患者和所有规则文本，保留 72 例、216 条 T/F/U/null 信号。CPU 测试及真实 tokenizer 检查原始输出见本目录。

模型固定 Qwen3-4B-Base revision 906bfd4b4dc7f14ee4320094d8b41684abff8539，下载 metadata 校验；BF16、SDPA、batch=1、use_cache=False。沿用 PR8 的 post-RoPE score 语义及 GQA 处理，通过 transformers 官方 SDPA 添加最后 query 的局部 action/source key score bias；不改输出 logits。新的残差坐标操作在 S1 独立检查。N1/N5 干预原动作 token，N2/N3 干预最后 prompt token；记录与直接 bias 的位置差异、真实残差范数和 QK score 幅度，不能声称这两类算子数值等效。

训练：只读 train。成对差分中心化，rank=2，谱混合 lambda 固定 0.5，不搜索层头。N1 同一家族内 R/A 配对；N2/N3 同输入两个 train 政策；N5 从预测 T/F 的原动作位置拟合。随机方向匹配实际请求的残差范数，重新配对轴使用独立排列，不以差分翻号假充随机子空间。dev 为每种主方法独立选 beta={0,.5,1}，主指标最大、同分最小；负对照继承 route 强度。选到零则保留零。

G2：N1 route 四格完整正确率超过真实 direct attention bias 且训练 R/A 差分均非零，才执行 test。失败停止，无额外头/秩搜索。
G3/G4：非零 route 优于 baseline 且不低于同信息文本策略；N2 跨内容 IIA 优于 baseline，receiver 正确率高于 donor-copy 率；N3 两类风险均不高于文本对照。只有两阶段同时通过才做 N4。N4 12 种固定组合、无重拟合/选参；N1 失败使用其独立选择的 direct bias。
N2 真实知识轨：冻结 24 个基础事实、4 领域；两种无选项问题、两种选项顺序全过才算 known。至少8组且4领域才评价 parametric 轨；否则只报告 synthetic 机制。无补题。知识轨为固定 synthetic 控制轴的迁移，不重新选参。
N3 额外记录追问后两种回答下最终动作，错误追问仍计分母。N5 使用全部逐规则信号做文本、F 过滤、direct bias、坐标路由；U/null 不干预、不转换成 Q。原24例旧四臂没有同输入重跑。

统计：按家族主指标，10000 次配对 bootstrap 和小样本精确符号翻转，实际完成 test 的主要比较做 Holm；dev-only 失败不混入 test 推断。所有结果均为探索性机制证据，不作临床有效性主张。

实际 CLI：
`PYTHONPATH=src python -m care_m2_v2.cli {prepare,verify,smoke,fit,select,evaluate,analyze,run-task,run} --out ...`
统一 `run --data ... --out 全新目录 --push` 自动顺序执行、按门槛停止并普通 push 同一 PR；不 merge。完整 help 见 cli_help.txt。

资源：GPU0 单卡串行，预计10–14 GiB，保守不超过20 GiB。S1 7 次前向；N1 <=864、N2 <=1176、N3 <=576、N4 条件执行 <=1536、N5 <=372；均是展开网格及生成最大长度后的上限，精确已执行数由不可覆盖 journal 记录。零强度、同 prompt 同算子和同训练激活缓存共享，预计少于上限。预计全程20–45分钟，若单轴门槛不通过则缩短。
