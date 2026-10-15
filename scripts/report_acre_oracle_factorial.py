#!/usr/bin/env python3
"""Produce the Chinese review and explicit stop decisions from frozen computed results."""
import collections,csv,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.acre_pilot.oracle_diagnostic import *
from scripts.summarize_acre_oracle_factorial import distribution

def main():
    assert read(OUT/"recompute_check.json")["E1_independently_recomputed"]
    d=read(OUT/"oracle_factorial_metrics.json");g=d["gates"];assert not g["E2_offline_triggered"]
    effects=jsonl(OUT/"case_effects.jsonl.gz");items,cases,annotations=normal_inputs();oracle,refs=oracle_inputs()
    idx={(r["case_id"],r["method"]):r for r in effects}
    changes=[]
    for cid in refs:
        for a,b in [("OS@2.0","OF@2.0"),("OS-target@2.0","OF-target@2.0")]:
            x,y=idx[cid,a],idx[cid,b]
            if x["prediction"]!=y["prediction"]:
                item=next(i for i in items if i["case_id"]==cid);ref=refs[cid]
                changes.append({"case_id":cid,"comparison":a+" vs "+b,"target_state":ref["target_applicability"],
                    "patient_decisive_sentence":ref["evidence"],"source_target_rule":next(r["text"] for r in item["guideline"]["rules"] if r["rule_id"]==ref["target_rule_id"]),
                    "a_action":x["action"],"b_action":y["action"],"gold_action":ref["gold_action"],
                    "a_correct":x["correct"],"b_correct":y["correct"],"a_oar":x["oar"],"b_oar":y["oar"],
                    "a_minus_b_margin":x["margin"]-y["margin"],"removed_positions":sorted(set(y["attention"]["targets"])-set(x["attention"]["targets"])),
                    "added_positions":sorted(set(x["attention"]["targets"])-set(y["attention"]["targets"]))})
    write(OUT/"error_analysis.json",{"all_incorrect_case_ids_by_method":{method:[r["case_id"] for r in effects if r["method"]==method and not r["correct"]] for method in d["all_metrics"]},
        "all_primary_and_target_prediction_changes":changes,
        "full_oracle_conclusion":"OS has no corrections over OF and loses two OF-correct decisions; one new OAR. One further prediction change remains wrong in both.",
        "target_only_conclusion":"OS-target corrects three OF-target errors, all inapplicable; one fewer OAR. Highly answer-aware secondary diagnostic; no length or direction sham control.",
        "upstream_analysis_status":"E2 stopped by full-reference primary gate; E0 contains every rule-level discrepancy but no posthoc model changes.",
        "annotation_metrics_reused_from":"round2/revision3/annotation_metrics.json"})
    controls={"E3_executed":False,"reason":"Full-reference OS is 2/80 worse and increases OAR by 1/40; the primary stop gate takes precedence over secondary target-only gains.",
        "extra_gpu_forwards":0,"length_matched_unrelated_control":"not run under stop condition",
        "non_target_control_mask_validity":None,"robustness_claim_supported":False,
        "directional_margins":"already reported as mandatory E1 outcomes; not an E3 experiment",
        "target_only_positive_signal_preserved":True,"target_only_claim_limitation":"cannot distinguish evidence direction, removing action emphasis, or target length; cannot replace full-rule primary comparison",
        "full_oracle_new_negative_tokens_cases":0,"target_only_new_negative_tokens_cases":40}
    write(OUT/"control_validity.json",controls)
    write(OUT/"factorial_metrics.json",{"status":"E2 not executed","reason":controls["reason"],
        "E1_four_cell_results":"oracle_factorial_metrics.json","module1_modified":False})
    (OUT/"upstream_bottleneck_analysis.md").write_text("# E2 未执行\n\n完整规则主对照不支持切换收益，并增加一次不适用动作误用。按任务书停止模块一优化。E0 的 1250 条规则审计保留于 annotation_error_audit.csv；它不构成 E2 的进一步归因或修正验证。仅目标规则的正向结果完整保留，但不替代主比较。\n")
    write(OUT/"run_notes.json",{"experimental_worker_failures":0,"GPU_reruns_for_new_outcomes":0,
        "audit_and_reporting_repairs":[
            {"stage":"E0","issue":"81 cached annotations include one unused smoke record; 80 cases reference 80 unique keys","repair":"validate linkage and uniqueness rather than require archive length 80","data_removed":False},
            {"stage":"summary","issue":"conflict metadata was not copied into evaluation records","repair":"copy original conflict flag; no prediction or metric definition changed"},
            {"stage":"historical regression","issue":"historical references use a samples wrapper, and invalid old spans must fail closed","repair":"adapt historical schema and skip invalid spans exactly as the historical mapper; all 24 masks and scores replayed"}],
        "detailed_logs":"server-only logs/; original failures retained there","metric_definition_changes":False,
        "historical_outputs_changed":False,"new_models_or_algorithmic_modules_added":False})
    # All margin strata and mask lengths are published, including negative subsets.
    strata={}
    for method in d["all_metrics"]:
        rs=[r for r in effects if r["method"]==method]
        groups={"all":rs,"applicable":[r for r in rs if r["target_state"]=="applicable"],
            "inapplicable":[r for r in rs if r["target_state"]=="inapplicable"],
            "predicted_state_correct":[r for r in rs if r["predicted_state_correct"]],
            "predicted_state_wrong":[r for r in rs if not r["predicted_state_correct"]],
            "conflict_disabled":[r for r in rs if r["conflict_disabled"]],
            "conflict_not_disabled":[r for r in rs if not r["conflict_disabled"]],
            "both_nonempty_different":[r for r in rs if r["both_nonempty_different"]],
            "other_masks":[r for r in rs if not r["both_nonempty_different"]]}
        strata[method]={name:{"delta_margin":distribution([r["delta_margin"] for r in rows]),
            "target_token_count":distribution([r["actual_target_count"] for r in rows]),
            "nonzero_enhancement_count":sum(r["attention"]["enabled"] for r in rows)} for name,rows in groups.items()}
    write(OUT/"stratified_margins_and_lengths.json",strata)
    write(OUT/"data_lineage.json",{"base_remote_commit":REMOTE_BASE,"base_remote_tree":"98c3c80e7090b0983386c21c762e6bc13775dd86",
        "dataset":read(BASE/"data_audit.json")["source"],"frozen_normal_manifest":sha(BASE/"development_manifest.json"),
        "frozen_oracle_mask_source":sha(BASE/"development_oracle.json"),"frozen_evaluation_references":sha(BASE/"development_references.json"),
        "normal_annotation_source":sha(OLD/"annotations.jsonl.gz"),"normal_decision_archive":sha(OLD/"case_results.jsonl.gz"),
        "read_boundaries":[{"stage":"E0 normal mask reconstruction","entry":"audit_acre_switch_masks.py before oracle_inputs()","reference_states_used":False},
            {"stage":"E1 base prompt","entry":"span_mapping.build_prompt(tokenizer, item)","input_fields":["guideline","patient","question","decision_protocol"],"reference_text_inserted":False},
            {"stage":"E1 Oracle masks","entry":"get_case: oracle rules to positions; target rule ID only for secondary diagnostic","deployable":False},
            {"stage":"offline scoring","entry":"summarize/verify scripts: frozen references and action aliases","model_calls":False}],
        "normal_model_annotation_calls":0,"normal_B2_B3_reused_unchanged":True,
        "samples":[{"case_id":i["case_id"],"guideline_id":i["guideline_id"],"pair_id":i["pair_id"]} for i in items],
        "independent_validation_selected_or_run":False,"historical_guideline_overlap":[]})
    m=d["all_metrics"]
    def line(name):
        r=m[name]
        return f'| {name} | {r["accuracy"]["numerator"]}/80 | {r["applicable_accuracy"]["numerator"]}/40 | {r["pair_accuracy"]["numerator"]}/40 | {r["oar"]["numerator"]}/40 | {r["retention"]["numerator"]}/64 |'
    table="\n".join(line(name) for name in ["O0","B2@2.0","B3@2.0","OF@2.0","OS@2.0","OF-target@2.0","OS-target@2.0"])
    table1="\n".join(line(name) for name in ["B2@1.0","B3@1.0","OF@1.0","OS@1.0","OF-target@1.0","OS-target@1.0"])
    resource=read(OUT/"resource_plan.json")
    review=f"""# 08 · 正确标注下的目标切换预实验评审

**结论：E0、E1 已实际运行并独立复算。完整规则主对照中，OS 没有优于 OF。按停止条件暂停当前完整规则目标切换路线，不优化模块一，不追加 E3 前向测试，不启动独立验证。**

共同 β=2 时，OF 为 68/80，OS 为 66/80；OS 少答对 2 例，少正确 1 对，并增加 1 次目标动作误用。仅参考目标规则的附加诊断存在 3 例纠正；该结果保留为高度知晓答案的探索性信号，不能替代主比较。没有扩大 β 网格、选头、训练或追加干预算子。

## 1. 固定范围与公平比较

起点为分支 codex/acre-pilot-20261015，远端提交 {REMOTE_BASE}，完整文件树 98c3c80e7090b0983386c21c762e6bc13775dd86。服务器原先落后，已通过已连接 GitHub 接口恢复精确 Git 提交并核对文件树。现有 PR #1 继续承载交付。

使用冻结的 MEGA-CDP train_val 40 对、80 例、36 个指南。数据 revision 为 8f058c88ce1ed43693785b435ad0ff99907a2890。样本与指南 IDs、输入读取点和 SHA256 见 [data_lineage.json](../../results/acre_pilot/oracle_switch_diagnostic/data_lineage.json)。没有新筛选、替换或删例，没有读取独立测试进行实验。历史 24 例仅作 CPU 回归。

OF/OS 来自同一份冻结 development_oracle：相同相关性、状态、动作、支持／否定原文和例外字段，仅选择策略不同。**冻结 Oracle 协议将全部所供规则标为相关。这里的“正确标注”指沿用任务书指定、已核验路径和证据的参考协议，不是独立专家确认了所有规则的语义相关性。** 没有更换该协议以获得有利结果。数据无独立例外片段，没有可靠未知真值样本。

模型仅为现有 Qwen3-8B，BF16、SDPA，seed=20261015。正常 B2/B3 原样复用，未重新自标注。参考信息只用于明确标为不可部署的 Oracle 掩码与离线评分。完整指南、患者、问题、聊天模板和全部候选动作保持相同；所有 80 例基本提示哈希一致。模型及 5 个权重分片的实际 SHA256 已重新核验。

## 2. E0：B2/B3 究竟改变了什么

80 例全部覆盖，旧掩码与保存位置逐项一致，全部旧网格指标精确复现。81 条标注缓存中有 1 条未被这 80 例引用的 smoke 记录；按病例保存的缓存键追溯 80 条正确缓存，没有丢弃病例。

| 互斥类别 | 例数 |
| --- | ---: |
| 掩码相同且非空 | 23 |
| 两侧均非空且不同 | 26 |
| 仅一侧为空 | 5 |
| 因预测规则动作冲突统一关闭 | 25 |
| 其他未增强 | 1 |

共 31/80 例集合不同，其中真正双侧非零对照为 **26/80**。两侧皆空的 Jaccard 留空。31 例都移除了动作 token，只有 **3/80** 例新增否定证据位置，分别是 v2_244_7_original、v2_244_7_counterfactual、v2_1599_17_original；其余变化不能描述成“新增否定证据增强”。

B2/B3 的最终标签不同 2 例，但正确／错误集合相同，均为 64/80、29/40 对、OAR 3/40。掩码和动作同义表达都不能只看总准确率。完整位置、角色与原文见 [mask_audit.csv](../../results/acre_pilot/oracle_switch_diagnostic/mask_audit.csv)。全部 1250 条规则的状态、相关性、证据与源位置见 [annotation_error_audit.csv](../../results/acre_pilot/oracle_switch_diagnostic/annotation_error_audit.csv)。

## 3. E1：完整规则是主要结论

| 方法（β=2） | 总正确 | 适用正确 | 成对正确 | 不适用动作误用 | 原正确保持 |
| --- | ---: | ---: | ---: | ---: | ---: |
{table}

OF/OS 的 **80/80** 例最终集合均非空且不同，均实际增强。然而，OS 的最终集合全部是 OF 的子集：**0/80 例新增否定证据，80/80 例只删除增强位置**。其他规则的支持证据已覆盖 OS 的否定证据；不能将其描述为 80 例有效的否定目标切换。平均增强词元数 OF=658.89，OS=146.20，长度与覆盖率尚未控制。

OS 相对 OF：只 OS 正确 0 例、只 OF 正确 2 例、共同正确 66 例、共同错误 12 例；另有 1 例两者都错误但标签不同。三例所有动作变化均公开，未只展示成功病例。

在真实不适用的 40 例中，OF 正确 31/40、OS 正确 30/40，OAR 分别为 2/40、3/40。原 B0 正确的 64 例均保留，但这不能抵消相对 OF 损失的 2 例新纠正。

正确动作与固定反事实竞争动作的边际差 OS−OF：全部 80 例为 **正 36／零 4／负 40**，均值 -0.315625；不适用 40 例为 **正 13／零 1／负 26**，均值 -0.771875。并无方向一致的主比较收益。

预先固定的 β=1 结果如下，没有重新择优：

| 方法（β=1） | 总正确 | 适用正确 | 成对正确 | 不适用动作误用 | 原正确保持 |
| --- | ---: | ---: | ---: | ---: | ---: |
{table1}

β=1 的 OF/OS 均为 65/80、30/40 对、OAR 4/40；仍有 1 例标签不同且双方都错。不适用边际 OS−OF 为正 13／零 3／负 24，均值 -0.25625，不支持主方向改善。

## 4. 仅目标规则的附加结果，不能隐去也不能替代主结果

在相同参考目标规则下，适用 40 例 OF-target/OS-target 掩码完全相同；不适用 40 例均为动作与否定证据之间真正不同的非零对照，40 例都新增否定证据。该目标规则 ID 来自离线参考，属于高度知晓答案、不可部署的诊断。

β=2 时，OS-target 比 OF-target 多纠正 3 例、没有正确性退化；准确率 65→68/80，PairAcc 29→31/40，OAR 4→3/40。三例都真实不适用：v2_169_21_original、v2_1903_24_original、v2_2315_3_original。不适用边际正 35／零 2／负 3，均值 +1.653125。β=1 仍多正确 1 例，但 OAR 相同。

这说明在参考直接指定目标、排除其他规则合并时存在值得保留的局部机制信号。它不证明完整规则算法有效，也不能区分否定证据方向、删除动作强调或增强长度的作用。本轮按完整规则主比较停止，不能据此再挑选规则、改相关性或追加模块获得开发集正结果。

## 5. 具体错误与相反方向的证据

- v2_169_21_counterfactual：原文明确 “Remote self-sampling was available as a local testing option.”。OF 选择正确的远程自采样，OS 改为其他检测方式，边际下降 3.25。
- v2_2315_3_original：原文明确无法及时获得完整新生儿心超。OF 选择有限 cPOCUS 并尽早安排完整检查，OS 改为及时完整影像，构成新增误用，边际下降 3.25。相反，同一病例的仅目标规则 OS-target 能纠正 OF-target 的误用，表明多规则合并改变了实际作用。
- v2_775_26_counterfactual：OF 弃答，OS 选择另一手术分支，双方都不符合参考的单独放疗；不能将标签变化算纠正。
- 仅目标规则的另外两例纠正分别涉及远程自采样不可用与替代 CT 不可用，完整患者决定句、规则条件、两组动作和 token 变动保存在 [error_analysis.json](../../results/acre_pilot/oracle_switch_diagnostic/error_analysis.json)。

全部错误病例 ID 按条件保存，逐例预测、固定竞争动作、边际、掩码和隔离标记均在 case_effects.jsonl.gz 中。没有后验修改动作别名，OAR 继续使用上一轮对全部 40 对统一审计的别名。

## 6. 统计与停止判断

按指南聚类 bootstrap 2000 次，seed=20261015，反事实对始终完整。OS−OF 的准确率差为 -2.5 个百分点，探索性 95% 区间 [-6.41, 0.00] 个百分点，McNemar p=0.50；PairAcc 差 -2.5 个百分点，区间 [-8.11, 0.00]；OAR 差 +2.5 个百分点，区间 [0.00, 8.11]。仅目标规则准确率差 +3.75 个百分点，区间 [0.00, 8.54]，p=0.25。所有配对四格、样本量和区间见 paired_comparisons.csv；p 值未经多重比较校正，均为描述性、探索性结果。

本批已用于开发和选择 β。36 指南的区间既不是独立证据，也不能证明总体等效或普遍无效。没有未知真值、临床专家盲审、有效等长无关片段对照或独立验证。

**模块一不是本轮主对照失利的充分解释。** 参考标注下 OS 仍不优于同源 OF。旧预测目标状态错误 20/80、其中假适用 12/40、冲突关闭 25/80 均真实存在；但不能继续优化自标注来证明主方法。E2 进一步归因与修改、E3 额外前向均未执行，原因记录于 control_validity.json、factorial_metrics.json 和 upstream_bottleneck_analysis.md。

对研究方向的建议是：**暂停当前完整规则实现和正式验证**。已有结果更直接指向实际目标对照覆盖与多规则合并问题，不能单独归因为全部注意力引导无效。若未来另行研究，须预先提出新指南上的真实否定目标覆盖方案，并解决目标选择来源和长度控制；本轮没有生成或运行该新方案。

## 7. 工程与复现

E0 完全在服务器 CPU 执行，并在新目录内调用原有安全解压脚本恢复归档。旧 revision3、历史诊断、模块二／三均经过前后 SHA256 核对，未改动。20 项现有 CPU 测试通过；历史 24 例掩码与 B0/B2/B3 21/24 正确率回归通过。

OS 与旧 B6 的 80 例掩码全部重算相同；按提示长度预先选定的 6 例在两个 β 上实测候选 logits 和对数概率精确相同，因此复用 B6，未全量重复已有 OS 前向。全部 80 例重新确认 B0 选项 logits 一致，以及零偏置和空目标完整词表 logits 精确恒等。因果未来注意力为零，输入查询 logits 不变，全部 36 层×32 查询头、8 KV 头生效。

GPU 只用 0、1、2、3、6、7。正式前已汇报原计划及 smoke 修订。完整 E1 墙钟 {resource["full_wall_seconds"]:.0f} 秒；峰值分配 {resource["actual_peak_allocated_gib"]:.2f} GiB、峰值保留 {resource["actual_peak_reserved_gib"]:.2f} GiB。新前向 {resource["actual_new_forwards"]} 次，完全相同掩码复用 {resource["identical_mask_reuses"]} 次；另有 smoke 和算子检查。稳定后没有持续监控，全部实验进程已退出。

独立复算脚本未调用 evaluator、select_targets 或 map_annotations，单独从原文字符坐标重建增强集合，逐条核验 **80 例×13 条件=1040 条记录**、全部主指标和分层指标、配对计数及 McNemar。完整权重 SHA256、配置和代码指纹见 recompute_check.json 与 final_code_fingerprints.json。GPU 无失败或为寻找正结果而重跑；审计／报告适配错误与修复单列在 run_notes.json，原始详细日志仅保留服务器。

复现入口见 [结果目录 README](../../results/acre_pilot/oracle_switch_diagnostic/README.md)。不要运行旧汇总脚本改写旧目录，也不要在已完成 worker 路径覆盖重跑。结果文件已独立校验后才标记本轮 E0/E1 完成；E2/E3/独立验证仍明确未运行。
"""
    (ROOT/"docs/ACRE/08-oracle-switch-results.md").write_text(review)
    readme="""# Oracle 目标切换诊断（2026-10-15）

状态：E0/E1 实际完成且独立复算。主对照不支持 OS 优于 OF，停止 E2/E3 新实验和独立验证。Oracle 均不可部署。正式中文评审见 ../../../docs/ACRE/08-oracle-switch-results.md。

β=2：O0 64/80；完整 OF 68/80、31/40 对、OAR 2/40；完整 OS 66/80、30/40 对、OAR 3/40。仅目标规则 OF-target 65/80，OS-target 68/80，是次要机制诊断，不能替代主比较。

## 复现

所有命令只在 gcx@10.29.128.115 的 /home/gcx/TRIAGE 运行，source /home/gcx/gcx261014/bin/activate，HF_HUB_OFFLINE=1、TRANSFORMERS_OFFLINE=1。新 checkout 先将原有数据和模型放到原路径。

CPU 审计：python scripts/audit_acre_switch_masks.py。它把历史归档复制到本目录 _restored 后，用已提交的 unpack_acre_pilot_artifacts.py 安全恢复，验证哈希和缓存链接，不写旧实验。

GPU 首先按 README 向用户报告资源；随后 CUDA_VISIBLE_DEVICES=0 python scripts/check_acre_oracle_operator.py，CUDA_VISIBLE_DEVICES=0 python scripts/run_acre_oracle_factorial.py --smoke。完整启动：bash scripts/launch_acre_oracle_factorial.sh，六 worker 映射 GPU 0、1、2、3、6、7。已完成 worker/case 不允许覆盖；复现须另外归档本目录或在隔离 checkout 使用新副本。

离线汇总：python scripts/summarize_acre_oracle_factorial.py --from-archive（直接使用已交付压缩记录）；完整重跑后可省略该参数以合并 worker 原始文件。独立复核：CUDA_VISIBLE_DEVICES="" python scripts/verify_acre_oracle_factorial.py。中文报告与停止记录：python scripts/report_acre_oracle_factorial.py。

只验证交付的压缩记录时，运行独立复核脚本会直接读取 case_effects.jsonl.gz，不需要加载 GPU 模型。worker/done 和详细日志不提交；若这些文件未恢复，资源退出检查标为“仅复核已提交的完成凭证”，不伪造新运行。

## 结果入口

- mask_audit.csv、mask_audit_summary.json：80 例旧 B2/B3 最终集合、角色、Jaccard 与互斥类别。
- oracle_mask_audit.csv、oracle_mask_plan.json：同一参考来源下完整与仅目标规则对照。
- annotation_error_audit.csv：1250 条规则，无失败病例排除；字符重叠精确性仅为注册决定句代理，非临床语义精确率。
- oracle_factorial_metrics.json：13 条件、全部和分层指标、固定边际、配对统计。
- case_effects.jsonl.gz：1040 条记录，全部预测、logits、目标位置、提示哈希和隔离标记。
- prediction_changes.csv、paired_comparisons.csv、error_analysis.json、stratified_margins_and_lengths.json：所有正负变化、四格、指南聚类区间、分层方向及目标长度。
- control_validity.json、factorial_metrics.json、upstream_bottleneck_analysis.md：E2/E3 未执行原因。
- resource_plan.json、smoke_checks.json、operator_gpu_checks.json、recompute_check.json：计划、真实资源与验收。
- data_lineage.json、historical_fingerprints.json、final_code_fingerprints.json、run_notes.json：版本、IDs、读取点、哈希、历史保护与工程修复。

CPU 审计可恢复缺失的未压缩冻结输入；已存在输入须字节相同，不覆盖任何已有历史结果。

CSV 含 JSON 字段和长原文。使用 Python csv 读取时先设置 csv.field_size_limit(sys.maxsize)，不能按换行手工切分。JSONL 需通过 gzip.open 逐行读取。本批开发结果不支持独立验证、等效性或部署性能主张。
"""
    (OUT/"README.md").write_text(readme)
    root=ROOT/"README.md";text=root.read_text()
    marker="## 下一轮关键实验（任务书；尚未执行）" if "## 下一轮关键实验（任务书；尚未执行）" in text else "## 最新关键实验：2026-10-15 正确标注下的目标切换诊断"
    start=text.index(marker)
    end=text.find("\n## ",start+len(marker))
    if end<0:end=len(text)
    replacement="""## 最新关键实验：2026-10-15 正确标注下的目标切换诊断

已完成 E0/E1 并独立复算全部 80 例。相同参考关系与证据、共同 β=2 下，固定增强 OF **68/80、31/40 对、OAR 2/40**；切换增强 OS **66/80、30/40 对、OAR 3/40**，原正确均保持 64/64。主比较没有切换收益，按停止条件暂停当前完整规则路线；E2 自标注优化、E3 新前向和独立验证均未运行。

完整 Oracle 的 80 例掩码虽不同，但 OS 只是 OF 的子集，没有新增否定证据。仅参考目标规则的次要诊断在 40 例不适用病例真正改变目标，OS-target 比 OF-target 多纠正 3 例（65→68/80），保留为不可部署、知晓答案的局部机制信号，不能替代主比较。β=1 的完整 OF/OS 指标相同，未重新选择参数。

- [任务书](docs/ACRE/07-next-experiments.md)、[中文评审](docs/ACRE/08-oracle-switch-results.md)
- [新结果与复现](results/acre_pilot/oracle_switch_diagnostic/README.md)、[完整指标](results/acre_pilot/oracle_switch_diagnostic/oracle_factorial_metrics.json)、[独立复算](results/acre_pilot/oracle_switch_diagnostic/recompute_check.json)
- 历史 revision3 和旧模块不变。正常预测与 Oracle 来源隔离。全部结果仍属已观察开发集的探索性分析。

"""
    text=text[:start]+replacement+text[end:]
    # Earlier round conclusions are historical, not the latest next-step directive.
    text=text.replace("**B3 没有优于 B2，当前不进入独立验证或完整论文实验。**","**上一轮 B3 没有优于 B2。最新同源 Oracle 对照见下节；当前不进入独立验证或完整论文实验。**")
    text=text.replace("## 当前阶段：2026-10-15 简化 ACRE 开发集预实验","## 上一轮：2026-10-15 简化 ACRE 开发集预实验")
    latest=text.index("## 最新关键实验：")
    endlatest=text.find("\n## ",latest+3)
    if endlatest<0:endlatest=len(text)
    block=text[latest:endlatest]
    without=text[:latest]+text[endlatest:]
    first=without.index("## ")
    text=without[:first].rstrip()+"\n\n"+block.strip()+"\n\n"+without[first:].lstrip()
    root.write_text(text)
    write(OUT/"final_code_fingerprints.json",code_fingerprints())
if __name__=="__main__":main()
