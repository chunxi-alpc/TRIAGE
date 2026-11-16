from pathlib import Path
import collections
import html
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]

def read(name, default=None):
    path = ROOT / name
    return json.loads(path.read_text()) if path.exists() else default

def rows(name):
    path = ROOT / name
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []

def pct(value):
    return "—" if value is None else f"{100 * value:.2f}%"

def number(value, decimals=4):
    return "—" if value is None else f"{value:.{decimals}f}"

def metric(value, interval=True):
    if not value:
        return "未完成"
    text = f"{pct(value.get('estimate'))} ({value.get('numerator')}/{value.get('denominator')})"
    ci = value.get("ci95")
    return text + (f" [{pct(ci[0])}, {pct(ci[1])}]" if interval and ci else "")

def estimate(metrics, method, name):
    return metrics.get(method, {}).get(name, {}).get("estimate")

def matches(output, answer):
    normalize = lambda s: re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
    a, b = normalize(answer), normalize(output)
    return bool(a) and (a == b or (len(a) >= 4 and re.search(r"(?:^| )" + re.escape(a) + r"(?: |$)", b) is not None))

sections, plain = [], []

def h(title):
    sections.append("<h2>" + html.escape(title) + "</h2>")
    plain.append("\n" + title + "\n")

def p(text):
    sections.append("<p>" + html.escape(text) + "</p>")
    plain.append(text + "\n")

def table(headers, data):
    head = "".join("<th>" + html.escape(str(x)) + "</th>" for x in headers)
    body = "".join("<tr>" + "".join("<td>" + html.escape(str(x)) + "</td>" for x in row) + "</tr>" for row in data)
    sections.append("<div class=scroll><table><thead><tr>" + head + "</tr></thead><tbody>" + body + "</tbody></table></div>")
    plain.append(" | ".join(headers) + "\n" + "\n".join(" | ".join(map(str, row)) for row in data) + "\n")

analysis = read("results/analysis.json", {})
free_analysis = read("results/free_analysis.json", {})
rules_complete = read("results/rules_complete.json", {})
free_complete = read("results/free_complete.json", {})
cfull_complete = read("results/c_full/complete.json", {})
free_protocol = read("audit/free_generation_protocol.json", {})
free_acceptance = read("audit/free_acceptance.json", {})
cfull_acceptance = read("audit/c_full_acceptance.json", {})
cfull_plan = read("data/c_full_subset_plan.json", {})
cfull_prepared = read("results/c_full/prepared.json", {})
direction = read("results/direction.json", {})
chosen = read("results/selected.json", {})
key = read("results/key_selected.json", {})
key_space = read("results/key_subspace.json", {})
data_manifest = read("data/manifest.json", {})
facts_provenance = read("sources/facts_provenance.json", {})
official_commit = read("sources/official_commit.json", {})
environment = read("results/engine.json", {})
filter_audit = read("audit/filter_attrition.json", {})
direction_bootstrap = read("audit/training_direction_bootstrap.json", {})
cases = rows("data/cases.jsonl")
fact_inputs = rows("data/facts.jsonl")
predictions = rows("results/predictions.jsonl")
cfull = rows("results/c_full_predictions.jsonl")
free_predictions = rows("results/free_predictions.jsonl")
methods = ["A", "B", "C_reduced", "D_adapt"]
epsilon = key.get("epsilon")
labels = {"A": "A 无干预", "B": "B 固定来源偏好方向", "C_reduced": "C 简化适用性门控",
          "D_adapt": f"D 键空间适配（epsilon={epsilon if epsilon is not None else '未选定'}）"}
main = analysis.get("metrics", {}).get("task:patient_routing", {})
transfer = analysis.get("metrics", {}).get("task:equipment_routing", {})
primary = analysis.get("comparisons", {}).get(analysis.get("primary_scope", "source_test"), {}).get("C_reduced-B", {})
accuracy_comparison = primary.get("accuracy", {})
main_cases = [r for r in cases if r["split"] == "test"]
transfer_cases = [r for r in cases if r["split"] == "transfer"]
targets = main_cases + transfer_cases
main_families = len({r["rule_family_id"] for r in main_cases})
main_count = len(main_cases)
transfer_count = len(transfer_cases)
cfull_expected_ids = set(cfull_plan.get("item_ids", []))
cfull_expected = cfull_plan.get("items", len(cfull_expected_ids))
cfull_done = (cfull_complete.get("status") == "complete" and len(cfull) == cfull_expected
              and {r["item_id"] for r in cfull} == cfull_expected_ids and bool(cfull_expected_ids))
expected_free_keys = {(r["item_id"], method) for r in targets for method in methods}
actual_free_keys = {(r["item_id"], r["method"]) for r in free_predictions}
free_done = (free_complete.get("status") == "complete" and actual_free_keys == expected_free_keys
             and len(free_predictions) == len(expected_free_keys) and bool(targets))
free_coverage = free_analysis.get("coverage", {})
free_hash = hashlib.sha256((ROOT / "results/free_predictions.jsonl").read_bytes()).hexdigest() if (ROOT / "results/free_predictions.jsonl").exists() else None
free_accepted = ((free_acceptance.get("accepted") is True or free_acceptance.get("status") in {"passed", "accepted"})
                 and free_acceptance.get("predictions_sha256") == free_hash and free_hash is not None)
free_stats_done = (free_done and free_accepted and bool(free_analysis) and all(
    free_coverage.get(method, {}).get("observed") == free_coverage.get(method, {}).get("expected")
    and free_coverage.get(method, {}).get("expected") == len(targets) for method in methods))
free = free_analysis if free_stats_done else {}
expected_rule_keys = {(r["item_id"], method) for r in cases if r["split"] in {"dev", "test", "transfer"} for method in methods}
rules_done = (rules_complete.get("status") == "complete" and
              {(r["item_id"], r["method"]) for r in predictions} == expected_rule_keys and
              len(predictions) == len(expected_rule_keys))
source_train_count = sum(r["split"] == "train" for r in fact_inputs)
source_test = rows("results/source_test.jsonl")
source_dev = rows("results/source_dev.jsonl")
fact_by_id = {r["item_id"]: r for r in fact_inputs}
filter_rows = rows("results/fact_filter.jsonl")
train_both = sum(r.get("stage") == "conflict" and r.get("correct", False)
                 and matches(r.get("raw_output", ""), fact_by_id[r["item_id"]]["answer_original"])
                 for r in filter_rows if r["item_id"] in fact_by_id)
dev_both = sum(r.get("context_match", False) and r.get("parametric_match", False) for r in source_dev)
test_both = sum(r.get("context_match", False) and r.get("parametric_match", False) for r in source_test)
commit_sha = official_commit.get("sha", "未记录")
source_grid = [r["alpha"] for r in chosen.get("selection", [])]
case_splits = collections.Counter(r["split"] for r in cases)
fact_splits = collections.Counter(r["split"] for r in fact_inputs)

h("结论与完成范围")

early_norm = read("results/free_format_normalized_analysis.json", {})
if early_norm:
    early_main = early_norm.get("metrics", {}).get("task:patient_routing", {})
    if early_main:
        p("更贴近实际回答的自由生成结果（兼容常见答案格式、事后补充）：主测试A=" +
          pct(early_main["A"]["accuracy"]["estimate"]) + "，B=" +
          pct(early_main["B"]["accuracy"]["estimate"]) + "，C_reduced=" +
          pct(early_main["C_reduced"]["accuracy"]["estimate"]) +
          "。该格式修复不使用gold、不重新生成；原严格格式解析和首次选项评分全部保留，二者不应与实际生成准确率混淆。")

p("已完成部分及未完成部分按下表区分。这是单模型、小规模方法适配，不是原论文全规模复现，也不是CARE原上传方法的完整复现。所有实验数值读取保存的JSON/JSONL；没有完成标记和完整输出的阶段不报告计划中的成绩。")
if main and accuracy_comparison:
    difference = accuracy_comparison.get("difference")
    hp = accuracy_comparison.get("primary_holm_p")
    cp = accuracy_comparison.get("cluster_test", {}).get("p")
    p(f"首轮为约束选项评分：主测试{main_count}例，A={pct(estimate(main, 'A', 'accuracy'))}，B={pct(estimate(main, 'B', 'accuracy'))}，C_reduced={pct(estimate(main, 'C_reduced', 'accuracy'))}。"
      f"C−B准确率差为{number(100 * difference, 3) if difference is not None else '—'}个百分点，家族置换检验p={number(cp)}，预设主要比较Holm校正p={number(hp)}。"
      f"新规则误用率B={pct(estimate(main, 'B', 'new_rule_misuse'))}，C={pct(estimate(main, 'C_reduced', 'new_rule_misuse'))}。")
    conclusions = []
    if hp is not None:
        conclusions.append("C−B准确率差未达到校正后的显著性标准" if hp >= 0.05 else "C−B准确率差达到校正后的显著性标准")
    ca, aa = estimate(main, "C_reduced", "accuracy"), estimate(main, "A", "accuracy")
    if ca is not None and aa is not None and ca <= aa:
        conclusions.append("C的本次准确率没有超过无干预A")
    cbm, bm = estimate(main, "C_reduced", "new_rule_misuse"), estimate(main, "B", "new_rule_misuse")
    if cbm is not None and bm is not None and cbm >= bm:
        conclusions.append("C没有降低相对B的新规则误用率")
    if conclusions:
        p("；".join(conclusions) + "。不能仅凭准确率差将本次结果概括为CARE全面改进。")
if predictions:
    option_maps = {r["item_id"]: r["options"] for r in cases}
    label_first = sum(r.get("raw_output", "").strip() in option_maps.get(r["item_id"], {}) for r in predictions)
    p(f"首轮{len(predictions)}条保存预测中，自然首词元可直接读作选项字母的有{label_first}条；这些主指标表示候选字母评分。自由生成作为探索性补充单列，不覆盖首次结果。")
free_plan_count = free_protocol.get("unique_completions")
free_detail = (f"实际{free_complete.get('unique_completions')}项唯一生成，部分方法精确复用"
               if free_done else f"计划{free_plan_count if free_plan_count is not None else '未记录'}项唯一生成，尚不报告完成数量")
status = [
    ("论文/官方代码核验", "已完成" if commit_sha != "未记录" else "未完成", "固定commit及实现差异见来源审查"),
    ("CARE原上传文件", "未完成", "指定txt未找到；不能声明读过"),
    ("A/B/C_reduced/D_adapt约束评分", "已完成" if rules_done else "未验收完成",
     f"实际保存{len(predictions)}条预测；计划{len(expected_rule_keys)}条"),
    ("原D完整适用性键子空间", "未完成", f"只实现适配方法；dev所选epsilon={epsilon}"),
    (f"自由生成{len(targets)}例", "已完成并统计验收" if free_stats_done else "生成完成，独立验收待完成" if free_done else "未验收完成", free_detail),
    (f"实际V6模块一{cfull_expected}例", "已完成" if cfull_done else "未验收完成",
     f"固定子集；只覆盖{len(cfull_plan.get('families', []))}家族，不代表全部{len(cases)}例"),
    ("多模型、多方向训练种子、真实临床外部验证", "未完成", "不得以本轮替代")
]
table(["项目", "状态", "说明"], status)

h("可核验来源与方法选择")
p(f"原论文：2026-10-14 arXiv:2610.20067v1。官方代码：AheadOFpotato/knowledge_selection_steering，commit {commit_sha}。来源链接及逐文件校验值见sources/paper_method_audit.md。论文描述的末提示词元干预与官方全词元广播不同；本实验采用论文描述，不声称逐字运行官方hook。")
sections.append("<p><a href=https://arxiv.org/abs/2610.20067>原论文</a> · <a href=https://github.com/AheadOFpotato/knowledge_selection_steering/tree/" + html.escape(commit_sha) + ">固定版本官方代码</a> · <a href=https://huggingface.co/datasets/kewu93/ClashEval>ClashEval来源</a></p>")
p(f"B：训练集clean与conflict残差均值之差，选择均值余弦最小的非最终层，向量不单位化。一次prefill只修改末提示词元。训练候选须各clean指令变体都命中旧答案，并在冲突提问命中修改答案；指令变体不等同于独立问题改写，完整定义见direction.json。系数只在源dev网格{source_grid}中选择，之后冻结。")
p(f"本次train合格{direction.get('eligible_train')}/{source_train_count}，层索引{chosen.get('layer')}（从0开始），alpha={chosen.get('alpha')}。原论文Qwen3公布层为21；本次使用更换来源与提示后的重新定位结果。字符串匹配可能漏掉别名、变音符号；合格样本规模与单训练种子不足以验证方向稳定性。")
p("C_reduced：从患者文本和已给定结构化规则中独立计算，复用现有V6数值比较与三值逻辑；布尔条件只做明确句子解析。存在冲突时，NEW=T给+1（不要求旧规则有效）；NEW=F且旧规则有效给−1；NEW=F且旧规则失效，或NEW=U，给0。无冲突时给0。这个门控值乘以B的冻结系数，方向和层与B相同。它不是完整Phi-4/BioLinkBERT模块，也没有直接读取gold/applicable字段。")
p(f"D_adapt：用{key_space.get('train_pairs')}个train最小对，在层索引{key_space.get('layer')}的每个KV头k_norm后、RoPE前提取动作词Key均值差，建rank={key_space.get('rank')}的SVD投影。只在目标dev选择相对扰动，结果如下。")
table(["dev扰动系数", "正确数/病例数", "准确率"],
      [(r["epsilon"], f"{round(r['accuracy'] * r['n'])}/{r['n']}", pct(r["accuracy"])) for r in key.get("selection", [])])
p(f"同分选择较小扰动，所选epsilon={epsilon}。" + ("因此本次D没有实际Key扰动，预测等于A。" if epsilon == 0 else "D按该冻结扰动系数评估。") + "这不能证明原CARE子空间有效或无效。")

h("训练筛选与方向稳定性诊断")
if filter_audit:
    stages = filter_audit.get("stages", {})
    attrition = filter_audit.get("attrition", {})
    table(["筛选阶段", "实际尝试", "原匹配器通过", "未通过"],
          [(stage, values.get("attempted"), values.get("passed"), values.get("failed")) for stage, values in stages.items()])
    p(f"从{filter_audit.get('train_candidates')}个候选开始，第一种clean提问通过{stages.get('clean0', {}).get('passed')}个；全部clean变体通过{stages.get('clean4', {}).get('passed')}个；最终conflict跟随通过{filter_audit.get('original_eligible')}个。"
      f"第一遍未匹配{attrition.get('first_clean_failed')}个，后续clean变体另外淘汰{attrition.get('later_clean_variant_failed')}个，conflict另外淘汰{attrition.get('conflict_did_not_follow')}个。"
      "因此不能把低合格数主要归因于多遍要求；第一遍覆盖、答案匹配、题目与标签质量同样需要检查。未匹配本身不等于模型没有相关知识。")
    p(f"Unicode规范化复核恢复了{filter_audit.get('normalization_recovered_cases')}个已有阶段的匹配，但这些候选的后续阶段未完成；新增完整验证合格样本为{len(filter_audit.get('newly_verified_eligible_ids', []))}个。原方向和正式合格数没有因此修改。")
else:
    p("尚无完整筛选分解审查，不从最终合格数推断主要淘汰原因。")
if direction_bootstrap:
    repeats = direction_bootstrap.get("replicates", 0)
    counts = direction_bootstrap.get("selected_layer_counts", {})
    table(["bootstrap重新选中的层", "次数/重采样次数", "比例"],
          [(layer, f"{count}/{repeats}", pct(count / repeats) if repeats else "—") for layer, count in sorted(counts.items(), key=lambda item: int(item[0]))])
    quantiles = direction_bootstrap.get("cosine_to_reference_quantiles_025_50_975", [])
    categories = direction_bootstrap.get("eligible_source_categories", {})
    p(f"这是对同一批{direction_bootstrap.get('n')}对训练样本的{repeats}次探索性经验bootstrap，seed={direction_bootstrap.get('seed')}。"
      f"固定参考层{direction_bootstrap.get('reference_layer')}的方向余弦相似度2.5%/50%/97.5%分位数为{[round(x, 3) for x in quantiles]}；"
      "合格来源类别计数为" + json.dumps(categories, ensure_ascii=False) + "。")
    p("这些重采样只刻画现有训练对的内部波动；没有新增合格数据，没有重新训练其他模型，也没有检验不同来源分布。不能据此证明跨数据方向稳定性，未使用该诊断重新选择推断参数。")

h("数据与防泄漏")
split_order = ["train", "dev", "test", "transfer"]
family_counts = [len({r["rule_family_id"] for r in cases if r["split"] == split}) for split in split_order]
table(["数据"] + split_order + ["隔离单位"], [
    ["ClashEval事实问题"] + [fact_splits[s] for s in split_order] + [f"question去重后拆分，seed={facts_provenance.get('seed')}"],
    ["合成规则病例"] + [case_splits[s] for s in split_order] + [f"对应具体家族数{family_counts}，seed={data_manifest.get('seed')}"]
])
p("合成数据有最小年龄条件翻转、多规则联合动作、例外、旧规则失效、适用性未知、无冲突。患者在规则之前，避免动作Key因因果遮罩读不到患者。最小对选项顺序相同，动作与选项映射随机，标签由独立确定性逻辑构造。训练/开发/测试的具体家族互斥，但通用逻辑模板共享。")
p("新旧规则都明确放在提示中，因此目标数据只验证上下文内部规则选择；旧规则回退不是模型参数旧知识回退。设备路由迁移保持相同逻辑模板，只能称领域表述迁移。ClashEval另有闭卷检查，未把全部事实问题都当成模型已掌握知识。")

h(f"首次约束评分：主测试集{main_count}例")
metric_names = [("accuracy", "准确率"), ("new_rule_misuse", "新规则误用率↓"), ("flip_pair_joint_accuracy", "翻转对联合正确率"),
                ("multi_rule_exact_accuracy", "多规则完整正确率"), ("valid_old_fallback_accuracy", "有效旧规则回退"), ("no_conflict_accuracy", "无冲突准确率")]
for keys in [metric_names[:3], metric_names[3:]]:
    table(["方法"] + [name for _, name in keys], [[labels[m]] + [metric(main.get(m, {}).get(k)) for k, _ in keys] for m in methods])
pair_count = main.get("A", {}).get("flip_pair_joint_accuracy", {}).get("denominator")
p(f"方括号为按规则家族重采样的95% bootstrap区间（{analysis.get('bootstrap_replicates')}次）；主测试只有{main_families}家族。翻转对共{pair_count}对，同家族病例并不独立。区间不包括重新训练方向的随机性。条件分层、次要McNemar检验和多重校正见results/analysis.json。")
p("若区间为[100%,100%]（或其他零宽区间），只表示当前观测家族在重采样中没有相应变异，不代表总体性能确定为100%，也不代表没有失败风险。家族数少、共享模板和仅一个方向训练种子都限制外推。")

h(f"首次约束评分：迁移{transfer_count}例")
table(["方法", "准确率", "误用率↓", "多规则完整正确率"],
      [[labels[m]] + [metric(transfer.get(m, {}).get(k)) for k in ["accuracy", "new_rule_misuse", "multi_rule_exact_accuracy"]] for m in methods])

h("事实源任务的留出测试")
facts_table, source_counts = [], {}
for method in ["closed_book", "A", "B"]:
    rr = [r for r in source_test if r["method"] == method]
    if rr:
        hits = sum(r.get("correct", False) for r in rr) if method == "closed_book" else sum(r.get("context_match", False) for r in rr)
        source_counts[method] = (hits, len(rr))
        facts_table.append((method, len(rr), hits, "旧答案闭卷正确数" if method == "closed_book" else "冲突上下文答案匹配数"))
table(["方法", "分母", "匹配数", "含义"], facts_table)
if source_counts:
    detail = "；".join(f"{method}={hits}/{count}" for method, (hits, count) in source_counts.items())
    p("源任务保存结果：" + detail + "。总体上下文匹配与闭卷已知子集是不同评价人群，不能把前者都解释为可靠参数知识与上下文的冲突。")
if "A" in source_counts and "B" in source_counts and source_counts["A"][1] == source_counts["B"][1]:
    a, b = source_counts["A"], source_counts["B"]
    p("本次源test的B上下文匹配数" + ("低于" if b[0] < a[0] else "等于" if b[0] == a[0] else "高于") + "A；该计数比较本身不是显著性结论。")
p(f"接受的训练conflict、源dev、源test中同时命中新旧答案的记录分别为{train_both}、{dev_both}、{test_both}。"
  + ("因此当前匹配器没有显式排除both的缺口未改变本次系数选择。" if train_both + dev_both + test_both == 0 else "存在both记录，必须同时核对contextual-only指标，不能把含两个答案的输出当成单一来源选择。")
  + "仍需别名评估和更多闭卷合格样本。")

h("自由生成补充评测")
if free:
    for scope, title in [("task:patient_routing", "主测试"), ("task:equipment_routing", "迁移")]:
        fm = free.get("metrics", {}).get(scope, {})
        p(title + "：")
        table(["方法", "准确率", "全体答案可判读覆盖"],
              [[labels[m], metric(fm.get(m, {}).get("accuracy")), metric(fm.get(m, {}).get("answer_parse_coverage"))] for m in methods])
        table(["方法", "不适用病例误用判断覆盖", "误用率（可判读子集）", "全体误用率下界", "全体误用率上界"],
              [[labels[m]] + [metric(fm.get(m, {}).get(k)) for k in
                ["new_rule_misuse_judgement_coverage", "new_rule_misuse", "new_rule_misuse_lower_bound", "new_rule_misuse_upper_bound"]] for m in methods])
    table(["方法", "主测试可判读", "主测试截断", "迁移正确数/病例数"],
          [(m, sum(r.get("valid_answer", False) for r in free_predictions if r["method"] == m and r["split"] == "test"),
            sum(r.get("truncated", False) for r in free_predictions if r["method"] == m and r["split"] == "test"),
            f"{sum(r['correct'] for r in free_predictions if r['method'] == m and r['split'] == 'transfer')}/{sum(r['method'] == m and r['split'] == 'transfer' for r in free_predictions)}") for m in methods])
    p(f"该补充在dev输出格式诊断后增加，输入、方向与系数冻结，生成上限{free_protocol.get('max_new_tokens')}词元。只读取最后非空行的明确字母或完整选项；不能解析或截断未得答案按错误计准确率。")
    p("不适用病例误用判断覆盖是能判定是否用了新规则的比例；它与全体答案覆盖使用不同分母。误用率下界把不可判读项均视为未误用，上界把它们均视为误用，分母均为全部不适用病例；这两个界不是置信区间，方括号仍是家族bootstrap区间。不能把无答案解释为没有误用。严格格式解析可能拒绝语义正确的回答，未用gold修正解析。")
else:
    p("尚未通过完整输出、统计和独立验收检查，不报告计划中的准确率。计划见audit/free_generation_protocol.json；原始输出见results/free_completions.jsonl。独立验收标记audit/free_acceptance.json必须绑定预测文件SHA256。")


h("自由生成：兼容常见答案格式的补充结果")
free_norm = read("results/free_format_normalized_analysis.json", {})
norm_rows = rows("results/free_format_normalized_predictions.jsonl")
if free_norm and norm_rows:
    nmain = free_norm.get("metrics", {}).get("task:patient_routing", {})
    table(["方法", "主测试准确率", "答案可判读率", "不适用样本误用率下界", "不适用样本误用率上界"],
          [[labels[m]] + [metric(nmain.get(m, {}).get(k)) for k in
           ["accuracy", "answer_parse_coverage", "new_rule_misuse_lower_bound", "new_rule_misuse_upper_bound"]] for m in methods])
    ntransfer = free_norm.get("metrics", {}).get("task:equipment_routing", {})
    table(["方法", "迁移准确率", "迁移可判读率"],
          [[labels[m], metric(ntransfer.get(m, {}).get("accuracy")), metric(ntransfer.get(m, {}).get("answer_parse_coverage"))] for m in methods])
    p("这是同一批真实输出的事后格式兼容分析：原先末行解析把带✅等装饰的明确答案判成无效。修复仅处理常见排版并读取明确最终答案，不查询gold、不改模型或原输出。原严格解析结果完整保留。这里仍把无法确定最终答案计错，并给出误用率界限。不能把排版解析失败解释成模型逻辑错误。")
    summary_path = ROOT / "results/c_full_free_subset_summary.json"
    if summary_path.exists():
        p("实际V6模块一的同20例自由生成比较见 results/c_full_free_subset_summary.json；所有V6门控为0，因此精确复用A完成项，不产生额外模型推断。")
else:
    p("格式兼容分析尚未完成；不填写结果。")

h(f"真实V6模块一固定{cfull_expected}例补充")
if cfull_done:
    bymethod = collections.defaultdict(list)
    subset = {r["item_id"] for r in cfull}
    for row in predictions:
        if row["item_id"] in subset:
            bymethod[row["method"]].append(row)
    bymethod["C_V6"] = cfull
    table(["方法", "病例数", "正确数", "准确率"],
          [(m, len(rr), sum(r["correct"] for r in rr), pct(sum(r["correct"] for r in rr) / len(rr))) for m, rr in bymethod.items()])
    family_n = len({r["rule_family_id"] for r in cfull})
    p(f"预先按家族字典序选固定子集；实际{len(cfull)}例、{family_n}家族，{cfull_complete.get('unique_requests')}个唯一请求。实际调用Phi-4+TrialMatchAI LoRA+BioLinkBERT，保留原始V6排除条件映射和预设证据冲突门控。只作描述，不做全数据外推。")
    p("NEW状态计数为" + json.dumps(cfull_complete.get("state_counts", {}), ensure_ascii=False) +
      f"，状态判定与参考一致{cfull_acceptance.get('new_state_correct')}/{len(cfull)}。真实生成解析状态为" +
      json.dumps(cfull_acceptance.get("generation_parse_status", {}), ensure_ascii=False) +
      "；截断与未知保持原记录，未为改善成绩补写标签。")
    gate_counts = cfull_complete.get("gate_counts", {})
    zero_gates = sum(count for gate, count in gate_counts.items() if float(gate) == 0)
    p("NEW证据状态计数为" + json.dumps(cfull_complete.get("evidence_counts", {}), ensure_ascii=False) +
      f"；最终{zero_gates}/{len(cfull)}例的干预门控为0。")
    if zero_gates == len(cfull):
        same = cfull_acceptance.get("same_scores_as_A_exact") is True
        p("本次C_V6全部关闭干预。" + ("这些病例的所有选项分数与A逐值相同。" if same else "不能把该零干预结果当作有效干预的比较。") +
          "因此这组结果不构成动态干预有效的证据；它暴露的是当前模型链路在这组输入上的证据门控问题。")
    p("无关句被NLI判为支持或矛盾、冻结V6的保守排除条件映射，以及截断传播为未知的具体实例见sources/c_full_review.md。该补充与C_reduced还存在证据关闭策略差异，不能把全部变化只归因于适用性预测。")

else:
    p(f"计划在固定{cfull_expected}例上验证实际模型链路，尚未验收完整输出；不能用简化门控结果代替。")

h("计算开销与复现")
run_table = []
for filename, label in [("facts_complete.json", "事实筛选+方向+dev+test"), ("rules_complete.json", "键空间dev+四组约束评分"), ("free_complete.json", "自由生成补充")]:
    rr = read("results/" + filename, {})
    if rr.get("status") == "complete":
        memory = rr.get("gpu_peak_allocated_bytes", rr.get("peak_allocated_bytes", rr.get("peak_memory_bytes")))
        run_table.append((label, f"{rr['total_seconds']:.2f}秒", f"{memory / 2 ** 30:.2f} GiB" if memory is not None else "未记录"))
if cfull_done:
    memory = cfull_complete.get("overall_peak_allocated_bytes")
    run_table.append(("实际V6小子集", f"{cfull_complete['actual_elapsed_seconds']:.2f}秒", f"{memory / 2 ** 30:.2f} GiB" if memory is not None else "未记录"))
table(["阶段", "真实整阶段时间", "进程累计峰值显存"], run_table)
p("各阶段GPU配置保存在启动脚本和环境记录；当前模型环境记录的设备为" + str(environment.get("device", "未记录")) + "。单词元score时间不含输入tokenization，不能称完整端到端延迟。批量自由生成的均摊时间只表示吞吐成本。相同输入及相同干预的输出复用不计作新增生成；独立部署各方法的成本与本次复用后的总成本应区分。")
p("复现入口见REPRODUCE.txt。保存内容包含冻结输入、逐例输出、种子、环境、模型文件摘要、方向、Key空间、dev网格和统计审查。附包不包含模型权重、依赖二进制或完整外部论文/仓库；原服务器sources保留核验副本，官方链接和commit用于重新获取。")
h("仍未完成，不能据此声称")
p(f"原始CARE上传文件未读取；完整D未复现；C_V6是固定子集，未跑全部{len(cases)}例；没有多模型、多方向训练种子、真实患者验证、真实参数旧规则回退或不同逻辑模板迁移。应先恢复原文件并锁定版本，再扩大闭卷合格样本与模板隔离。当前证据不能证明来源偏好与患者适用性在所有模型中必然分离。")

style = "body{font-family:system-ui,sans-serif;max-width:1150px;margin:40px auto;padding:0 24px;color:#172033;line-height:1.65;background:#fcfcfd}h1{font-size:30px}h2{margin-top:36px;font-size:21px;color:#163c62}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;border:1px solid #dce1e9;text-align:left}th{background:#eaf1f7}.scroll{overflow:auto}a{color:#0b5bad}.meta{color:#667085}"
content = "<!doctype html><html lang=zh-CN><meta charset=utf-8><title>CARE方向干预最小实验报告</title><style>" + style + "</style><h1>CARE方向干预：最小实测报告</h1><p class=meta>2026-10-19 · 单模型方法适配 · 合成规则研究 · 不构成临床验证</p>" + "\n".join(sections) + "</html>"
(ROOT / "REPORT.html").write_text(content)
(ROOT / "REPORT.txt").write_text("CARE方向干预最小实测报告\n2026-10-19\n" + "\n".join(plain))
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    panels = ([(analysis, "Constrained option scores")] if main else []) + ([(free_norm, "Free generation: format tolerant")] if free_norm else [(free, "Free generation: strict parser")] if free else [])
    if panels:
        fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4.5), squeeze=False)
        for ax, (data, title) in zip(axes[0], panels):
            ms = data["metrics"]["task:patient_routing"]
            values = [ms[m]["accuracy"]["estimate"] * 100 for m in methods]
            ci = np.array([ms[m]["accuracy"]["ci95"] for m in methods]) * 100
            ax.bar(range(len(methods)), values, color=["#3d647e", "#ce7d46", "#579b86", "#9b9eaa"], width=.65)
            ax.errorbar(range(len(methods)), values, yerr=np.maximum(0, np.array([np.array(values) - ci[:, 0], ci[:, 1] - np.array(values)])), fmt="none", color="#172033", capsize=4)
            ax.set_xticks(range(len(methods)), ["A", "B", "C reduced", f"D ({epsilon})"])
            ax.set_ylim(0, 105)
            ax.set_ylabel("Accuracy (%)")
            ax.set_title(title)
            ax.spines[["top", "right"]].set_visible(False)
            for i, value in enumerate(values):
                ax.text(i, value + 3, f"{value:.1f}", ha="center", fontsize=10)
        model_name = Path(environment.get("model", "model")).name
        fig.suptitle(f"{model_name} | {main_count} synthetic test cases | {main_families} rule families")
        fig.tight_layout()
        fig.savefig(ROOT / "results/test_accuracy.png", dpi=200)
        fig.savefig(ROOT / "results/test_accuracy.svg")
        plt.close(fig)
except ImportError:
    (ROOT / "logs/plot_limitation.txt").write_text("matplotlib unavailable; no figure generated")
print("REPORT_SAVED")
