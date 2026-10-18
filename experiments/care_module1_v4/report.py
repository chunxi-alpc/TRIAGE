"""Report observed metrics without changing frozen labels, prompts or predictions."""
import collections,json,pathlib
root=pathlib.Path('.');d=root/'data';m=json.loads((d/'metrics.json').read_text());audit=json.loads((d/'dataset_audit.json').read_text());env=json.loads((d/'model/environment.json').read_text());read=lambda p:[json.loads(s) for s in p.read_text().splitlines()]
labels=read(d/'labels.jsonl');pred={r['sample_id']:r for r in read(d/'model/predictions.jsonl')};records={r['sample_id']:r for r in read(d/'model/normalized_records.jsonl')}
names={'direct':'直接判定','structured_no_guard':'结构化、不核对证据','structured':'修复后的完整结构化'};v=m['methods'];full=v['structured'];direct=v['direct'];ng=v['structured_no_guard']
lines=['CARE 模块一 v4 实测报告','',f'实际完成：服务器 Qwen3-8B，{audit["new_rules"]} 条新规则、{audit["new_cases"]} 个新原病例、{m["n"]} 个样本、32 对反事实。推理于 {env["start_utc"]} 至 {env["end_utc"]}（UTC）执行。没有填写未运行的模型结果。','',
f'完整结构化相对直接判定：正确数变化 {full["correct"]-direct["correct"]:+d}/192，宏 F1 变化 {full["macro_f1"]-direct["macro_f1"]:+.4f}，判定和证据同时正确变化 {full["joint_label_and_evidence_correct"]-direct["joint_label_and_evidence_correct"]:+d}/192。',
f'完整结构化相对不核对证据：正确数变化 {full["correct"]-ng["correct"]:+d}/192，同时正确变化 {full["joint_label_and_evidence_correct"]-ng["joint_label_and_evidence_correct"]:+d}/192。','',
'方法 | 判定正确 | 宏 F1 | 已知样本证据正确 | 全部样本证据正确 | 判定和证据同时正确 | 反事实成对正确 | 推理秒数']
for k,n in names.items():
 x=v[k];lines.append(f'{n} | {x["correct"]}/192 ({x["accuracy"]:.2%}) | {x["macro_f1"]:.4f} | {x["evidence_known_correct"]}/{x["known_n"]} ({x["evidence_known_accuracy"]:.2%}) | {x["evidence_all_correct"]}/192 | {x["joint_label_and_evidence_correct"]}/192 ({x["joint_accuracy"]:.2%}) | {x["pair_correct"]}/32 | {x["generation_seconds"]:.1f}')
lines+=['','分组：每组 32 条；按直接 / 不核对 / 完整结构化顺序列出判定正确数。']
for typ in ['original','counterfactual','partial_missing','multiple_values','negation','contradiction']:lines.append(typ+'：'+' / '.join(str(v[k]['by_variant'][typ]['correct']) for k in names))
lines+=['','证据评价与统计：','证据须覆盖一个充分的参考证明，且没有无关引用；允许运行前标注的替代证据。缺失未知按约定空引用，矛盾未知引用两份报告。引用位置正确不等于语义或标签正确。']
for k,n in names.items():
 x=v[k];lines.append(f'{n}：引用支持率 {x["supported_citation_count"]}/{x["citation_count"]}，矛盾证据 {x["conflict_evidence_correct"]}/32，已知误判未知 {x["known_to_unknown"]}/{x["known_n"]}。')
for unit,z in m['bootstrap'].items():lines.append(f'按 {unit} 的 {z["clusters"]} 个组自助抽样 2000 次：完整减直接的宏 F1 差 95% 区间 {z["f1_difference_ci95"]}，同时正确率差区间 {z["joint_difference_ci95"]}。组数小，区间仅供探索性判断。')
lines+=['',f'规则结构归一化准确 {m["rule_extraction"]["exact_normalized"]}/16；不匹配规则：{m["rule_extraction"]["mismatches"]}。没有用参考规则替换模型输出。',
f'证据核对相对不核对：纠正 {m["paired"]["structured_vs_structured_no_guard"]["rescued"]} 个判定，同时引入 {m["paired"]["structured_vs_structured_no_guard"]["introduced"]} 个错误。',
'定位处理统计：'+json.dumps(m['verification_audit'],ensure_ascii=False),'','本轮修复：',
'1. 统一编号、数字字符串、对象引用与原文引用格式；小数不能错误取整为句号，分类字段不能把 1/-1 或实验室数值当阴阳性。',
'2. 允许区间和开闭边界；模型给出的原文片段必须能在病例找到，错误句号可通过精确片段重新定位。',
'3. 删除字段附近必须出现特定词的拒绝规则。增加一次模型全文核对，处理主体、时间、否定和冲突；已知事实需要真实原文片段，冲突保留两份证据。',
'两种结构化方案共享格式处理、规则提取、首轮事实和确定性逻辑；无核对组不以证据有效性否决事实。完整组额外使用全文核对和原文片段约束。不是仅改一个开关的旧版同集比较，也不是独立模型裁判。']
errors=[]
from build_dataset import atom_truth
def entails(g,p):
 if g is None or p is None:return False
 if isinstance(p,dict):return all(atom_truth({'op':o,'value':v},g)=='T' for o,v in p.items())
 return atom_truth({'op':'eq','value':p},g)=='T'
field_diag={}
for arm in ['unverified','verified']:
 n=correct=supported=0
 for x in labels:
  rec=records[x['sample_id']][arm]
  for f,p in rec['facts'].items():
   if p is None:continue
   n+=1;good=entails(x['gold_observations'].get(f),p);correct+=good
   ids=set(rec['evidence'].get(f,[]));groups=x['field_evidence_groups'].get(f,[]);allowed={i for group in groups for i in group}
   supported+=good and bool(groups) and any(set(group)<=ids for group in groups) and ids<=allowed
 field_diag[arm]={'non_null_assertions':n,'reference_entailed_values':correct,'correct_value_and_field_evidence':supported,'supported_assertion_precision':supported/n if n else None}
field_diag['note']='Supplementary post-run diagnostic; does not replace pre-frozen primary metrics. Direct arm does not emit per-field values, so it is not included.'
(d/'field_diagnostics.json').write_text(json.dumps(field_diag,indent=2)+'\n')
parse_failures={stage:sum(not isinstance(z.get('parsed'),dict) for z in read(d/f'model/{stage}_raw.jsonl')) for stage in ['rule_extraction','direct','facts','verification']}
lines+=['','补充字段诊断（不替代预先设定的主指标）：',json.dumps(field_diag,ensure_ascii=False),'未能解析为 JSON 对象的模型输出数：'+json.dumps(parse_failures,ensure_ascii=False)]
for x in labels:
 p=pred[x['sample_id']];rec=records[x['sample_id']]
 if p['structured']['label']!=x['label']:
  diffs={f:{'gold':g,'unverified':rec['unverified']['facts'].get(f),'verified':rec['verified']['facts'].get(f)} for f,g in x['gold_observations'].items() if rec['verified']['facts'].get(f)!=g}
  errors.append({'sample_id':x['sample_id'],'rule_id':x['rule_id'],'variant':x['variant'],'gold_label':x['label'],'predicted_label':p['structured']['label'],'field_differences':diffs,'rule_mismatch':x['rule_id'] in m['rule_extraction']['mismatches']})
(d/'remaining_errors.json').write_text(json.dumps(errors,indent=2)+'\n')
from grounding import normalize
from logic import evaluate
from specs import NUMERIC
raw_verified={r['id']:r['parsed'] for r in read(d/'model/verification_raw.jsonl')}
raw_rules={r['id']:r['parsed'] for r in read(d/'model/rule_extraction_raw.jsonl')}
source_inputs={r['sample_id']:r for r in read(d/'private/inputs.jsonl')}
categories=collections.Counter();introduced_categories=collections.Counter();classified=[]
for x in labels:
 sid=x['sample_id'];p=pred[sid]
 if p['structured']['label']==x['label']:continue
 if p['structured']['label']=='INVALID':category='unsupported_rule_format'
 else:
  fs=list(records[sid]['verified']['facts']);ungated=normalize(raw_verified[sid],source_inputs[sid]['sentences'],fs,NUMERIC,False)
  category='evidence_gate_lost_correct_decision' if evaluate(raw_rules[x['rule_id']],ungated['facts'])==x['label'] else 'extraction_or_verification_semantic_error'
 categories[category]+=1
 introduced=p['structured_no_guard']['label']==x['label']
 if introduced:introduced_categories[category]+=1
 classified.append({'sample_id':sid,'category':category,'introduced_by_verification':introduced})
(d/'error_categories.json').write_text(json.dumps({'all_remaining':dict(categories),'introduced':dict(introduced_categories),'cases':classified,'note':'Post-run diagnostic, frozen primary predictions unchanged.'},indent=2)+'\n')
lines+=['','完整结构化剩余错误：',f'共 {len(errors)} 条，逐例字段差异见 data/remaining_errors.json；所有组的判定或证据错误见 data/errors.jsonl。']
lines+=['错误分类：'+json.dumps(dict(categories),ensure_ascii=False),'新增错误分类：'+json.dumps(dict(introduced_categories),ensure_ascii=False),'V01 把 negative 放入 op 且缺少 value；V06 把 positive 放入 op。两种格式不在冻结的规则协议内，导致共 24 条 INVALID，全部计错，未删除或用参考结构代替。其余常见错误是第二次核对恢复了已经撤回的旧值、把缺失当阴性、把新旧记录误当尚未解决的矛盾，以及引文不匹配后将正确事实置为未知。']
for e in [e for e in errors if e['predicted_label']!='INVALID'][:10]:lines.append(f'{e["sample_id"]}：参考 {e["gold_label"]}，预测 {e["predicted_label"]}；字段差异 '+json.dumps(e['field_differences'],ensure_ascii=False))
lines+=['','数据和限制：',
f'此前 173 个原病例、46 条规则均排除，行号、原文哈希、规则来源节点哈希和标准化规则签名重叠均为零。对旧病例最大三词组 Jaccard 相似度 {audit["max_trigram_jaccard_to_previous"]:.3f}。不保证模型预训练未见过数据，也不表示每个原子条件都是新概念。',
f'32 个原病例完整保留；160 个控制样本通过更正补充说明明确撤回目标旧信息。所有样本保留原病例全部文字，原病例平均 {audit["word_counts"]["original"]["mean"]:.1f} 词，范围 {audit["word_counts"]["original"]["min"]}–{audit["word_counts"]["original"]["max"]} 词；多值样本平均 {audit["word_counts"]["multiple_values"]["mean"]:.1f} 词。部分缺失组含 {audit["partial_multifield"]} 个多条件样本，其中 {audit["partial_decisive"]} 个可由其他条件确定。',
'来源病例本身为合成病例，不是临床患者数据；控制修改有统一结构，不能排除对模板的适应。原病例往往已符合指南路径，分布偏向适用。标签由 Codex 阅读原文和明确修改产生，未做临床专家独立复核。证据参考组合可能未覆盖所有合法表述。',
'16 条规则集中于可解析条件；部分为原指南分支投影。这里只验证规则谓词，不验证完整治疗方案。完整方案使用第二次模型推理，若结果更好，也不能排除额外计算与再提取本身的贡献。',
f'本轮 GPU {env["visible_devices"]}：{env["gpu"]}，峰值分配 {env["peak_allocated_gib"]:.2f} GiB、预留 {env["peak_reserved_gib"]:.2f} GiB。加载 {env["model_load_seconds"]:.2f} 秒。表中耗时为 CUDA 同步后的批量生成；结构化包含一次规则提取，完整方案另包含核对，均不含加载、分词和文件 I/O。',
'模型、源病例、完整原文引用不上传；服务器保留未修改的原始输出。公开结果包包含其哈希、去除引用后的记录、全部指标和可执行复现脚本。详见 README.zh.txt。','',
'结论和后续：']
if full['correct']>direct['correct'] and full['joint_label_and_evidence_correct']>direct['joint_label_and_evidence_correct']:
 lines.append('本次只能认为部分改善，尚不能证明同时提高判定准确率与整体证据可靠性。虽然整体判定和联合指标的点估计提高，已知样本证据定位、全部样本证据正确数仍低于直接判定；未经修改的完整病例也只有 26/32 正确，低于直接判定的 28/32。')
else:lines.append('本次固定的新样本没有同时证明判定正确率和证据可靠性提高；应保留这个结果，不能仅用格式修复成功宣称结构化方案有效。')
if full['correct']<ng['correct']:lines.append('完整方案判定正确数低于无核对组，说明核对仍可能误伤正确抽取；应优先分析新增错误。')
lines.append('下一步最有价值的是加入相同推理预算的“直接判定后再核对”对照，再用自然存在的缺失、时间冲突和否定病例做独立验证，并对少量争议标签做临床复核。不要在本测试集继续调提示后把结果作为新验证。')
if min(m['bootstrap']['rule_id']['f1_difference_ci95'])<=0 or min(m['bootstrap']['rule_id']['joint_difference_ci95'])<=0:
 lines.append('至少一项主差值的按规则分组区间包含零，不能声称已统计证实两方面都有稳定提升。')
lines+=['','与上一轮的关系：上一轮完整结构化宏 F1 为 0.6190，主要问题是引用格式和过强的词面拒绝规则。本轮使用全新的数据与新实现，不能将两个数值相减当作同一测试集上的修复收益。旧缓存的四个定位例子见 provenance/old_format_regression.json：对象编号可恢复，但没有原文片段时，单靠格式转换不能修复引用错句；这也是本轮增加全文证据核对的原因。',
'控制样本的数值按条件边界生成，可能含零值或与保留病史不自然的组合。明确更正使目标条件标签可确定，但不保证整份人工修改病例在临床上自然一致；结论仍限于这套规则压力测试。']
(root/'REPORT.zh.txt').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:13]))
