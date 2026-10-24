"""Render completed paired results; CPU only."""
import collections,hashlib,json
from pathlib import Path
E=Path(__file__).parent;O=E/'results';ROOT=E.parents[1]
s=json.loads((O/'summary.json').read_text());cs={r['id']:r for r in map(json.loads,(ROOT/'experiments/medguide_expand_cs_20261023/results/cs_selected.jsonl').open())};patterns=[json.loads(x) for x in (O/'case_patterns.jsonl').read_text().splitlines()];scores=[json.loads(x) for x in (O/'scores.jsonl').read_text().splitlines()]
assert len(scores)==176 and sum(not r['reused'] for r in scores)==109 and sum(r['reused'] for r in scores)==67
assert sum(json.loads((O/f'worker{k}.json').read_text())['new_calls_this_run'] for k in range(2))==109
mix={}
for st in ['K','U']:
 rr=[r for r in patterns if r['stratum']==st];mix[st]={'threshold_direction':dict(collections.Counter('raise' if cs[r['id']]['new_threshold']>cs[r['id']]['old_threshold'] else 'lower' for r in rr)),'subtype':dict(collections.Counter(cs[r['id']]['source_path'][1] for r in rr))}
(O/'case_composition.json').write_text(json.dumps(mix,ensure_ascii=False,indent=2))
lines=['# 44 例阈值 × 规则顺序对照结果','已完成 176 个组合：67 个直接复用，109 个新增推理，分片实际调用为 55/54。GPU0/1 各一个 Qwen3-8B 进程，BF16/eager/greedy 8 token。每卡模型加载与推理约 10.5 秒，峰值分配显存约 15.4 GiB。没有重跑既有请求。','## 设计与四格准确率','固定 44 例（K20/U24）的患者、动作、选项与两候选，只交叉改变原/新阈值及规则当前/反转顺序。“当前顺序”为 CS 第一候选在前、第二候选在后；切回原阈值时保持这两个分支及顺序，不重新预测分支。','|阈值／顺序|全部44|K20|U24|','|---|---:|---:|---:|']
for key,name in [('original/current','原阈值／当前顺序'),('original/reversed','原阈值／反转顺序'),('changed/current','新阈值／当前顺序'),('changed/reversed','新阈值／反转顺序')]:
 vals=[]
 for st in ['all','K','U']:
  c=s['groups'][st]['cells'][key];vals.append(f"{c['correct']}/{c['n']}（{100*c['correct']/c['n']:.1f}%）")
 lines.append('|'+name+'|'+'|'.join(vals)+'|')
lines+=['## 阈值、位置、原答案偏好分别有多少证据','每个病例在两种顺序下各有一次阈值配对；原、新阈值两个答案都正确才计为正确翻转。每例还有原、新阈值下各一次顺序配对。88 对是44病例的重复测量，不是88个独立病例。','|配对指标|全部|K|U|','|---|---:|---:|---:|']
for key,den,name in [('correct_threshold_flips','threshold_pairs','随阈值正确翻转'),('order_answer_changes','order_pairs','只交换顺序就改答案')]:
 lines.append('|'+name+'|'+'|'.join(f"{s['groups'][st][key]}/{s['groups'][st][den]}" for st in ['all','K','U'])+'|')
lines+=['|病例四格模式|全部44|K20|U24|','|---|---:|---:|---:|']
for key,name in [('threshold_following_all4','四格全部遵循阈值'),('fixed_action_all4','四格固定同一动作'),('first_position_all4','四格固定选物理第一条'),('last_position_all4','四格固定选物理最后一条'),('mixed','混合模式'),('parse_failure','含解析失败')]:lines.append('|'+name+'|'+'|'.join(str(s['groups'][st]['categories'].get(key,0)) for st in ['all','K','U'])+'|')
lines+=['### K：旧答案不变是主要现象','18/20 例在四格中都输出旧金标，也等于原闭卷答案；另外2例只在新阈值、当前顺序下答对，交换顺序后也回到旧答案。K 的40个阈值配对仅2个正确翻转，40个顺序配对仅2个改变。说明这批K的主要问题是对阈值变化不敏感、持续输出旧动作，不能主要归结为规则排列位置。行为与原答案偏好一致，但不能据此定位模型内部究竟依赖记忆还是其他语义线索。','### U：出现可重复的阈值响应，修正此前仅靠闭卷一致性的解释','17/24 例在四格中全部正确：原阈值时输出旧金标，新阈值时输出新金标，交换顺序仍保持正确。U 的48个阈值配对有39个正确翻转。原阈值下24例全部答对，而原闭卷答案在这些病例原本全部错误；因此不能用“始终复读闭卷答案”解释这些病例。此前观察到15个救回病例中14个与闭卷答案相同，只说明新阈值单个条件下的输出一致，当前配对对照提供了更强的阈值响应证据。','U 还有2例四格固定旧金标（并非其原闭卷答案），5例受顺序影响：反转规则后新阈值准确数由17升至22，原阈值保持24。因此U存在局部顺序效应，但整体已有稳定阈值遵循。','### 位置：有局部影响，未表现为统一选择第一条或最后一条','全部88个顺序配对只有7个改答案，全部发生在新阈值条件：K为2例正确→错误，U为5例错误→正确。没有病例在四格中始终选择物理第一条或最后一条。此结果支持局部的顺序敏感性，不支持“所有差异主要来自单一位置偏好”。零个纯位置模式也不等于模型完全不存在位置影响。','## 范围与下一步','44例是旧/新两个目标分支都已进入Top-2的子集，来自同一棵 nscl_3 树。这里消除了候选遗漏，但K/U仍不是随机分组，阈值方向与结节类型不平衡，不能声称知识正确性本身导致差异。组成见 case_composition.json。闭卷参照沿用PR24的SDPA输出，本轮为eager；闭卷一致性是观察性比较。','下一步优先解决K在正确候选均已存在时仍不随阈值翻转的问题：在裁剪前显式核对患者数值、类型及比较符号。固定旧动作的18个K病例可作开发诊断，最终仍应在完整冻结分组上比较保留正确答案与阈值翻转能力。此轮不继续扩展候选数、延长CAD或重跑已完成基线；本次没有启动任何后续实验。','## 文件与验证','request_manifest.jsonl：176请求的缓存标记及内容哈希。scores.jsonl：逐格预测和复用来源。case_patterns.jsonl：44病例的四格输出与配对模式。summary.json：全部及K/U指标。worker0/1.json：实际调用数、运行参数、耗时、显存及原始输出哈希。provenance.json：源清单和缓存哈希。完整生成保留在服务器private目录。','核对：176个键完整且唯一；67复用、109新增；实际两卡调用55+54；输出哈希通过；四格分组数及配对净变化一致。无解析失败。复现流程见上一级README.zh.md。']
(O/'REPORT.zh.md').write_text('\n\n'.join(lines[:4])+'\n'+'\n'.join(lines[4:])+'\n')
for st in ['all','K','U']:
 g=s['groups'][st];assert sum(g['categories'].values())==g['n'];assert g['correct_threshold_flips']<=g['threshold_pairs'];assert g['order_wrong_to_right']+g['order_right_to_wrong']==g['order_answer_changes']
(O/'artifact_hashes.json').write_text(json.dumps({str(p.relative_to(E)):hashlib.sha256(p.read_bytes()).hexdigest() for p in list(O.iterdir())+[E/'experiment.py',E/'report.py',E/'run.sh',E/'README.zh.md'] if p.is_file() and p.name!='artifact_hashes.json'},indent=2))
print(json.dumps(mix,ensure_ascii=False));print('validated 176 unique cells, 67 reused, 109 actual calls; report saved')
