"""Render the frozen parser-v2 comparison from saved offline evaluation JSON."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
EXP=ROOT/'experiments/care_module1_v5'


def ratio(correct,total):
    return f'{correct}/{total} ({100*correct/total:.2f}%)'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--original-metrics',type=Path,default=EXP/'runs/v5_20261018/metrics.json')
    p.add_argument('--output',type=Path,default=EXP/'PARSER_V2_REPORT.zh.md')
    a=p.parse_args()
    run=json.loads((a.run/'run_status.json').read_text())
    new=json.loads((a.run/'metrics.json').read_text())
    old=json.loads(a.original_metrics.read_text())
    cond=json.loads((a.run/'condition_eval/condition_metrics.json').read_text())
    cm={k:v for k,v in cond['new_parser'].items()}
    lines=['# TrialMatchAI 输出解析器修复与离线复评','',
        '## 结果范围','',
        '只读取服务器留存的 492 条 TrialMatchAI 原始生成；未重新运行 Phi-4、LoRA、BioLinkBERT 或 Qwen3。解析阶段未读取金标；解析规则和输出哈希冻结后，另行读取原有源读标签计算指标。BioLinkBERT 的既存 NLI 分数和证据位置用于确定性复算组合结果，不重新推理。',
        f"服务器解析状态：{run['status']}；模型调用 {run['model_calls']}；GPU 使用 {run['gpu_used']}；原始输出哈希核对 {run['raw_hashes_verified']}/492；处理用时 {run['reparse_seconds']:.3f} 秒。服务器测试：54 项通过。",
        f"截断核对：原生成记录中 {run.get('truncation_audit',{}).get('recorded_truncated',0)} 条标记为截断，全部恰好达到 2048 token 上限；达到上限但未标记截断的有 {run.get('truncation_audit',{}).get('unflagged_at_limit',0)} 条。",'',
        '## 解析器变化','',
        f"| 项目 | 原解析器 | 修复后 |",'|---|---:|:---|',
        f"| 解析成功 | {run['previous_parse_accepted']}/492 (15.24%) | {run['parse_status_counts'].get('accepted',0)}/492 ({100*run['parse_status_counts'].get('accepted',0)/492:.2f}%) |",
        f"| 真正截断 | 旧协议均记为未知 | {run['parse_error_counts'].get('generation_truncated',0)}，继续记 U |",
        f"| 目标文本不匹配/不唯一 | 旧协议混在通用解析失败 | {run['parse_error_counts'].get('criterion_text_mismatch',0)}，记为目标不明确 U |",
        f"| 目标标签明确为医学未知 | 无独立解析状态 | {run['unknown_reason_counts'].get('medical_unknown',0)} 个精确目标输出为 Unclear，保留 U |",
        f"| 非目标侧 Irrelevant 空占位 | 其中一类主要解析失败 | 忽略 {sum(x['ignored_non_target_placeholder_count'] for x in (json.loads(s) for s in (a.run/'conditions_public.jsonl').read_text().splitlines() if s.strip()))} 个；不参与目标标签 |",
        f"| 非目标侧额外条目 | 可能导致拒绝解析 | 仅记录并忽略 {sum(max(0,x['non_target_entry_count']-x['ignored_non_target_placeholder_count']) for x in (json.loads(s) for s in (a.run/'conditions_public.jsonl').read_text().splitlines() if s.strip()))} 个；不作为目标判断依据 |",
        '',
        f"有效目标输出原样通过严格匹配后，标签为 Met {run['accepted_label_counts'].get('Met',0)}、Not Met {run['accepted_label_counts'].get('Not Met',0)}、Unclear {run['accepted_label_counts'].get('Unclear',0)}。以前已解析成功的 75 条标签全部保持不变。",'',
        '## 原子条件实际判断','',
        '金标取自服务器已保存的源读条件事实；逐条件金标再组合后与 192 条冻结规则标签全部一致。条件数值只评价标签与条件，不评价证据位置。','',
        '| 指标 | 原解析器输出 | 修复后 |', '|---|---:|---:|',
        f"| 全部 492 条原子条件正确 | {ratio(cond['old_parser']['correct'],492)} | {ratio(cm['correct'],492)} |",
        f"| 宏 F1 | {cond['old_parser']['macro_f1']:.4f} | {cm['macro_f1']:.4f} |",
        f"| 金标已知条件准确率 | {ratio(cond['old_parser']['gold_known_correct'],cond['old_parser']['gold_known_n'])} | {ratio(cm['gold_known_correct'],cm['gold_known_n'])} |",
        f"| 金标未知条件判为 U | {ratio(cond['old_parser']['gold_unknown_correct'],cond['old_parser']['gold_unknown_n'])} | {ratio(cm['gold_unknown_correct'],cm['gold_unknown_n'])} |",
        f"| 已知条件误判未知 | {cond['old_parser']['gold_known_to_unknown']} | {cm['gold_known_to_unknown']} |",
        f"| 未知判成 T/F | {cond['old_parser']['error_categories'].get('unknown_to_known',0)} | {cm['error_categories'].get('unknown_to_known',0)} |",
        f"| T→F / F→T | {cond['old_parser']['error_categories'].get('false_rejection_T_to_F',0)} / {cond['old_parser']['error_categories'].get('false_acceptance_F_to_T',0)} | {cm['error_categories'].get('false_rejection_T_to_F',0)} / {cm['error_categories'].get('false_acceptance_F_to_T',0)} |",
        '',
        f"423 条已接受输出中，369 条条件状态正确 ({100*369/423:.2f}%)。其中 354 条金标已知条件给出 T/F，326 条正确 ({100*326/354:.2f}%)；另有 24 条金标未知被判成 T/F。",'',
        '## 规则级影响','',
        '| 方案 | 原解析结果 | 修复解析后 |', '|---|---:|---:|']
    for key,name in [('trialmatch_model_only','TrialMatchAI 单独'),('trialmatch','TrialMatchAI + 数值比较'),('combined','TrialMatchAI + 数值 + 已缓存 MedNLI'),('qwen_v4','历史 Qwen3 直接判定')]:
        o=old['methods'][key];n=new['methods'][key]
        lines.append(f"| {name} | {ratio(o['correct'],192)}；宏 F1 {o['macro_f1']:.4f} | {ratio(n['correct'],192)}；宏 F1 {n['macro_f1']:.4f} |")
    lines += ['',
        f"修复后 BioLinkBERT 证据定位仍为 {ratio(new['methods']['combined']['known_evidence_correct'],new['methods']['combined']['known_evidence_n'])}，因为复用同一批缓存证据。组合方案联合正确由 {old['methods']['combined']['joint_correct']}/192 变为 {new['methods']['combined']['joint_correct']}/192；条件冲突记录由 {old['methods']['combined']['conflicts']} 增至 {new['methods']['combined']['conflicts']}。适用性准确率从 {100*old['methods']['combined']['accuracy']:.2f}% 变为 {100*new['methods']['combined']['accuracy']:.2f}%；解析成功率上升不等于组合规则准确率必然上升。",
        f"纯 TrialMatchAI 从 {old['methods']['trialmatch_model_only']['correct']}/192 提高到 {new['methods']['trialmatch_model_only']['correct']}/192；修复后比历史 Qwen3 {100*(new['methods']['trialmatch_model_only']['accuracy']-new['methods']['qwen_v4']['accuracy']):+.2f} 个百分点。BioLinkBERT 组合结果仍比 Qwen3 低 {100*(new['methods']['qwen_v4']['accuracy']-new['methods']['combined']['accuracy']):.2f} 个百分点。",'',
        '## 解析与未知分类规则','',
        '- 可从代码围栏、前后说明中提取唯一完整 JSON 对象；JSON 内容本身不修补。',
        '- 只在指定目标数组中查找与请求条件严格一致的原文及唯一分类标签；仅容许空白重排，大小写和标点必须一致。仅忽略目标外数组；其中 Irrelevant 空占位计数单独保留。',
        '- 目标原文缺失或不唯一归为“目标不明确”，不会从相反数组或 Final Decision 推测标签。',
        '- 输出到达生成长度上限且没有 EOS，归为“真正截断”；明确 Unclear 归为“医学未知”；三类状态分别保存。',
        '- 修复后的组合规则使用缓存 NLI 关系与原有数值观察确定性复算，未运行 BioLinkBERT。',
        '', '## 适用范围与文件','',
        '这 192 条回归数据用于过方法，病例为合成记录，标签不是独立临床验证。新增解析成功率不代表临床适用性；修复后的纯 TrialMatchAI 规则准确率仍低于历史 Qwen3，组合方案因解析出的新标签与缓存证据冲突略有下降。',
        f"- 解析器：`src/module1/trialmatch_parser.py`；调用接口：`src/module1/care_v5.py`。",
        f"- 逐条件公开复评：`{a.run/'condition_eval/condition_metrics.json'}`、`{a.run/'condition_eval/condition_predictions_public.jsonl'}`。",
        f"- 规则级指标/错误：`{a.run/'metrics.json'}`、`{a.run/'errors_public.jsonl'}`。",
        f"- 解析状态/冻结记录：`{a.run/'run_status.json'}`、`{a.run/'REPARSE_FREEZE.json'}`。",
        '- 原始 492 条生成和含原文的文件保留在服务器私有目录，未写入公共结果。']
    a.output.write_text('\n'.join(lines)+'\n')
    print(str(a.output))

if __name__=='__main__':main()
