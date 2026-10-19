"""Evaluate the frozen new synthetic set without historical Qwen predictions."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('v5_evaluate_core',BASE.parent/'scripts/evaluate.py')
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)


def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--predictions',type=Path,required=True)
 parser.add_argument('--output',type=Path,required=True)
 args=parser.parse_args()
 labels=core.read_jsonl(BASE/'data/labels_public.jsonl');pairs=core.read_jsonl(BASE/'data/pairs_public.jsonl')
 core.unique_index(labels,'exploratory labels')
 if len(pairs)!=4 or len({p['pair_id'] for p in pairs})!=4:raise ValueError('Expected 4 frozen unique exploratory pairs')
 label_by_id={x['sample_id']:x for x in labels}
 for pair in pairs:
  if label_by_id[pair['original_id']]['label']!='T' or label_by_id[pair['counterfactual_id']]['label']!='F':raise ValueError('Invalid exploratory counterfactual pair')
 predictions=core.unique_index(core.read_jsonl(args.predictions),'new exploratory predictions')
 methods={m:core.method_metrics(labels,predictions,m,pairs) for m in ['trialmatch_model_only','trialmatch','combined']}
 errors=[]
 for label in labels:
  if label['sample_id'] not in predictions:continue
  prediction=predictions[label['sample_id']]
  for method in methods:
   actual=core.predicted_state(prediction,method)
   evidence=core.evidence_correct_ids(core.all_evidence_ids(prediction,method),label) if method!='trialmatch_model_only' else None
   if actual!=label['label'] or evidence is False:
    errors.append({'sample_id':label['sample_id'],'rule_id':label['rule_id'],'variant':label['variant'],'method':method,'gold':label['label'],'predicted':actual,'evidence_correct':evidence,'predicted_evidence_ids':sorted(core.all_evidence_ids(prediction,method)) if evidence is not None else None,'sufficient_gold_evidence':label['acceptable_evidence_proofs'],'conflict_evidence_ids':label['conflict_evidence_ids']})
 comparison={}
 if all(m['status']=='complete' for m in methods.values()):
  comparison={'combined_vs_model_only_corrected':sum(core.predicted_state(predictions[x['sample_id']],'trialmatch_model_only')!=x['label'] and core.predicted_state(predictions[x['sample_id']],'combined')==x['label'] for x in labels),'combined_vs_model_only_damaged':sum(core.predicted_state(predictions[x['sample_id']],'trialmatch_model_only')==x['label'] and core.predicted_state(predictions[x['sample_id']],'combined')!=x['label'] for x in labels)}
 args.output.mkdir(parents=True,exist_ok=True)
 report={'evaluated_utc':datetime.now(timezone.utc).isoformat(),'protocol':'24 new synthetic exploratory logic probes; 4 new complete-tree rules; no independent clinical gold','methods':methods,'combined_change':comparison,'qwen_comparison':'not available on these new synthetic inputs; no Qwen was called','source_freeze':str(BASE/'data/FREEZE.json'),'predictions':str(args.predictions)}
 (args.output/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 (args.output/'errors.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in errors))
 lines=['# 新合成探索集结果','', '4 条此前未使用的完整规则树，24 条自编明确事实；这些是逻辑压力测试，不是独立临床验证。','', '| 方法 | 正确数 | 准确率 | 宏 F1 | 已知样本证据定位 | 联合正确 | 反事实成对正确 |','|---|---:|---:|---:|---:|---:|---:|']
 for name,m in methods.items():
  if m['status']!='complete':
   lines.append(f"| {name} | 未完成 | — | — | — | — | — |")
  else:
   ev='不适用' if m['known_evidence_accuracy'] is None else f"{m['known_evidence_correct']}/{m['known_evidence_n']} ({m['known_evidence_accuracy']:.2%})"
   joint='不适用' if m['joint_accuracy'] is None else f"{m['joint_correct']}/{m['n']} ({m['joint_accuracy']:.2%})"
   lines.append(f"| {name} | {m['correct']}/{m['n']} | {m['accuracy']:.2%} | {m['macro_f1']:.4f} | {ev} | {joint} | {m['counterfactual_pairs_correct']}/{m['counterfactual_pair_n']} |")
 lines.extend(['','trialmatch_model_only 为纯 TrialMatchAI；trialmatch 另含确定性数值比较。本集没有数值阈值，因此两者预期使用相同条件判断。纯 TrialMatchAI 不生成定位证据，相关指标不适用。','',f"逐例失败：`{args.output/'errors.jsonl'}`。逐类错误、混淆矩阵、分场景表现、耗时见 `metrics.json`。",'', '证据指标只评价短合成事实的定位；未测试长病历检索能力。无临床专家复核。第三方树许可证待明确，原文及动作只保留服务器，公开提供哈希、位置、标签和重建脚本。'])
 (args.output/'REPORT.zh.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps({'status':'evaluated','methods':{k:{'status':v['status'],'accuracy':v.get('accuracy')} for k,v in methods.items()}},ensure_ascii=False))

if __name__=='__main__':main()
