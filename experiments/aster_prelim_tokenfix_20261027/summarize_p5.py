from pathlib import Path
import collections,csv,json,shutil,statistics
from run_p1 import dump
BASE=Path(__file__).resolve().parent

def main():
 res=BASE/'results/p5';res.mkdir(parents=True,exist_ok=True);summary=[];states={};gates={};green=[];massrows=[]
 for name in ['qwen','llama']:
  src=BASE/'private'/('p5_'+name);log=BASE/'private'/('p5_'+name+'.log')
  if log.exists():shutil.copyfile(log,res/log.name)
  if not (src/'DONE.json').exists():states[name]='not_completed';continue
  states[name]='completed';records=[json.loads(s) for s in (src/'records.jsonl').read_text().splitlines()];groups=collections.defaultdict(list)
  for r in records:groups[r['variant'],r['method']].append(r)
  metrics={}
  for (variant,method),rs in groups.items():
   acc=100*sum(r['correct'] for r in rs)/len(rs);metrics[variant,method]=acc
   summary.append({'model':name,'volume':variant,'method':method,'n':len(rs),'accuracy_pct':acc,'mean_seconds':statistics.mean(r['seconds'] for r in rs)})
   masses=[m for r in rs for m in r['attention_mass']]
   if masses:massrows.append({'model':name,'volume':variant,'method':method,'case_count':sum(bool(r['attention_mass']) for r in rs),'applicable_mass':statistics.mean(m['applicable_mass'] for m in masses),'competitor_mass':statistics.mean(m['competitor_mass'] for m in masses),'applicable_conditional_share':statistics.mean(m['applicable_conditional_share'] for m in masses)})
  finite=['gate_0.5','gate_1','gate_2','gate_4','gate_8','gate_16'];nonzero=finite+['gate_hard']
  # Fixed SW+AC macro evaluation; retain all volumes and strengths in the table.
  best=max(finite,key=lambda m:statistics.mean(metrics[v,m] for v in ['SW','AC']))
  gaps={v:metrics[v,'gold_prune']-metrics[v,best] for v in ['SW','AC']}
  hardgaps={v:metrics[v,'gold_prune']-metrics[v,'gate_hard'] for v in ['SW','AC']}
  gates[name]={'best_development_gate':best,'prune_minus_best_gate_pp':gaps,'prune_minus_hard_gate_pp':hardgaps,'gate_matches_prune_within_2pp':all(x<=2 for x in hardgaps.values()) and all(x<=2 for x in gaps.values()),'finite_gate_matches_prune_within_2pp':all(x<=2 for x in gaps.values()),'hard_gate_includes_position_compaction':True,'selection_is_development_only':True}
  green.append({'model':name,'AC_gate_8_accuracy_pct':metrics['AC','gate_8'],'AC_gate_8_green_accuracy_pct':metrics['AC','gate_8_green']})
  with (res/(name+'_raw_predictions.jsonl')).open('w') as f:
   for r in records:
    f.write(json.dumps({k:v for k,v in r.items() if k!='attention_mass'})+'\n')
  # Preserve full per-layer measured mass on the server, export a separate numerical file.
  (res/(name+'_attention_mass.jsonl')).write_text(''.join(json.dumps({k:r[k] for k in ['model','id','tree','variant','method','attention_mass']})+'\n' for r in records if r['attention_mass']))
  shutil.copyfile(src/'manifest.json',res/(name+'_manifest.json'))
 for fn,rs in [('summary.csv',summary),('attention_summary.csv',massrows)]:
  if rs:
   with (res/fn).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
 complete=all(v=='completed' for v in states.values());passed=complete and all(g['gate_matches_prune_within_2pp'] for g in gates.values())
 gate={'P5':'PASS_PRUNE_GATE' if passed else 'FAIL_STOP' if complete else 'INCOMPLETE','models':states,'gold_gate_comparison':gates,'green_AC':green,'P6':'NOT_RUN','test_inference_count':0};dump(res/'gate_status.json',gate)
 lines=['# P5 金标执行通道实测','',f'状态：**{gate["P5"]}**。只在两模型 P4 全部通过后运行。','','| 模型 | 卷 | 方法 | n | 准确率 % |','|---|---|---|---:|---:|']
 for r in summary:lines.append(f"| {r['model']} | {r['volume']} | {r['method']} | {r['n']} | {r['accuracy_pct']:.2f} |")
 lines+=['','金标裁剪门槛：`'+json.dumps(gates,ensure_ascii=False)+'`。','','- 主裁剪为普通文本裁剪、自然位置重排与原生因果注意力；隔离裁剪另外列出，不替代主对照。','- 门控采用概率下限 1e-4，报告全部强度与硬门控；硬门控额外压缩位置编号，另存不压缩位置的硬门控对照；有限强度不压缩位置，继续门槛要求硬门控及 SW/AC 平均准确率选出的同一个有限强度门控，在 SW、AC 上分别不低于主裁剪超过 2 个百分点。','- 投影使用匹配模型的既有官方 SEKA 投影，实施 k ← k−γPk，γ∈{0.5,1}。这是算子诊断，不是完整 SEKA 标准任务复现，也不能推广为所有投影方法更弱。','- 有界强调对金标规则的编号、条件和处置加 +1，层段取 P3 开发选择；没有更改病例证据。','- 注意力份额仅在事先固定的 20 个 SW 开发病例记录，从真实 Q/K 与掩码用 FP32 重算，逐层原值另存。不同强度会改变上游表示，端到端非单调不能冒充固定 Q/K 解析性质的反例或证明。','- 全部数据是开发数据，含本轮参数选择，不作为独立测试。P6 仍需其独立数据有效性与人工核验门槛。']
 if complete and not passed:lines+=['','**P5 未通过金标裁剪门槛，停止进入 P6。需进一步定位位置、阶段编码或信息读取差异，不能写成通道能力已追平裁剪。**']
 (BASE/'P5_REPORT.zh.md').write_text('\n'.join(lines)+'\n');print(json.dumps(gate))
if __name__=='__main__':main()
